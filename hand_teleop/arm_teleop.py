#!/usr/bin/env python3
"""
Webカメラ + MediaPipe Pose(3Dワールド座標)で人間の腕の動きに SO-101 を連動させる。
(arm shadowing: 人間の骨格の角度→ロボットの関節を1:1対応)

MediaPipe Pose の pose_world_landmarks(メートル単位の推定3D座標)を使うので、
カメラに対してどの向きに腕を曲げても角度が取れる(2D方式の弱点を解消)。

  人間の測る量                        -> SO-101 の関節
  手首の左右位置(肩基準, 画面内2D)    -> 関節0: 土台の回転
  上腕の傾き(肩→肘の仰角, 3D)        -> 関節1: shoulder_lift
  肘の曲げ角(上腕と前腕のなす角)      -> 関節2: elbow_flex
  手首の曲げ(前腕と手のひらのなす角)  -> 関節3: wrist_flex ※Hand検出時
  親指と人差し指のピンチ                -> 関節5: グリッパー ※Hand検出時

追跡する腕は「よく見えている方」を自動選択(HUDに R/L 表示)。
各角度は「r で記録した中立姿勢」からの差分で動く。

前提: phosphobot run --no-cameras で起動しておくこと。

操作:
    q          終了(アームをスリープ姿勢へ)
    スペース    一時停止/再開
    r          いまの腕の姿勢を「中立」として記録(起動後まず必ず1回押す)
"""
import os
import sys
import glob
import time
import math
import argparse

import cv2
import numpy as np
import requests
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL = os.path.join(HERE, "pose_landmarker_lite.task")
HAND_MODEL = os.path.join(HERE, "hand_landmarker.task")

# ----------------------------- CONFIG -----------------------------
CONFIG = {
    "phosphobot_url": "http://localhost",
    "robot_id": 0,
    "send_hz": 15,

    # 中立姿勢(rad)[土台, 肩, 肘, 手首ピッチ, 手首ロール, グリッパー]
    "neutral": [0.0, 0.5, -0.2, -0.3, 0.0, 1.1],

    # --- ゲイン ---
    # 左右: 手首の画面内x位置(肩基準, 正規化)の差分 -> 土台[rad]
    "gain_pan": 3.0,
    # 以下は人間の角度変化1度 → ロボット関節の変化(rad)。0.0175 ≒ 1:1
    # 上下が逆に動く場合は gain_shoulder の符号を反転
    "gain_shoulder": -0.0175,
    "gain_elbow": 0.0175,
    # 肘の曲げ伸ばし→肩(j1)への連動。肘を伸ばすと腕全体が前に出て
    # リーチが伸びる(ロボットは肘単独では前後がほぼ出ないため)
    "gain_reach": 0.020,
    "gain_wrist": 0.0175,

    "use_wrist": True,           # 手首制御(Hand検出が必要)
    "wrist_compensation": False, # Trueなら手首は常に水平(gain_wrist無視)

    # グリッパー(Hand検出時): ピンチ正規化 -> 開閉
    "pinch_close": 0.25,
    "pinch_open": 0.7,
    "grip_closed": 0.0,
    "grip_open": 2.2,

    # 関節リミット(安全クランプ, rad)
    "joint_limits": [
        (-1.5, 1.5),   # 0 土台
        (-0.2, 1.3),   # 1 肩
        (-1.2, 0.6),   # 2 肘
        (-1.6, 1.6),   # 3 手首ピッチ
        (-1.5, 1.5),   # 4 手首ロール
        (-0.1, 2.3),   # 5 グリッパー
    ],

    "max_step_rad": 0.12,   # 1サイクルの最大関節変化
    "smoothing": 0.25,      # EMA係数(3D推定はノイジーなので強め)
    "deadzone_deg": 3.0,    # 角度差分の微動無視[deg]
    "deadzone_pan": 0.02,   # 左右(正規化座標)の微動無視

    "cam_index": 0,
    "cam_width": 1280,
    "cam_height": 720,
}
# ------------------------------------------------------------------

# BlazePose ランドマーク番号 [肩, 肘, 手首]
SIDE_IDX = {"L": (11, 13, 15), "R": (12, 14, 16)}
POSE_CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]


def _load_jp_font(size=24):
    """macOSの日本語フォントを探す(なければNone=英語フォールバック)"""
    pats = ["/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Hiragino*",
            "/System/Library/Fonts/Supplemental/ヒラギノ*"]
    for pat in pats:
        for path in sorted(glob.glob(pat)):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


_JP_FONT = _load_jp_font(24)
_JP_FONT_S = _load_jp_font(19)


def draw_neutral_guide(frame):
    """中立姿勢のお手本(人型)と R キーの案内を描画"""
    h, w = frame.shape[:2]
    bw, bh = 560, 300
    x0, y0 = w // 2 - bw // 2, h // 2 - bh // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # --- 棒人間(鏡像表示に合わせ、向かって右側の腕をお手本にする) ---
    fx, fy = x0 + 40, y0 + 30          # 人型の描画原点
    C = (255, 255, 255)                 # 体
    A = (80, 220, 255)                  # お手本の腕(黄色系)
    # 頭・体
    cv2.circle(frame, (fx + 60, fy + 30), 22, C, 3)
    cv2.line(frame, (fx + 60, fy + 52), (fx + 60, fy + 160), C, 3)
    cv2.line(frame, (fx + 20, fy + 75), (fx + 100, fy + 75), C, 3)   # 肩
    # 反対の腕(下ろしたまま・グレー)
    cv2.line(frame, (fx + 20, fy + 75), (fx + 8, fy + 140), (130, 130, 130), 3)
    # お手本の腕: 上腕やや前方(斜め下) → 肘90〜120度で前腕を上げる
    sh = (fx + 100, fy + 75)
    el = (fx + 145, fy + 120)
    wr = (fx + 175, fy + 60)
    cv2.line(frame, sh, el, A, 5)
    cv2.line(frame, el, wr, A, 5)
    cv2.circle(frame, el, 7, A, -1)
    # 開いた手のひら
    cv2.circle(frame, (wr[0] + 6, wr[1] - 10), 12, A, 3)
    for ang in (-60, -30, 0, 30):
        rad = math.radians(ang - 60)
        cv2.line(frame, (wr[0] + 6, wr[1] - 10),
                 (int(wr[0] + 6 + 22 * math.cos(rad)),
                  int(wr[1] - 10 + 22 * math.sin(rad))), A, 2)
    # 肘の角度表示
    cv2.ellipse(frame, el, (26, 26), 0, -65, -160, (0, 255, 255), 2)

    # --- テキスト(人型の右側) ---
    lines = [("この姿勢をマネして", False),
             ("ひじ90〜120度・手のひらをカメラへ", False),
             ("", False),
             ("★ R キーで中立を記録!", True)]
    if _JP_FONT is not None:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        ty = y0 + 60
        for ln, em in lines:
            font = _JP_FONT if em else _JP_FONT_S
            color = (255, 220, 80) if em else (255, 255, 255)
            d.text((x0 + 265, ty), ln, font=font, fill=color)
            ty += 40 if em else 34
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, "Mimic this pose, palm to camera", (x0 + 265, y0 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'R' to set neutral!", (x0 + 265, y0 + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2)
    return frame


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def v_sub(a, b):
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def v_angle(u, v):
    """3Dベクトル間の角度[deg]"""
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(a * a for a in v))
    return math.degrees(math.acos(clamp(dot / (nu * nv + 1e-9), -1.0, 1.0)))


class PhosphoClient:
    def __init__(self, base, robot_id):
        self.base = base.rstrip("/")
        self.rid = robot_id
        self.sess = requests.Session()

    def _post(self, path, json=None, timeout=1.0):
        return self.sess.post(f"{self.base}{path}?robot_id={self.rid}",
                              json=json, timeout=timeout)

    def robot_connected(self):
        try:
            r = self.sess.get(f"{self.base}/status", timeout=2)
            return len(r.json().get("robots", [])) > 0
        except Exception:
            return False

    def init(self):
        return self._post("/move/init", json={}, timeout=5.0)

    def sleep(self):
        try:
            return self._post("/move/sleep", json={}, timeout=5.0)
        except Exception:
            pass

    def read_joints(self):
        r = self._post("/joints/read", json={})
        if r.status_code != 200:
            raise RuntimeError(f"joints/read failed: {r.text[:80]}")
        return r.json().get("angles")

    def write_joints(self, angles):
        return self._post("/joints/write",
                          json={"angles": angles, "unit": "rad"}, timeout=0.8)


def pick_side(img_lms):
    """よく見えている方の腕を選ぶ(手首の visibility で判定)"""
    vis_l = getattr(img_lms[15], "visibility", 0.0) or 0.0
    vis_r = getattr(img_lms[16], "visibility", 0.0) or 0.0
    return "L" if vis_l >= vis_r else "R"


def extract_features(world_lms, img_lms, hand_lms, side):
    """3Dワールド座標から制御用の角度[deg]を抽出。
    world座標系: x=画面右, y=下, z=奥行き(腰原点・メートル)"""
    i_sh, i_el, i_wr = SIDE_IDX[side]
    sh, el, wr = world_lms[i_sh], world_lms[i_el], world_lms[i_wr]

    # 肘の曲げ角(3D): 伸ばすと180、直角で90
    elbow = v_angle(v_sub(sh, el), v_sub(wr, el))

    # 上腕の仰角: 真下=0, 水平=90, 真上=180
    u = v_sub(el, sh)
    elevation = v_angle(u, (0.0, 1.0, 0.0))   # y+ は下向き

    # 左右: 画面内の手首x位置(肩基準, 正規化座標)。3D方位角は腕を下ろした
    # 姿勢で不安定なため、安定な2Dを使う
    i_wr2 = SIDE_IDX[side][2]
    pan = img_lms[i_wr2].x - img_lms[i_sh].x

    # 手首の曲げ(2D: 前腕 vs 手のひら)とピンチ(Hand検出時のみ)
    wrist = None
    pinch = None
    if hand_lms is not None:
        iel, iwr = SIDE_IDX[side][1], SIDE_IDX[side][2]
        el2, wr2 = img_lms[iel], img_lms[iwr]
        pts = hand_lms
        fore = (wr2.x - el2.x, wr2.y - el2.y)
        palm = (pts[9].x - pts[0].x, pts[9].y - pts[0].y)
        d = fore[0] * palm[0] + fore[1] * palm[1]
        nf = math.hypot(*fore)
        npm = math.hypot(*palm)
        wrist = math.degrees(math.acos(
            clamp(d / (nf * npm + 1e-9), -1.0, 1.0)))
        hand_size = math.hypot(pts[9].x - pts[0].x, pts[9].y - pts[0].y)
        pinch_raw = math.hypot(pts[4].x - pts[8].x, pts[4].y - pts[8].y)
        pinch = pinch_raw / (hand_size + 1e-6)

    return {"pan": pan, "elevation": elevation, "elbow": elbow,
            "wrist": wrist, "pinch": pinch}


def dz(delta, deadzone):
    """デッドゾーン付き差分"""
    if abs(delta) < deadzone:
        return 0.0
    return delta - math.copysign(deadzone, delta)


def compute_target(feat, base, cfg):
    """特徴量 feat と 中立基準 base から目標関節角(6)を計算"""
    n = list(cfg["neutral"])
    d = cfg["deadzone_deg"]

    # 左右: 手首の画面内x位置差分(鏡像表示なので手を右へ→画面上も右へ)
    dpan = feat["pan"] - base["pan"]
    if abs(dpan) < cfg["deadzone_pan"]:
        dpan = 0.0
    j0 = n[0] - cfg["gain_pan"] * dpan

    # 肩: 上腕の仰角差分(上げ下げ) + 肘の曲げ伸ばし連動(前後リーチ)
    delb = dz(feat["elbow"] - base["elbow"], d)
    j1 = (n[1]
          + cfg["gain_shoulder"] * dz(feat["elevation"] - base["elevation"], d)
          + cfg["gain_reach"] * delb)

    # 肘: 曲げ角差分(伸ばす=180に近づく=ロボットの肘も伸ばす=負方向)
    j2 = n[2] - cfg["gain_elbow"] * delb

    if cfg["wrist_compensation"] or not cfg["use_wrist"] \
            or feat["wrist"] is None or base.get("wrist") is None:
        j3 = -(j1 + j2) if cfg["wrist_compensation"] else n[3]
    else:
        j3 = n[3] + cfg["gain_wrist"] * dz(feat["wrist"] - base["wrist"], d)

    j4 = n[4]

    if feat["pinch"] is not None:
        t = clamp((feat["pinch"] - cfg["pinch_close"])
                  / (cfg["pinch_open"] - cfg["pinch_close"] + 1e-9), 0.0, 1.0)
        j5 = cfg["grip_closed"] + t * (cfg["grip_open"] - cfg["grip_closed"])
    else:
        j5 = n[5]

    target = [j0, j1, j2, j3, j4, j5]
    return [clamp(v, lo, hi)
            for v, (lo, hi) in zip(target, cfg["joint_limits"])]


def draw(frame, img_lms, hand_lms, side):
    h, w = frame.shape[:2]
    if img_lms:
        pxs = {i: (int(img_lms[i].x * w), int(img_lms[i].y * h))
               for i in [11, 12, 13, 14, 15, 16]}
        active = set(SIDE_IDX[side])
        for a, b in POSE_CONNECTIONS:
            col = (255, 180, 0) if a in active and b in active else (120, 120, 120)
            cv2.line(frame, pxs[a], pxs[b], col, 3)
        for i, p in pxs.items():
            cv2.circle(frame, p, 6,
                       (0, 0, 255) if i in active else (100, 100, 100), -1)
    if hand_lms:
        for p in hand_lms:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (0, 255, 0), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CONFIG["phosphobot_url"])
    ap.add_argument("--cam", type=int, default=CONFIG["cam_index"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    client = PhosphoClient(args.url, cfg["robot_id"])
    current = None

    if not args.dry_run:
        if not client.robot_connected():
            print("[エラー] phosphobot にロボットが接続されていません!")
            print("  電源/USB を確認し、phosphobot run --no-cameras を再起動")
            sys.exit(1)
        try:
            client.init()
            print("[init] ロボットを初期化しました")
            time.sleep(2.5)
            start = client.read_joints() or [0.0] * 6
            for i in range(1, 21):
                t = i / 20.0
                client.write_joints([(1 - t) * s + t * nn
                                     for s, nn in zip(start, cfg["neutral"])])
                time.sleep(0.08)
            current = list(cfg["neutral"])
            print("[init] 中立姿勢へ移行しました")
        except Exception as e:
            print(f"[警告] init 失敗: {e}")

    for path in (POSE_MODEL, HAND_MODEL):
        if not os.path.exists(path):
            print(f"[エラー] モデルが見つかりません: {path}")
            sys.exit(1)

    pose = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=0.6))
    hand = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1, min_hand_detection_confidence=0.6))

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["cam_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["cam_height"])
    if not cap.isOpened():
        print(f"[エラー] カメラ {args.cam} を開けません。"
              "phosphobot を --no-cameras で起動しているか確認。")
        sys.exit(1)

    paused = False
    base = None
    sm = None       # スムージング済み特徴量
    side = "R"
    locked_side = None   # 中立記録時に追跡する腕を固定
    a = cfg["smoothing"]
    last_send = 0.0
    send_interval = 1.0 / cfg["send_hz"]

    print("=== arm_teleop 開始(3D腕トラッキング) ===")
    print(f" 送信先: {args.url}  dry_run={args.dry_run}")
    print(" ★上半身がカメラに入る距離で、腕を楽な基準姿勢にして 'r' を押す")
    print(" q=終了  スペース=一時停止/再開  r=中立記録")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)
            pres = pose.detect_for_video(img, ts)
            hres = hand.detect_for_video(img, ts)

            img_lms = pres.pose_landmarks[0] if pres.pose_landmarks else None
            world_lms = (pres.pose_world_landmarks[0]
                         if pres.pose_world_landmarks else None)
            hand_lms = hres.hand_landmarks[0] if hres.hand_landmarks else None

            feat = None
            if img_lms and world_lms:
                # 中立記録前は自動選択、記録後はその腕に固定(切替暴れ防止)
                if locked_side is None:
                    side = pick_side(img_lms)
                else:
                    side = locked_side
                feat = extract_features(world_lms, img_lms, hand_lms, side)
                if sm is None:
                    sm = dict(feat)
                    if sm["wrist"] is None:
                        sm["wrist"] = 0.0
                    if sm["pinch"] is None:
                        sm["pinch"] = 0.5
                else:
                    for k in ("pan", "elevation", "elbow"):
                        sm[k] = a * feat[k] + (1 - a) * sm[k]
                    if feat["wrist"] is not None:
                        sm["wrist"] = a * feat["wrist"] + (1 - a) * sm["wrist"]
                    if feat["pinch"] is not None:
                        sm["pinch"] = a * feat["pinch"] + (1 - a) * sm["pinch"]

            draw(frame, img_lms, hand_lms, side)

            now = time.time()
            if feat and base and not paused and not args.dry_run \
                    and (now - last_send) >= send_interval:
                sfeat = {"pan": sm["pan"], "elevation": sm["elevation"],
                         "elbow": sm["elbow"],
                         "wrist": sm["wrist"] if feat["wrist"] is not None else None,
                         "pinch": sm["pinch"] if feat["pinch"] is not None else None}
                target = compute_target(sfeat, base, cfg)
                try:
                    if current is None:
                        current = client.read_joints() or list(cfg["neutral"])
                    step = cfg["max_step_rad"]
                    current = [c + clamp(t - c, -step, step)
                               for c, t in zip(current, target)]
                    client.write_joints(current)
                except requests.exceptions.Timeout:
                    pass
                except Exception as e:
                    print(f"[send err] {e}")
                last_send = now

            # HUD
            status = "PAUSED" if paused else ("DRY" if args.dry_run else "LIVE")
            color = (0, 165, 255) if paused else (0, 255, 0)
            if base is None:
                frame = draw_neutral_guide(frame)
            if not feat:
                cv2.putText(frame, "no arm detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif sm:
                lock_mark = "" if locked_side else "?"
                cv2.putText(frame,
                            f"[{side}{lock_mark}] pan={sm['pan']:+.2f} "
                            f"elev={sm['elevation']:.0f} elbow={sm['elbow']:.0f} "
                            f"wrist={('%.0f' % sm['wrist']) if feat['wrist'] is not None else '-'} "
                            f"hand={'o' if hand_lms else 'x'}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            shown = current if current else cfg["neutral"]
            cv2.putText(frame,
                        f"[{status}] j0={shown[0]:+.2f} j1={shown[1]:+.2f} "
                        f"j2={shown[2]:+.2f} j3={shown[3]:+.2f} g={shown[5]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not args.dry_run and now - globals().get("_lc", 0) > 5.0:
                globals()["_lc"] = now
                globals()["_dc"] = not client.robot_connected()
            if globals().get("_dc"):
                cv2.putText(frame, "ROBOT DISCONNECTED!", (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            cv2.imshow("SO-101 arm teleop 3D (q/space/r)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"[{'一時停止' if paused else '再開'}]")
            elif key == ord('r'):
                if feat and sm:
                    locked_side = side   # 以後この腕に固定
                    base = {"pan": sm["pan"],
                            "elevation": sm["elevation"],
                            "elbow": sm["elbow"],
                            "wrist": sm["wrist"] if feat["wrist"] is not None else None}
                    print(f"[中立記録] 腕={side} pan={base['pan']:+.2f} "
                          f"elev={base['elevation']:.0f} elbow={base['elbow']:.0f} "
                          f"wrist={base['wrist']}")
                else:
                    print("[中立記録] 腕が検出されていません")
    finally:
        print("終了処理: アームをスリープ姿勢へ...")
        if not args.dry_run:
            client.sleep()
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        hand.close()


if __name__ == "__main__":
    main()

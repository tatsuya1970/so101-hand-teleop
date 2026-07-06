#!/usr/bin/env python3
"""
Webカメラ + MediaPipe Hands で人間の手の動きに SO-101 を連動させる。

制御方式: 関節空間ダイレクトマッピング(phosphobot /joints/write 経由)。
実機検証の結果、この個体では /move/absolute の IK が左右(y)を解けないため、
関節を直接動かす方式を採用。人間の腕の動きと直感的に一致する。

  - 手の画面内の左右位置          -> 関節0(土台の回転)
  - 手の大きさ(=カメラとの距離)  -> 関節1(肩の前後)
  - 手の画面内の上下位置          -> 関節2(肘の上下)
  - 手首の水平維持                -> 関節3 = -(関節1+関節2) で自動補正
  - 親指と人差し指の開き(ピンチ) -> 関節5(グリッパー, 0=閉 2.2rad=開)

前提: phosphobot を --no-cameras で起動しておく(カメラをこのスクリプトが使うため)。
    phosphobot run --no-cameras

操作:
    q          終了(終了時にアームをスリープ姿勢へ)
    スペース    一時停止/再開(トグル)。停止中はアームは動かない
    r          現在の手の位置を「原点(中立)」として再センタリング

チューニング: 下の CONFIG を編集。まずは小さいスケールから。
"""
import os
import sys
import time
import math
import argparse

import cv2
import numpy as np
import requests
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# ----------------------------- CONFIG -----------------------------
CONFIG = {
    "phosphobot_url": "http://localhost",
    "robot_id": 0,

    # 送信レート(Hz)
    "send_hz": 15,

    # --- 中立姿勢(rad)。手が画面中央・中距離のときの姿勢 ---
    # [土台, 肩, 肘, 手首ピッチ, 手首ロール, グリッパー]
    # 肩をやや前傾+肘をやや伸ばし、少し前に構えた姿勢
    "neutral": [0.0, 0.5, -0.2, -0.3, 0.0, 1.1],

    # --- 各軸の振り幅(rad)。中立からの最大変化 ---
    "range_pan": 0.9,       # 左右(土台) ±0.9rad ≈ ±51°
    "range_shoulder": 0.45, # 前後(肩)
    "range_elbow": 0.55,    # 上下(肘)

    # 各軸の向き反転(実機で逆だったら True/False を入れ替える)
    "invert_pan": False,
    "invert_shoulder": False,
    "invert_elbow": False,

    # 手首ピッチ自動補正(グリッパーを水平に保つ)。実測: EEピッチ = -(j1+j2+j3)
    "wrist_compensation": True,

    # グリッパー角(rad): 実測 0=閉, 2.2=開
    "grip_closed": 0.0,
    "grip_open": 2.2,

    # --- 関節可動リミット(安全クランプ, rad) ---
    "joint_limits": [
        (-1.5, 1.5),   # 0 土台
        (-0.2, 1.3),   # 1 肩
        (-1.2, 0.6),   # 2 肘
        (-1.6, 1.6),   # 3 手首ピッチ
        (-1.5, 1.5),   # 4 手首ロール
        (-0.1, 2.3),   # 5 グリッパー
    ],

    # 1サイクルあたりの最大関節変化量(rad)。急激な動きを抑制
    "max_step_rad": 0.12,

    # 手の大きさ(手首-中指付け根の距離,正規化)-> 前後にマップする基準
    # 画面の size= 表示を見て調整(near=近づけた時, far=遠ざけた時)
    "hand_size_near": 0.32,
    "hand_size_far": 0.14,

    # ピンチ距離(親指-人差し指,手サイズで正規化)-> グリッパー
    "pinch_close": 0.25,
    "pinch_open": 0.7,

    # スムージング(EMA)係数。0に近いほど滑らかだが遅延増。0.2〜0.5
    "smoothing": 0.35,

    # デッドゾーン(中立付近の微動を無視,正規化-1..1に対して)
    "deadzone": 0.04,

    # カメラ
    "cam_index": 0,
    "cam_width": 1280,
    "cam_height": 720,
}
# ------------------------------------------------------------------

# MediaPipe Tasks API の手のランドマーク接続(描画用)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # 親指
    (0, 5), (5, 6), (6, 7), (7, 8),            # 人差し指
    (5, 9), (9, 10), (10, 11), (11, 12),       # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),     # 薬指
    (13, 17), (17, 18), (18, 19), (19, 20),    # 小指
    (0, 17),                                   # 手のひら
]


def draw_landmarks(frame, landmarks):
    """Tasks API のランドマーク(正規化座標)を描画"""
    h, w = frame.shape[:2]
    pxs = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pxs[a], pxs[b], (0, 200, 0), 2)
    for (px, py) in pxs:
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp_inv(v, a, b):
    """a->0, b->1 の線形補間(範囲外はクランプ)"""
    if abs(b - a) < 1e-9:
        return 0.0
    return clamp((v - a) / (b - a), 0.0, 1.0)


def apply_deadzone(v, dz):
    """v in [-1,1]。中立付近を殺し、外側を滑らかに残す"""
    if abs(v) < dz:
        return 0.0
    s = 1.0 if v > 0 else -1.0
    return s * (abs(v) - dz) / (1.0 - dz)


class PhosphoClient:
    def __init__(self, base, robot_id):
        self.base = base.rstrip("/")
        self.rid = robot_id
        self.sess = requests.Session()

    def _post(self, path, json=None, timeout=1.0):
        url = f"{self.base}{path}?robot_id={self.rid}"
        return self.sess.post(url, json=json, timeout=timeout)

    def init(self):
        return self._post("/move/init", json={}, timeout=5.0)

    def sleep(self):
        try:
            return self._post("/move/sleep", json={}, timeout=5.0)
        except Exception:
            pass

    def robot_connected(self):
        """phosphobotにロボットが接続されているか確認"""
        try:
            r = self.sess.get(f"{self.base}/status", timeout=2)
            return len(r.json().get("robots", [])) > 0
        except Exception:
            return False

    def read_joints(self):
        r = self._post("/joints/read", json={})
        if r.status_code != 200:
            raise RuntimeError(f"joints/read failed: {r.text[:80]}")
        return r.json().get("angles")

    def write_joints(self, angles):
        return self._post("/joints/write",
                          json={"angles": angles, "unit": "rad"},
                          timeout=0.8)


def compute_target_joints(nx, ny, fwd, open01, cfg):
    """正規化入力(-1..1, -1..1, -1..1, 0..1)から目標関節角(6)を計算"""
    n = list(cfg["neutral"])
    sgn = lambda inv: -1.0 if inv else 1.0

    j0 = n[0] + sgn(cfg["invert_pan"]) * (-nx) * cfg["range_pan"]
    j1 = n[1] + sgn(cfg["invert_shoulder"]) * fwd * cfg["range_shoulder"]
    # 画面の上(nyマイナス)=上げたい=肘を伸ばす(マイナス)方向
    j2 = n[2] + sgn(cfg["invert_elbow"]) * ny * cfg["range_elbow"]
    if cfg["wrist_compensation"]:
        j3 = -(j1 + j2)          # グリッパーを常に水平に
    else:
        j3 = n[3]
    j4 = n[4]
    j5 = cfg["grip_closed"] + open01 * (cfg["grip_open"] - cfg["grip_closed"])

    target = [j0, j1, j2, j3, j4, j5]
    return [clamp(v, lo, hi)
            for v, (lo, hi) in zip(target, cfg["joint_limits"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CONFIG["phosphobot_url"])
    ap.add_argument("--cam", type=int, default=CONFIG["cam_index"])
    ap.add_argument("--dry-run", action="store_true",
                    help="ロボットに送らずカメラ画面だけ確認")
    args = ap.parse_args()

    cfg = CONFIG
    client = PhosphoClient(args.url, cfg["robot_id"])
    current = None   # 直近に送った関節角

    if not args.dry_run:
        if not client.robot_connected():
            print("[エラー] phosphobot にロボットが接続されていません!")
            print("  1. SO-101 の電源アダプタと USB を確認")
            print("  2. phosphobot run --no-cameras を再起動")
            print("  3. ダッシュボード http://localhost で Status: Connected を確認")
            sys.exit(1)
        try:
            client.init()
            print("[init] ロボットを初期化しました")
            time.sleep(2.5)   # 初期化動作の完了を待つ
            # 中立姿勢へゆっくり移行(20ステップ補間)
            start = client.read_joints() or [0.0] * 6
            for i in range(1, 21):
                t = i / 20.0
                pose = [(1 - t) * s + t * n
                        for s, n in zip(start, cfg["neutral"])]
                client.write_joints(pose)
                time.sleep(0.08)
            current = list(cfg["neutral"])
            print("[init] 中立姿勢へ移行しました")
        except Exception as e:
            print(f"[警告] init 失敗: {e} (phosphobotが起動しているか確認)")

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["cam_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["cam_height"])
    if not cap.isOpened():
        print(f"[エラー] カメラ {args.cam} を開けません。"
              "phosphobot を --no-cameras で起動しているか確認してください。")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[エラー] モデルが見つかりません: {MODEL_PATH}")
        sys.exit(1)
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # 状態
    paused = False
    sm_nx = sm_ny = sm_fwd = 0.0
    sm_open = 0.5
    center_x, center_y = 0.5, 0.5   # 中立(再センタリングで更新)
    last_send = 0.0
    send_interval = 1.0 / cfg["send_hz"]
    a = cfg["smoothing"]

    print("=== hand_teleop 開始(関節ダイレクト制御) ===")
    print(" q=終了  スペース=一時停止/再開  r=再センタリング")
    print(f" 送信先: {args.url}  dry_run={args.dry_run}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)   # 鏡像(自分の手と同じ向きに)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            res = landmarker.detect_for_video(mp_image, ts_ms)

            hand_found = False
            hand_landmarks = res.hand_landmarks if res.hand_landmarks else None

            if hand_landmarks:
                hand_found = True
                pts = hand_landmarks[0]
                draw_landmarks(frame, pts)

                # 手の中心 = 手首(0)と中指付け根(9)の中点
                cx = (pts[0].x + pts[9].x) / 2.0
                cy = (pts[0].y + pts[9].y) / 2.0
                # 手の大きさ = 手首(0)-中指付け根(9)距離
                hand_size = math.hypot(pts[9].x - pts[0].x,
                                       pts[9].y - pts[0].y)
                # ピンチ = 親指先(4)-人差し指先(8)距離 / 手サイズ
                pinch = math.hypot(pts[4].x - pts[8].x,
                                   pts[4].y - pts[8].y)
                pinch_norm = pinch / (hand_size + 1e-6)

                # 正規化 -1..1(中立中心)
                nx = (cx - center_x) * 2.0
                ny = (cy - center_y) * 2.0
                nx = apply_deadzone(clamp(nx, -1, 1), cfg["deadzone"])
                ny = apply_deadzone(clamp(ny, -1, 1), cfg["deadzone"])

                # 前後 = 手の大きさ(near->+1, far->-1)
                fwd = lerp_inv(hand_size,
                               cfg["hand_size_far"],
                               cfg["hand_size_near"]) * 2.0 - 1.0

                # グリッパー: pinch_close以下->0(閉), pinch_open以上->1(開)
                open01 = lerp_inv(pinch_norm,
                                  cfg["pinch_close"],
                                  cfg["pinch_open"])

                # スムージング
                sm_nx = a * nx + (1 - a) * sm_nx
                sm_ny = a * ny + (1 - a) * sm_ny
                sm_fwd = a * fwd + (1 - a) * sm_fwd
                sm_open = a * open01 + (1 - a) * sm_open

                # HUD
                cv2.putText(frame, f"pinch={pinch_norm:.2f} size={hand_size:.2f}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

            # 送信
            now = time.time()
            target = compute_target_joints(sm_nx, sm_ny, sm_fwd, sm_open, cfg)
            if (hand_found and not paused and not args.dry_run
                    and (now - last_send) >= send_interval):
                try:
                    if current is None:
                        current = client.read_joints() or list(cfg["neutral"])
                    # 1サイクルの変化量を制限(安全)
                    step = cfg["max_step_rad"]
                    current = [c + clamp(t - c, -step, step)
                               for c, t in zip(current, target)]
                    client.write_joints(current)
                except requests.exceptions.Timeout:
                    pass   # 詰まったフレームは捨てる
                except Exception as e:
                    print(f"[send err] {e}")
                    cv2.putText(frame, f"send err: {e}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                last_send = now

            # ステータス表示
            status = "PAUSED" if paused else ("DRY" if args.dry_run else "LIVE")
            color = (0, 165, 255) if paused else (0, 255, 0)
            shown = current if current else target
            cv2.putText(frame,
                        f"[{status}] pan={shown[0]:+.2f} sh={shown[1]:+.2f} "
                        f"el={shown[2]:+.2f} grip={shown[5]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if not hand_found:
                cv2.putText(frame, "no hand", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 2秒ごとにターミナルへ状態を出力(デバッグ)
            if now - globals().get('_last_dbg', 0) > 2.0:
                globals()['_last_dbg'] = now
                shown_j = current if current else target
                print(f"[状態] hand={'○' if hand_found else '×'} "
                      f"{'PAUSED' if paused else 'LIVE'} "
                      f"pan={shown_j[0]:+.2f} sh={shown_j[1]:+.2f} "
                      f"el={shown_j[2]:+.2f} grip={shown_j[5]:.2f}")

            # 5秒ごとにロボット接続を監視(切断されたら大きく警告)
            if not args.dry_run and now - globals().get('_last_conn', 0) > 5.0:
                globals()['_last_conn'] = now
                if not client.robot_connected():
                    print("!!!! ロボットが切断されました。電源とUSBを確認して "
                          "phosphobot を再起動してください !!!!")
                    globals()['_disconnected'] = True
                else:
                    globals()['_disconnected'] = False
            if globals().get('_disconnected'):
                cv2.putText(frame, "ROBOT DISCONNECTED!", (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("SO-101 hand teleop (q=quit, space=pause, r=recenter)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"[終了] qキー(コード{key})が押されました")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"[{'一時停止' if paused else '再開'}]")
            elif key == ord('r'):
                if hand_landmarks:
                    pts = hand_landmarks[0]
                    center_x = (pts[0].x + pts[9].x) / 2.0
                    center_y = (pts[0].y + pts[9].y) / 2.0
                    print(f"[再センタリング] center=({center_x:.2f},{center_y:.2f})")
    finally:
        print("終了処理: アームをスリープ姿勢へ...")
        if not args.dry_run:
            client.sleep()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()

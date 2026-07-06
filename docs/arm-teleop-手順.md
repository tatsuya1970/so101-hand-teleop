# Macのカメラで腕をトラッキングして SO-101 を動かす手順

作成日: 2026-07-06
環境: MacBook Pro 2021 (M1 Pro) / macOS / SO-101(lerobot ロボットアーム)

## 概要

MacのWebカメラ(内蔵カメラでOK)で人間の腕の動きをトラッキングし、
SO-101 がその動きを真似する(arm shadowing)。追加ハードウェア不要・完全無料。

```
Webカメラ
  → MediaPipe Pose(骨格の3Dワールド座標) + Hand(手の21点)
  → 腕の関節角度を抽出(肩の傾き・肘の曲げ・手首・ピンチ)
  → SO-101 の関節角度に1:1マッピング
  → phosphobot REST API (/joints/write) で送信
  → SO-101 が腕の動きを再現
```

## 人間の体とSO-101モーターの対応

| 人間の動き(測る量) | SO-101 のモーター |
|---|---|
| 手首の左右位置(肩基準・画面内2D) | 関節0: 土台の回転 |
| 上腕の傾き=肩の上げ下げ(3D仰角) | 関節1: shoulder_lift |
| 肘の曲げ角(3D、上腕と前腕のなす角) | 関節2: elbow_flex(+肩に連動して前後リーチ) |
| 手首の曲げ(前腕と手のひらのなす角) | 関節3: wrist_flex |
| 親指と人差し指のピンチ | 関節5: グリッパー(つまむ=閉じる) |

ポイント:
- **関節の「位置」ではなく、骨の「向き」と関節の「曲げ角」を測る**
- 肘の曲げ角は MediaPipe Pose の **3Dワールド座標**(`pose_world_landmarks`)で計算
  するため、カメラの奥行き方向に曲げても角度が取れる(2Dだと180°に張り付く)
- 左右だけは3D方位角が不安定(腕を下ろすと暴れる)ため、安定な2D手首位置を使用
- 中立姿勢(`r`キーで記録)からの**差分**で動くので、体格やカメラ位置に依存しない

## セットアップ

### 1. phosphobot(ロボット制御サーバー)

[quest3-teleop-手順.md](quest3-teleop-手順.md) の手順1〜3参照。要点:

```bash
# インストール(Apple Silicon)
mkdir -p ~/.local/bin
curl -fsSL -o ~/.local/bin/phosphobot \
  "https://github.com/phospho-app/homebrew-phosphobot/releases/download/v0.3.134/phosphobot-0.3.134-arm64.bin"
chmod +x ~/.local/bin/phosphobot
# PATHを通して、SO-101をUSB+電源接続し、ダッシュボードでキャリブレーション
```

### 2. Python環境

```bash
cd ~/projects/so101
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe opencv-python numpy requests pillow

# MediaPipe モデル2つをダウンロード
curl -fsSL -o hand_teleop/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -fsSL -o hand_teleop/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
```

## 実行手順

### 1. phosphobot をカメラ無効で起動(ターミナル1)

```bash
phosphobot run --no-cameras
```

※ `--no-cameras` が重要。付けないと phosphobot がカメラを掴んで MediaPipe が使えない。

### 2. 腕トラッキングを起動(ターミナル2)

```bash
cd ~/projects/so101
source .venv/bin/activate
python hand_teleop/arm_teleop.py          # 本番
python hand_teleop/arm_teleop.py --dry-run  # ロボットに送らず画面確認だけ
```

### 3. 中立姿勢の記録(重要)

起動すると画面に人型のお手本が表示される:
- **ひじを90〜120度に曲げて、手を体の前に**
- 上腕はやや前方
- 手のひらをカメラに向ける(HUDが `hand=o` になる)
- 上半身全体がカメラに入る距離で

この姿勢で **`R` キー**を押すと中立が記録され、追従が始まる。
(ウィンドウをクリックしてフォーカスし、**英数モード**で押すこと)

### 4. 操作

| 腕の動き | ロボット |
|---|---|
| 腕全体を左右に振る | 土台が回転 |
| 上腕を上げ下げ | 肩が上下 |
| **肘を伸ばす/曲げる** | **前へリーチ/引っ込む** |
| 手首を曲げる | 手首 |
| つまむ/開く | グリッパー |

| キー | 動作 |
|---|---|
| `q` | 終了(アームをスリープ姿勢へ) |
| スペース | 一時停止/再開 |
| `r` | 中立を記録し直す(追跡する腕もこの時点で固定) |

## チューニング(arm_teleop.py の CONFIG)

| 設定 | 意味 | 調整の目安 |
|---|---|---|
| `gain_pan` | 左右の感度 | 3.0。大きいほど敏感 |
| `gain_shoulder` | 上下の感度 | -0.0175。**逆に動いたら符号反転** |
| `gain_reach` | 肘→前後リーチの効き | 0.020。前後が弱ければ0.03まで上げる |
| `gain_elbow` | 肘の感度 | 0.0175 ≒ 人間1度→ロボット1度 |
| `smoothing` | 滑らかさ | 0.25。小さいほど滑らか(遅延増) |
| `deadzone_deg` | 微動無視の角度 | 3.0。手ブレが伝わるなら増やす |

## つまずきポイント(実際にハマった順)

1. **mediapipe 0.10.35 は旧API(`mp.solutions`)廃止**
   → Tasks API(`PoseLandmarker`/`HandLandmarker`)+ `.task` モデルを使う

2. **2Dの肘角度はカメラの奥行き方向に曲げると取れない**(常に170°前後)
   → `pose_world_landmarks`(3D)で計算する

3. **追跡する腕を毎フレーム自動選択すると暴れる**(両腕が見えるとR/Lがパタパタ切替)
   → 中立記録(`r`)の瞬間に腕を固定

4. **3D方位角による左右制御は腕を下ろすと不安定**(水平成分ゼロで発散)
   → 左右だけ2D(画面内の手首x位置)を使うハイブリッドに

5. **ロボットの肘単独では前後リーチがほぼ出ない**(SO-101の幾何学的特性)
   → 肘の曲げ伸ばしを肩(j1)にも連動させる(`gain_reach`)

6. **ロボット未接続でも /joints/write は 200 OK**(サイレント失敗)
   → 起動時+実行中5秒ごとに /status の robots 配列を監視

7. **OpenCVは日本語を描画できない**
   → Pillow + macOSのヒラギノフォントで画面ガイドを描画

## 関連

- 手のひら版(第1弾): [hand_teleop/README.md](../hand_teleop/README.md)
- phosphobotセットアップ: [quest3-teleop-手順.md](quest3-teleop-手順.md)
- リポジトリ: https://github.com/tatsuya1970/so101-hand-teleop

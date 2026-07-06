# SO-101 Webカメラ手/腕トラッキング teleop

Webカメラで人間の手・腕を MediaPipe で追跡し、SO-101 を連動させる。

2つのバージョンがある:

| スクリプト | 方式 | 特徴 |
|---|---|---|
| `arm_teleop.py`(推奨) | **腕全体**の骨格を3Dトラッキング | 人間の関節→ロボットの関節を1:1対応(arm shadowing)。直感的 |
| `hand_teleop.py` | 手のひらの位置・大きさ | 手だけで操作。カメラに手だけ映せばよい |

腕版の詳しい手順: [docs/arm-teleop-手順.md](../docs/arm-teleop-手順.md)

## 仕組み

```
Webカメラ → MediaPipe Hands(21点) → 手の位置/大きさ/ピンチを抽出
        → 関節角度に直接マッピング → phosphobot /joints/write → SO-101
```

制御は**関節空間ダイレクトマッピング**(IK不使用)。実機検証の結果、
phosphobot の /move/absolute(IK)は左右(y)を解けなかったため、
関節を直接動かす方式にした。人間の腕の構造と対応するため直感的。

| 人間の動き | ロボットの関節 |
|---|---|
| 手を左右に動かす | 関節0: 土台の回転(肩の旋回に相当) |
| 手をカメラに近づける/遠ざける | 関節1: 肩の前後 |
| 手を上下に動かす | 関節2: 肘の曲げ伸ばし |
| (自動) | 関節3: 手首ピッチ = -(関節1+関節2) でグリッパー水平維持 |
| 親指と人差し指をつまむ/開く | 関節5: グリッパー(0=閉, 2.2rad=開) |

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe opencv-python numpy requests pillow

# MediaPipe のモデルをダウンロード(手 + 骨格)
curl -fsSL -o hand_teleop/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -fsSL -o hand_teleop/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
```

動作確認済み: Python 3.12 / mediapipe 0.10.35 / opencv-python 5.0 / macOS (M1)

## 実行手順

### 1. phosphobot をカメラ無効で起動

Webカメラをこのスクリプトが使うため、phosphobot にはカメラを握らせない。
**すでに `phosphobot run` している場合は Ctrl+C で止めてから**:

```bash
phosphobot run --no-cameras
```

### 2. teleop スクリプトを起動(別ターミナル)

```bash
cd ~/projects/so101
source .venv/bin/activate
python hand_teleop/hand_teleop.py
```

まず動きを確認したいだけ(ロボットに送らない)なら:

```bash
python hand_teleop/hand_teleop.py --dry-run
```

## 操作キー

| キー | 動作 |
|---|---|
| `q` | 終了(アームをスリープ姿勢へ) |
| スペース | 一時停止/再開(停止中はアーム静止) |
| `r` | いまの手の位置を中立(原点)として再センタリング |

## 安全に使うコツ

- 初回は必ず `--dry-run` で画面の数値(x/y/z/open)の動きを確認
- 本番でも最初は**スペースで一時停止した状態**で手を中立に置き、`r` で再センタリングしてから再開
- アームの周囲に物・手を置かない。異常時はスペースor`q`で停止
- 動きが速すぎ/敏感すぎるときは `hand_teleop.py` の `CONFIG` を調整

## チューニング(CONFIG)

`hand_teleop.py` 冒頭の `CONFIG` を編集:

- `range_x_cm / range_y_cm / range_z_cm` … 可動範囲。大きいほどダイナミックだが危険。まず小さく。
- `invert_x / invert_y / invert_z` … 軸の向きが逆なら True/False を反転
- `hand_size_near / hand_size_far` … 前後制御の感度。画面下部に出る `size=` の実測値を見て、
  手を近づけた時の値を near、遠ざけた時の値を far に設定
- `pinch_close / pinch_open` … グリッパーのしきい値。画面の `pinch=` を見て調整
- `smoothing` … 0に近いほど滑らかだが遅延増(0.2〜0.5)
- `deadzone` … 中立付近の微動無視。手ブレが気になれば増やす
- `send_hz` … 送信レート。カクつくなら下げる

## トラブル

- **カメラが開けない** → phosphobot が `--no-cameras` で起動しているか確認
- **send err / Timeout** → phosphobot が起動しているか、`http://localhost` にアクセスできるか確認
- **軸が逆** → CONFIG の `invert_*` を反転
- **動きが逆に敏感/鈍い** → `range_*_cm` と `hand_size_*` を調整

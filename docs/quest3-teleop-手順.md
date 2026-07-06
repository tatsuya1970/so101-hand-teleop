# SO-101 を Meta Quest 3 で操作する(phosphobot)セットアップ手順

作成日: 2026-07-06
環境: MacBook Pro 2021 (M1 Pro) / macOS

## 目的

SO-101 を人間の動きに連動させる。第1弾として **Meta Quest 3 のコントローラーの動きで SO-101 を動かす**(phosphobot 使用)。
第2弾として Webカメラ + MediaPipe の手トラッキングによる操作を自作予定。

前提: [lerobot での SO-101 セットアップと遠隔操作](https://qiita.com/tatsuya1970/items/3f04c9c6d21744190f41) は完了済み。

## 使用するソフトウェア

- [phosphobot](https://github.com/phospho-app/phosphobot) — SO-100/SO-101 対応のオープンソース制御ミドルウェア(Mac 上でサーバーとして動作)
- [phospho teleoperation](https://www.meta.com/experiences/phospho-teleoperation/8873978782723478/) — Meta Quest 用アプリ(Quest 2 / Pro / 3 / 3s 対応)

## 手順1: phosphobot のインストール(Mac)

公式の推奨は Homebrew:

```bash
brew tap phospho-app/phosphobot
brew install phosphobot
```

### つまずきポイント①: tap の信頼確認

新しめの Homebrew ではサードパーティ tap の信頼確認が必要。

```bash
brew trust phospho-app/phosphobot
```

### つまずきポイント②: Command Line Tools が古いとエラー

```
Error: Your Command Line Tools are too outdated.
```

と出る場合、本来は「システム設定 → 一般 → ソフトウェアアップデート」で
Command Line Tools を更新すればよい。

今回は更新を待たず、formula が参照しているビルド済みバイナリを直接ダウンロードして回避した:

```bash
mkdir -p ~/.local/bin
curl -fsSL -o phosphobot.bin \
  "https://github.com/phospho-app/homebrew-phosphobot/releases/download/v0.3.134/phosphobot-0.3.134-arm64.bin"
# チェックサム検証(formula 記載の sha256 と照合)
echo "c453049aaeb8c7bdb82bc381283b1d7410b5720b50a503e3b5e0ca937a0415f3  phosphobot.bin" | shasum -a 256 -c -
chmod +x phosphobot.bin
mv phosphobot.bin ~/.local/bin/phosphobot
```

※ Intel Mac の場合は `x86_64` 版バイナリを使う。最新バージョンは
[リリースページ](https://github.com/phospho-app/homebrew-phosphobot/releases) を確認。

### つまずきポイント③: `command not found: phosphobot`

`~/.local/bin` が PATH に入っていないと `zsh: command not found: phosphobot` になる。
`~/.zshrc` に以下を追記して反映する:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

インストール確認:

```bash
phosphobot --version
# => phosphobot 0.3.134
```

## 手順2: Quest 3 側の準備

1. Quest 3 の Meta Store で **「phospho teleoperation」** アプリをインストール
2. Quest 3 と Mac を **同じ Wi-Fi ネットワーク** に接続(必須)

## 手順3: SO-101 の接続と phosphobot サーバー起動

1. SO-101 **フォロワーアーム** に電源アダプタを接続(リーダーアームは不要)
2. USB ケーブルで Mac に接続
3. デバイス認識を確認:

```bash
ls /dev/tty.usb* /dev/cu.usb*
```

4. phosphobot サーバーを起動:

```bash
phosphobot run
```

5. ブラウザでダッシュボードを開く: `http://localhost` (ポート80) または `http://localhost:8020`
   - ダッシュボード上で SO-101 が認識されていることを確認
   - 必要に応じてキャリブレーションを実施

## 手順4: Quest 3 から操作

1. Quest 3 で phospho teleoperation アプリを起動
2. 接続先一覧に「phosphobot」または Mac のコンピュータ名が表示されるので、トリガーで選択して接続
3. 操作方法:
   - **A ボタン**: テレオペレーション開始/停止(コントローラーの動きにアームが追従)
   - **トリガー**: グリッパーの開閉
   - **B ボタン**: 録画(データセット記録)の開始/停止
   - **グリップボタン**: ウィンドウの移動

## 進捗状況

- [x] phosphobot インストール(v0.3.134)
- [x] SO-101 接続・ダッシュボード確認・キャリブレーション
- [x] Quest 3 からのテレオペ → **中止**: アプリがMeta Storeで「購入できません」
      +「VR操作はphospho proサブスク必須」と判明したため
- [x] (第2弾)Webカメラ + MediaPipe 手トラッキング自作 → **成功!**
      (`hand_teleop/hand_teleop.py`、詳細は `hand_teleop/README.md`)

## 追加のつまずきポイント集

### ④ キャリブレーションのポーズは3D画像と完全一致させる
向きがズレると軸が反転する(前後・上下が逆になった)。
グリッパーは完全に閉じ、可動爪は土台から見て左、アームは水平前方。

### ⑤ モーター通信が突然切れる(2回発生)
症状: ダッシュボードは表示されるが /status の temperature が null →
そのうち robots: [] に。原因は電源アダプタの接触不良が濃厚。
復旧: 電源・USB を挿し直して phosphobot 再起動。

### ⑥ phosphobot の IK は左右(y)を解けなかった
/move/absolute の x(前後)・z(上下)・open は動くが y(左右)が無反応。
→ 自作スクリプトでは /joints/write による関節ダイレクト制御を採用。
関節ID指定(joints_ids)は1始まり。全関節一括書き込みは index 0 = 土台。

### ⑦ mediapipe 0.10.35 は旧API(mp.solutions)廃止
新しい Tasks API(HandLandmarker)+ hand_landmarker.task モデルを使う。

### ⑧ ロボット未接続でも /joints/write は 200 OK を返す
エラーにならないので「動かないのに成功している」ように見える。
/status の robots 配列で接続確認が必要。

## 参考リンク

- [phosphobot GitHub](https://github.com/phospho-app/phosphobot)
- [phospho 公式ドキュメント: VR Control with a Meta Quest](https://docs.phospho.ai/examples/teleop)
- [phospho インストールドキュメント](https://docs.phospho.ai/installation)
- [解説動画: SO-100 and SO-101 VR control with a Meta Quest!](https://www.youtube.com/watch?v=AQ-xgCTdj_w)

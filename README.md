# 🐰 Eve AI VTuber + VLM 統合システム

**人間のような視覚認識を備えた、画面共有型 AI VTuber アプリケーション**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Groq](https://img.shields.io/badge/Groq-Llama_4-f55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5.4_mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Object_Detection-00FFFF?style=for-the-badge)](https://docs.ultralytics.com/)

[Portfolio 07（Eve AI）](https://github.com/kmtkzy1379/portfolio7-AI.git)に **9段ステージの画面認識パイプライン（VLM）** を統合。
ユーザーの画面を Eve と共有しながら自然に会話できるアプリケーションです。

&nbsp;

## ✨ Portfolio 07 からの進化点

| 課題（Portfolio 07） | 解決（本プロジェクト） |
|:---|:---|
| ゲーム実況モードで OCR がボトルネック | 9段ステージ VLM パイプラインで画面全体をリアルタイム認識 |
| 画面の内容を把握できない（テキストベースのみ） | 3層の視覚認識メカニズムで画面を常時把握 |
| 画像認識は手動トリガーのみ | MAJOR変化を自動検出し、Eve が自発的に声をかける |

&nbsp;

## 👁️ 3層の視覚認識メカニズム

人間の視覚的注意（選択的注意・不随意的注意）を参考に設計した3層構造:

| 層 | 名称 | 動作 |
|:---:|:---|:---|
| 層1 | **パッシブ認識** | VisionBuffer → タグ付きコンテキストをシステムプロンプトに毎回注入 |
| 層2 | **アクティブ認識** | LLM Function Calling → `get_screen_info` / `get_screen_image` / `set_watch_mode` |
| 自動 | **自動プッシュ** | MAJOR変化 → 会話に自動割り込み（nudge） |

### 監視モード (set_watch_mode)

- 応答AIが Function Calling で画面監視モードを ON/OFF
- ON: MODERATE も nudge 対象 / OFF: MAJOR のみ（デフォルト）
- ベースライン方式: 開始時の画面状態をスナップショットとして保存し、同一画面の微動を排除
- プログレッシブクールダウン: nudge 回数に応じてクールダウンを延長

&nbsp;

## 🔄 データフロー

### Eve 応答フロー

```
ユーザー入力 → BaseMode.process_input()
  → RAG検索 + ConversationCache直近5ターン取得
  → LLMHandler.generate_stream() [ストリーミング]
    システムプロンプト動的再構築:
      base + [AI2 Feedback] + [VLM] + [Vision Tools] + [RAG] + [Recent]
    文区切り(。！？!\n)でyield → TTSバッチ(3文並列) → AudioPlayer.queue
  → FeedbackHandler.process_pending() トリガー (非同期バックグラウンド)
```

### VLM 画面認識フロー（9段ステージ）

```
ScreenCapture(5fps)
  → ChangeDetector (pHash + SSIM 2段階)
  → YOLOv11m 物体検出
  → BoT-SORT + Re-ID 追跡
  → Optical Flow 動き分析
  → PerIDAnalyzer 個別分析
  → SceneGraph シーン構築
  → DeltaEncoder 差分圧縮
  → NarrationEngine (Llama 4 Scout via litellm) ナレーション生成
```

### VLMBridge nudge 判定フロー

```
ナレーション受信
  → _classify_change() でタグ決定 (MAJOR/MODERATE/MINOR/NONE)
  → VisionBufferに追加（全フレーム）
  → nudge条件:
    1. has_change: ナレーションに「変化なし」を含まない
    2. should_nudge: MAJOR, または MODERATE かつ watch_mode
    3. 重複チェック: 前回nudgeナレーションと類似度 < 0.60
    4. クールダウン: 前回nudgeから15秒以上 (プログレッシブ: 15s→30s→60s)
    5. watch_mode時: ベースラインナレーションと異なる
  → すべて通過 → auto_push_callback発火 → Eveの会話に割り込み
```

### 変化検出の仕組み (pHash + SSIM 2段階)

```
Stage 1 (高速): pHash hamming distance
  hamming < phash_threshold(16) → NONE（処理スキップ）

Stage 2 (SSIM確認):
  SSIM < 0.60 → MAJOR（画面が大きく変化、ウィンドウ切替レベル）
  SSIM < 0.85 → MODERATE（中程度の変化）
  SSIM >= 0.85 → MINOR（微小変化）

参照フレーム: MAJOR/MODERATEで更新
定期チェック: 40フレーム連続NONEで強制MODERATE
```

&nbsp;

## ⚙️ スレッドモデル

| スレッド | 役割 | 備考 |
|:---|:---|:---|
| メインスレッド | Tkinter UI イベントループ | UI更新は `root.after(0, fn)` |
| スレッド2 | asyncio イベントループ (Eve モード全処理) | `threading.Thread(daemon=True)` |
| スレッド3 (VLM) | ScreenCapture → frame_queue | VLM ON時のみ |
| スレッド4 (VLM) | メイン処理ループ (検出→追跡→分析→集約) | VLM ON時のみ |
| スレッド5 (VLM) | LLM ナレーション (litellm 呼び出し) | VLM ON時のみ |
| 別プロセス | vts.py (VTube Studio WebSocket 制御) | subprocess |

&nbsp;

## 🏗️ アーキテクチャ

```
portfolio8-VLM-AI/
├── app.py              # メインエントリ: asyncioループ(別スレッド) → Tkinter UI起動
├── run.py              # 統合ランチャー: VTS+VOICEVOX起動 → vts.py子プロセス → app.py
├── launcher.py         # 外部アプリ起動 (同期版)
├── vts.py              # VTube Studio WebSocket制御 (別プロセス)
├── config.py           # .env読み込み, Configクラス
├── config/
│   └── vlm_default.yaml  # VLMパイプライン設定
├── ui/
│   └── main_window.py  # Tkinter MainWindow (モード選択/VLMトグル/ログ)
├── modes/
│   ├── base_mode.py    # BaseMode(ABC): 共通初期化/process_input/TTSバッチ
│   ├── talk_mode.py    # TalkMode: マイク→STT→LLM→TTS
│   ├── game_mode.py    # GameMode: キー入力(OCR/Vision)→画像解析
│   └── youtube_mode.py # YouTubeMode: Live Chat API→LLM→TTS
├── modules/
│   ├── llm.py          # LLMHandler: OpenAI/Groq, ストリーミング, Function Calling
│   ├── tts.py          # TTSHandler: VOICEVOX HTTP API
│   ├── player.py       # AudioPlayer: PyAudio再生, asyncio.Queue, interrupt対応
│   ├── audio_input.py  # AudioInput: Silero VAD, 16kHz/mono
│   ├── stt.py          # STTHandler: Groq whisper-large-v3
│   ├── rag.py          # RAGHandler: OpenAI embedding, JSONL永続化, cosine検索
│   ├── feedback.py     # FeedbackHandler: GPT-4o自己フィードバック(FEP)
│   ├── conversation_cache.py # ConversationCache: deque(100), JSONL永続化
│   ├── vlm_bridge.py   # VLMBridge: Pipeline→別スレッド実行, ルールベース分類
│   ├── vision_buffer.py # VisionBuffer: deque(20)リングバッファ, スレッドセーフ
│   └── vision_analyzer.py # VisionAnalyzer: Groq Llama 4 Maverick画像解析
├── vlm/                # VLMパッケージ (画面認識パイプライン)
│   ├── main.py         # Pipeline: 3スレッド(capture/main/llm), 9段ステージ処理
│   ├── capture/        # screen.py(mss), change_detector.py(pHash+SSIM)
│   ├── detection/      # yolo_detector.py (YOLOv11m)
│   ├── tracking/       # id_authority.py(BoT-SORT+Re-ID), track_store.py
│   ├── analysis/       # per_id_analyzer.py, optical_flow.py, pose.py(MediaPipe), expression.py(DeepFace)
│   ├── aggregation/    # delta_encoder.py, scene_graph.py, token_budget.py
│   ├── narration/      # llm_client.py(litellm), prompt_builder.py
│   └── common/         # config.py(YAML), datatypes.py, device.py
└── prompts/            # システムプロンプト定義 (talk/game/youtube)
```

&nbsp;

## 🔧 技術スタック

| カテゴリ | 技術 | 用途 |
|:---:|:---|:---|
| 🤖 **応答AI** | `OpenAI API` — GPT-5.4 mini | メイン会話生成 / Function Calling |
| 👁️ **VLMナレーション** | `Groq API` — Llama 4 Scout (litellm経由) | 画面認識→自然言語ナレーション |
| 🖼️ **画像解析** | `Groq API` — Llama 4 Maverick | ネイティブマルチモーダル画像認識 |
| 🧪 **フィードバック** | `OpenAI API` — GPT-4o | 自己フィードバック (FEP) |
| 🎤 **STT** | `Groq API` — Whisper large-v3 | 音声認識 |
| 📎 **Embedding** | `OpenAI API` — text-embedding-3-small | RAG ベクトル埋め込み |
| 🔊 **TTS** | `VOICEVOX` — ローカル HTTP API | 音声合成 |
| 🔇 **VAD** | `Silero VAD` — PyTorch | 音声区間検出 |
| 🎯 **物体検出** | `YOLOv11m` + onnxruntime | 画面上の物体検出 |
| 🏷️ **物体追跡** | `supervision` — BoT-SORT + Re-ID | フレーム間のエンティティ追跡 |
| 🦴 **姿勢推定** | `MediaPipe` | 人物の姿勢推定 |
| 😊 **表情分析** | `DeepFace` + tf-keras | 表情分類 |
| 📷 **画面認識基盤** | `OpenCV` / `mss` / `ImageHash` / `scikit-image` | Optical Flow / キャプチャ / pHash+SSIM |
| 🎭 **アバター** | `VTube Studio` — WebSocket API | Live2D アバター制御 |
| 🖥️ **UI** | `Tkinter` | デスクトップ GUI |

&nbsp;

## 🛡️ 設計判断

### 誤検出の抑制
- Live2Dキャラのアイドルアニメーションはナレーション重複検出で抑制
- YOLO class_whitelist から "tv" と "monitor" を除外（VTuberオーバーレイの誤検出防止）
- min_entity_lifetime と min_box_area で一時的ノイズを排除

### ナレーション重複検出
- `difflib.SequenceMatcher` で前回 nudge ナレーションとの類似度を計算
- ratio > 0.60 で重複と判定
- MAJOR 変化は重複チェックをバイパス

### 応答AIの懐疑的判断
- システムプロンプトで VLM 認識の不完全さを明示
- 前回と今回のナレーションを比較して本当に変化したか判断するよう指示

&nbsp;

## 🚀 セットアップ

### 📋 前提条件

| 必須 | 任意 |
|:---|:---|
| Python 3.13+ | [VTube Studio](https://denchisoft.com/) |
| [VOICEVOX](https://voicevox.hiroshiba.jp/) | YouTube API キー |
| Groq API キー | |
| OpenAI API キー | |

---

### 1️⃣ リポジトリのクローン

```bash
git clone https://github.com/kmtkzy1379/portfolio8-VLM-AI.git
cd portfolio8-VLM-AI
```

### 2️⃣ 仮想環境の作成

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4️⃣ 環境変数の設定

`.env.example` をコピーして `.env` ファイルを作成:

```bash
cp .env.example .env
```

<details>
<summary>📄 .env の設定項目一覧（クリックで展開）</summary>

&nbsp;

```env
# ─── 必須 ───────────────────────────────────
OPENAI_API_KEY=sk-...          # GPT-5.4 mini (応答AI), GPT-4o (フィードバック), Embeddings
GROQ_API_KEY=gsk_...           # Llama 4 Scout/Maverick (VLM), Whisper (STT)

# ─── YouTube ライブモード用（任意）──────────
YOUTUBE_API_KEY=your_youtube_api_key_here
TARGET_CHANNEL_ID=your_channel_id_here

# ─── VOICEVOX 設定 ─────────────────────────
VOICEVOX_URL=http://127.0.0.1:50021
VOICEVOX_SPEAKER_ID=8
VOICEVOX_SPEED=1.3
VOICEVOX_PITCH=0.0

# ─── 外部アプリのパス（任意）────────────────
VTS_PATH=C:\path\to\VTube Studio.lnk
VOICEVOX_PATH=C:\path\to\VOICEVOX.lnk
```

</details>

### 5️⃣ 外部アプリの起動

- **VOICEVOX** を起動（TTS に必要）
- **VTube Studio** を起動（アバター連携を使う場合）

> 統合ランチャー (`run.py`) を使えば VOICEVOX と VTube Studio を自動で起動できます

### 6️⃣ アプリケーションの実行

```bash
# UI版（推奨）
python app.py

# 統合ランチャー（VOICEVOX + VTube Studio を自動起動）
python run.py
```

&nbsp;

## 🔗 関連リポジトリ

| # | プロジェクト | リンク |
|:---:|:-----------|:-------|
| 01 | FEP模倣AItuberアプリケーション | [portfolio1-AItuber](https://github.com/kmtkzy1379/portfolio1-AItuber.git) |
| 07 | Eve AI（マルチモード統合システム） | [portfolio7-AI](https://github.com/kmtkzy1379/portfolio7-AI.git) |
| 08 | Eve + VLM 統合版（本リポジトリ） | [portfolio8-VLM-AI](https://github.com/kmtkzy1379/portfolio8-VLM-AI.git) |
| -- | VLMパイプライン単体パッケージ | [vlm](https://github.com/kmtkzy1379/vlm.git) |

&nbsp;

## 📄 ライセンス

[MIT License](LICENSE) — Copyright (c) 2026 KMTKZY

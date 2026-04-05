# Eve AI VTuber + VLM 統合システム

## ディレクトリ構成
```
marge/
├── app.py              # メインエントリ: asyncioループ(別スレッド) → Tkinter UI起動
├── run.py              # 統合ランチャー: VTS+VOICEVOX起動 → vts.py子プロセス → app.py
├── launcher.py         # 外部アプリ起動 (同期版, os.startfile + wait_for_voicevox)
├── vts.py              # VTube Studio WebSocket制御 (別プロセス, ws://localhost:8001)
├── config.py           # .env読み込み, Configクラス (APIキー/VOICEVOX/VAD設定)
├── .env                # 環境変数 (GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, パス等)
├── config/
│   └── vlm_default.yaml  # VLMパイプライン設定 (キャプチャ/検出/追跡/ナレーション)
├── ui/
│   └── main_window.py  # Tkinter MainWindow (モード選択/VLMトグル/ログ2面)
├── modes/
│   ├── base_mode.py    # BaseMode(ABC): 共通初期化/process_input/TTS並列バッチ
│   ├── talk_mode.py    # TalkMode: マイク→STT→LLM→TTS, 無言8秒で「…」送信
│   ├── game_mode.py    # GameMode: Pキー(OCR)/Oキー(Vision) GPT-4o画像解析
│   └── youtube_mode.py # YouTubeMode: Live Chat API→LLM→TTS, 再生完了待ちループ
├── modules/
│   ├── llm.py          # LLMHandler: OpenAI/Groq, ストリーミング, Function Calling (Vision tools)
│   ├── tts.py          # TTSHandler: VOICEVOX HTTP API (audio_query→synthesis)
│   ├── player.py       # AudioPlayer: PyAudio再生, asyncio.Queue, interrupt対応
│   ├── audio_input.py  # AudioInput: Silero VAD, 16kHz/mono, 最小発話300ms
│   ├── stt.py          # STTHandler: Groq whisper-large-v3
│   ├── rag.py          # RAGHandler: OpenAI embedding, JSONL永続化, cosine検索
│   ├── feedback.py     # FeedbackHandler: GPT-4o AI2自己フィードバック(FEP)
│   ├── conversation_cache.py # ConversationCache: deque(100), JSONL永続化
│   ├── vlm_bridge.py   # VLMBridge: Pipeline→別スレッド実行, ルールベース分類, VisionBuffer統合
│   ├── vision_buffer.py # VisionBuffer: deque(20)リングバッファ, VisionFrame(change_tag), スレッドセーフ
│   └── vision_analyzer.py # VisionAnalyzer: Groq Llama 4 Maverick画像解析
├── vlm/                # VLMパッケージ (画面認識パイプライン)
│   ├── main.py         # Pipeline: 3スレッド(capture/main/llm), 9段ステージ処理
│   ├── capture/        # screen.py(mss), change_detector.py(pHash+SSIM), predictive_coder.py, saliency.py
│   ├── detection/      # yolo_detector.py (YOLOv11m), base.py
│   ├── tracking/       # id_authority.py(BoT-SORT+Re-ID), track_store.py, working_memory.py
│   ├── analysis/       # optical_flow.py, per_id_analyzer.py, pose.py(MediaPipe), expression.py(DeepFace)
│   ├── aggregation/    # delta_encoder.py, scene_graph.py, feature_store.py, token_budget.py
│   ├── narration/      # llm_client.py(litellm), prompt_builder.py, context_manager.py
│   └── common/         # config.py(YAML), datatypes.py, device.py, validators.py
└── prompts/            # システムプロンプト定義 (talk/game/youtube)
```

## スレッドモデル
| スレッド | 役割 | 備考 |
|---|---|---|
| メインスレッド | Tkinter UIイベントループ | UI更新は `root.after(0, fn)` 必須 |
| スレッド2 | asyncioイベントループ (Eveモード全処理) | `threading.Thread(daemon=True)` |
| スレッド3 (VLM) | ScreenCapture → frame_queue | VLM ON時のみ |
| スレッド4 (VLM) | メイン処理ループ (検出→追跡→分析→集約) | VLM ON時のみ |
| スレッド5 (VLM) | LLMナレーション (litellm呼び出し) | VLM ON時のみ |
| 別プロセス | vts.py (VTube Studio WebSocket制御) | run.pyがsubprocessで起動 |

## 現状のデータフロー

### Eve応答
```
ユーザー入力 → BaseMode.process_input()
  → RAG検索 (1.5sタイムアウト) + ConversationCache直近5ターン取得
  → LLMHandler.generate_stream() [ストリーミング]
    システムプロンプト毎回動的再構築: base + [AI2 Feedback] + [VLM] + [Vision Tools] + [RAG] + [Recent]
    文区切り(。！？!\n)でyield → TTSバッチ(3文並列) → AudioPlayer.queue
  → FeedbackHandler.process_pending() トリガー (非同期バックグラウンド)
```

### VLM画面認識
```
ScreenCapture(5fps) → frame_queue(2) → ChangeDetector → YOLO → IDAuthority
  → OpticalFlow → PerIDAnalyzer → SceneGraph → DeltaEncoder
  → narration_queue(4) → NarrationEngine(litellm) → narration_result_queue(8)

VLMBridge._custom_drain() [ルールベース分類]:
  narration + change_level → _classify_change() → change_tag (major/moderate/minor/none)
  → VisionBuffer.add(VisionFrame) [全フレーム蓄積]
  → nudge判定: MAJOR常時 or MODERATE(watch_mode時) → auto_push_callback

層1 (パッシブ): VisionBuffer.get_scene_journal(n=5) → タグ付きコンテキストをシステムプロンプトに毎回注入
層2 (アクティブ): LLM Function Calling → get_screen_info / get_screen_image / set_watch_mode
自動プッシュ: MAJOR変化 → TalkMode._vision_push_queue → process_input()

監視モード (set_watch_mode tool):
  応答AIがtool callで画面監視モードをON/OFF
  ON: MODERATEもnudge対象 / OFF: MAJORのみ(デフォルト)
  「画面変わったら教えて」→ set_watch_mode(enabled=true)
  「もういいよ」→ set_watch_mode(enabled=false)
```

### VLM変化検出の仕組み (vlm/capture/change_detector.py)
```
Stage 1 (高速): pHash hamming distance
  hamming < phash_threshold(16) → NONE（処理スキップ）

Stage 2 (SSIM確認、Stage 1で変化検出時のみ):
  SSIM < 0.60 → MAJOR（画面が大きく変化、ウィンドウ切替レベル）
  SSIM < 0.85 → MODERATE（中程度の変化）
  SSIM >= 0.85 → MINOR（微小変化）

参照フレーム: MAJOR/MODERATEで更新
定期チェック: 40フレーム連続NONEで強制MODERATE
```

### VLMBridge nudge判定フロー (modules/vlm_bridge.py)
```
ナレーション受信
  → _classify_change() でタグ決定
  → VisionBufferに追加（全フレーム）
  → nudge条件:
    1. has_change: ナレーションに「変化なし」を含まない
    2. should_nudge: (MAJOR) OR (MODERATE かつ watch_mode)
    3. 重複チェック: 前回nudgeしたナレーションと類似度 < 0.60
    4. クールダウン: 前回nudgeから15秒以上経過
    5. watch_mode時: ベースラインナレーションと異なる
  → すべて通過 → auto_push_callback発火
```

## 利用モデル一覧

| 役割 | モデル | 備考 |
|---|---|---|
| 応答AI (テキスト+Function Calling) | gpt-5.4-mini (OpenAI) | ストリーミング, tool_call対応, max_completion_tokens |
| 応答AI (画像解析) | Groq Llama 4 Maverick | ネイティブマルチモーダル, Function Calling経由 |
| VLMナレーション | litellm Groq Llama 4 Scout | 裏側定期処理 |
| AI2フィードバック | OpenAI GPT-4o | 深い分析用 |
| OCR (GameMode) | OpenAI GPT-4o | VLM OFF時 |
| Vision (GameMode) | Maverick (VLM ON) / GPT-4o (VLM OFF) | 自動切替 |
| STT | Groq whisper-large-v3 | |
| Embedding (RAG) | OpenAI text-embedding-3-small | |
| TTS | VOICEVOX localhost:50021 | |

### Groq Llama 4 Scout (VLMナレーション)
- モデル名: `meta-llama/llama-4-scout-17b-16e-instruct`
- アーキテクチャ: MoE, 17Bアクティブ, 16エキスパート
- コンテキスト: 128Kトークン
- 用途: VLMパイプラインのナレーション生成 (litellm経由)

### Groq Llama 4 Maverick (画像解析)
- モデル名: `meta-llama/llama-4-maverick-17b-128e-instruct`
- アーキテクチャ: MoE, 17Bアクティブ, 128エキスパート
- **ネイティブマルチモーダル**: 画像入力対応 (最大5枚/リクエスト, base64 max 4MB)
- **Function Calling/Tool Use**: 対応 (並列tool_callも可)
- 用途: modules/vision_analyzer.py でのスクリーンショット画像解析

### GPT-5.4 mini (応答AI)
- モデル名: `gpt-5.4-mini`
- 用途: 応答AI (テキスト生成 + Function Calling)
- **注意**: `max_tokens`ではなく`max_completion_tokens`を使用
- コンテキスト: GPT-5.4ファミリー

## VLM精度に関する設計判断

### 誤検出の抑制
- Live2Dキャラのアイドルアニメーション（揺れ・瞬き）は変化検出で捕捉されるが、ナレーション重複検出で抑制する
- YOLO class_whitelist から "tv" と "monitor" を除外（VTuberオーバーレイの誤検出防止）
- "person" は残す（コンテンツ視聴時の実人物検出に必要）
- min_entity_lifetime と min_box_area で一時的ノイズを排除

### Watch Modeの設計
- ベースライン方式: watch mode開始時の画面状態をスナップショットとして保存
- ベースラインと類似したナレーションはnudge対象外（同じ画面の微動を排除）
- プログレッシブクールダウン: nudge回数に応じてクールダウンを延長（15s → 30s → 60s）
- nudge成功後にベースラインを更新（次の変化を検出可能に）

### ナレーション重複検出
- `difflib.SequenceMatcher` で前回nudgeナレーションとの類似度を計算
- ratio > 0.60 で重複と判定（Live2Dの微妙な言い換えを捕捉）
- MAJOR変化は重複チェックをバイパス（画面が明確に変わった場合は常にnudge）

### 応答AIの懐疑的判断
- システムプロンプトでVLM認識の不完全さを明示
- 前回と今回のナレーションを比較して本当に変化したか判断するよう指示
- キャラクターの同一性に関する誤認防止指示

## 重要な設計制約
- Eve = asyncioベース / VLM = threadingベース → VLMBridge が橋渡し
- LLMHandler.generate_stream() 毎回システムプロンプト動的再構築 (history[0]上書き)
- LLMHandler.history 最大10ターン (system + 直近9)
- VLMナレーション 1秒間隔制限 (min_interval_seconds), batch_frames=1
- VLM Pipeline停止: `_stop_event.set()` + `_SENTINEL`送信 + `thread.join()`
- AudioPlayer.interrupt() でバージイン (interrupt_signal → キュー全クリア)
- FeedbackHandler: 起動10秒後に初回, 以降 process_pending() トリガー

## 環境・依存
- Python 3.13.3 / venv
- **要更新**: litellm 実環境1.82.0 → requirements.txtは1.82.3+指定済み、`pip install -U litellm` で更新
- APIキー: GROQ_API_KEY, OPENAI_API_KEY (2つで全機能動作。GEMINI_API_KEYは.envでコメントアウト中)
- 使い分け: 応答AI=OpenAI gpt-5.4-mini, VLMナレーション=Groq Scout(litellm経由), 画像解析=Groq Maverick

## 作業ルール
- コード変更前に必ずCLAUDE.mdを確認
- UI更新は root.after(0, fn) 必須
- VLM ON/OFFはモード実行中でも切替可能
- 1フェーズで複数ファイル変更時はファイル単位で順次
- 日本語コメント推奨（コードベースの慣習に合わせる）

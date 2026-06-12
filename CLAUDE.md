# CLAUDE.md

## Project Overview

### Purpose

Eve AI VTuber + VLM 統合システム。マイク・YouTube ライブチャット・ノベルゲーム OCR を入力源にした応答 AI（Eve）と、画面共有を 9 段ステージで自然言語ナレーション化する VLM パイプラインを統合し、Eve が画面の変化を自発的に nudge（割り込み発話）するデスクトップアプリケーション。VTube Studio 連携で Live2D アバターも制御する。製品仕様の詳細は `README.md` を参照（一部古いのでコードを正とする）。

## Tech Stack

- Language: Python（コメント日本語、識別子英語）
- Runtime: Python 3.13
- Framework: Tkinter (UI), asyncio (モード処理), threading (VLM パイプライン)
- Database: なし。永続化は JSONL / TXT のフラットファイル（リポジトリ root に保存）
- Testing: Phase 1 で `tools/` にテストハーネスを整備（下記「テスト方法」）。Tier-1（API不要・決定論）+ Tier-2（実LLM・ヘッドレス）。UI 上の手動確認も併用
- Linter/Formatter: 設定なし
- Package Manager: pip + `requirements.txt`

主要な外部 API: OpenAI / Groq / Anthropic / VOICEVOX (HTTP `127.0.0.1:50021`) / VTube Studio (WebSocket)。主要パッケージ: `groq`, `openai`, `anthropic`, `litellm`, `torch`, `ultralytics`, `mediapipe`, `deepface`, `mss`, `websockets`。

## Commands

- Install: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
- Run (full auto): `python run.py` — VTube Studio + VOICEVOX を自動起動 → `vts.py` を別プロセス起動 → `app.py` を呼ぶ
- Run (UI only): `python app.py` — 外部アプリは起動済みの前提
- Test (Tier-1, API不要): `$env:PYTHONIOENCODING="utf-8"; venv\Scripts\python.exe tools\test_*_phase1.py`（各ファイルが PASS/FAIL と合計を表示）
- Test (Tier-2, 実LLM): `venv\Scripts\python.exe tools\tier2_phase1.py <N>`（N回×シナリオ。要 `.env` キー・コスト発生）
- Build / Typecheck / Lint / Format: 設定なし

## Project Structure

- `app.py`: メインエントリ（Config 検証 → asyncio loop を別スレッド起動 → Tkinter `MainWindow` 起動）
- `run.py`: 統合ランチャー（外部アプリ + `vts.py` を起動して `app.main()` を呼ぶ）
- `launcher.py`: VTube Studio / VOICEVOX 起動と VOICEVOX HTTP API 待機
- `vts.py`: VTube Studio WebSocket 制御（`run.py` から **別プロセス** で起動される）
- `config.py`: `.env` 読み込みと `Config` クラス、`Config.validate()` で起動時 fail-fast
- `modules/`: 応答 AI 側のロジック（LLM / TTS / STT / RAG / 会話キャッシュ / 画面ブリッジ など）
- `modes/`: モード実装（`base_mode.py` を継承した `talk_mode` / `game_mode` / `youtube_mode`）
- `vlm/`: 画面認識パイプライン（capture / detection / tracking / analysis / aggregation / narration）
- `ui/main_window.py`: Tkinter UI（モード選択・VLM トグル・ログ表示）
- `prompts/`: モード別システムプロンプト（`talk_prompt.py` / `game_prompt.py` / `youtube_prompt.py`）
- `config/`: VLM 用 YAML（`vlm_default.yaml`, `botsort_reid.yaml`）
- `eve/`: **編集禁止**（下記参照）
- `vlm_orig/`: **編集禁止**（下記参照）

## Architecture Decisions

### 編集禁止領域（最優先で確認）

grep / Glob のヒットがこの配下なら **編集対象から除外**:

- `eve/`: Portfolio 07 のレガシースナップショット。**独自の `.git` / `venv` / `modules/` を持つ別プロジェクト**。`feedback.py` や `talk_mode.py` と同名のファイルが多数あるが、現行 `app.py` からは import されない。grep ヒットだけで「ここを直せばいい」と判断しないこと
- `vlm_orig/`: `vlm/` パッケージのアップストリーム。独自の `pyproject.toml` / `README.md` / `tests/` を持つ。**触らない**
- `venv/`, `__pycache__/`: 生成物
- `memory/`: Claude のメモリ用ディレクトリ。コード変更とは別系統

### スレッド／プロセスモデル

- メインスレッド: Tkinter UI。**UI 更新は `root.after(0, fn)` 経由**（asyncio スレッドから直接 widget を触らない）
- 別スレッド: asyncio loop（モード処理の本体）
- 別スレッド × 複数: VLM パイプライン（capture / main / narration の 3 スレッド構成）
- 別プロセス: `vts.py`

### I/O とエラーハンドリング

- I/O は asyncio + 非同期書き込みキューが基本。`modules/rag.py` / `modules/conversation_cache.py` の write queue パターンを踏襲する
- 例外は `logger.error(...)` + 安全な `None` / フォールバック返却。突然落とさない
- 各モジュール冒頭で `logger = logging.getLogger(__name__)`

### 起動時の前提

- `Config.validate()` が `app.py` 起動時に呼ばれ、`GROQ_API_KEY` / `OPENAI_API_KEY` 未設定なら `ValueError` で fail-fast
- VOICEVOX が `127.0.0.1:50021` で生きていない場合は警告だけで続行（TTS は機能しない）

## Working Rules

- 不明点・矛盾・仕様未確定事項がある場合は、実装前に質問する。
- Tech Stack、DB、認証方式、ディレクトリ構成を推測で決めない。
- 既存コード・README・設定ファイルから明確に判断できる場合は、それを根拠として進める。
- 実装後は可能な範囲でテスト・型チェック・lintを実行する（このリポジトリではテスト・lint は未整備のため、UI 上で動作確認する）。
- フォーマットやlintは、手作業ではなく既存のツール設定に従う。
- セキュリティ上重要な変更、破壊的変更、DBマイグレーション、外部API仕様変更は事前に確認する。
- README.md と現状コードが食い違う場合は **コードを正とする**。

## テスト方法（Phase 1 で整備）

`tools/` にヘッドレステストハーネスがある。実行時は `$env:PYTHONIOENCODING="utf-8"` を付ける（Windows コンソールの日本語文字化け回避）。

- **Tier-1（API不要・決定論・速い）**: 配線とロジックを検証。実 `TaskManager` を一時 `tasks.jsonl` に対して動かし、ハードウェア層（AudioInput/STT/AudioPlayer/TTS/RAG/Feedback）は軽量スタブにする。`TalkMode()` は `initialize()` を呼ばずに構築すれば mic/LLM/音声を作らない（`__init__` だけ）。`process_input(...)` や `generate_stream` を直接駆動する。代表: `test_taskmanager_phase1.py`（instruction ライフサイクル・committed-fact）/ `test_safety_net_phase1.py`（スタブ LLM + 実 TaskManager）/ `test_dedup_guard_phase1.py`（純関数）/ `test_llm_injection_phase1.py`（`_build_system_prompt` のブロック描画）/ `test_e2e_idle_phase1.py`（idle 経路 E2E）。
  - 時間の進め方: deadline は **未来で登録**してから（B2 クランプが過去日付を null 化するため）`inst.deadline_at` を直接過去にして `await tm._reconcile_instruction_status()` を呼ぶと ACTIVE 化できる。
- **Tier-2（実LLM・ヘッドレス）**: `tier2_phase1.py N` が実 `LLMHandler` を本番同等の文脈（talk_prompt + 実 TaskManager の instruction ブロック + committed_facts + silence_summary + meta-tools）で叩き、各シナリオを N 回回して合格率・文脈的自然さ・重複率を出す。要 `.env` キー（コスト発生）。サンプリングは `AI1_TEMPERATURE` 等の env で切替えて A/B できる。
- **Tier-3（実LLM × 実パイプライン × 複数ターン）**: `tier3_session.py N [model_id]` が実 `TalkMode`（音声/TTS/RAG のみスタブ）で台本化セッション（実機事故のリプレイ）を流し、セッション不変条件（再挨拶 ≤1 / 期限前の予約回答なし / deadline 精度 / fact ストア衛生 / 重複 / 孤児 ACTIVE）を検証。沈黙は ConversationCache の timestamp backdating でシミュレート、deadline は実時計 + 短い予約。`model_id` 指定で litellm 経由のモデル A/B も可（本番コード不変更）。**Tier-1/2 が緑でも実機で壊れた教訓**: capture→inject ループ・連続 nudge・LLM 自身の meta-tool 呼び出し・VLM 面は Tier-3 でしか検証できない（Tier-2 は `generate_stream` 直叩きのワンショット）。

### 規律はプロンプトでなくコードで強制する（実測に基づく設計判断）

gpt-5.4 系（mini/full）は ~10k 字のシステムプロンプト中の禁止規則（再挨拶禁止・約束の早期履行禁止）を**守れない**（Tier-3 実測: プロンプト規則の追加・強化でも 0〜2/3）。reasoning 系（gpt-5.5）は守るがレイテンシ 3.5〜12s でリアルタイム VTuber には不可。よって挙動の信頼性が必要な箇所は**コードゲートで決定論的に強制**する（例: `IDLE_SUPPRESS_PENDING_WINDOW_SEC` の沈黙 nudge 抑制、内部 nudge の先頭挨拶文フィルタ、Fix-8 の `_suppress_pending_block`、PENDING fact の render ゲート）。プロンプト規則は補助層。将来のローカル LLM (Qwen 系) 移行ではこの原則がさらに重要になる。また `_build_system_prompt` 内でループ変数が引数 `user_text` を shadow する事故（Bug-F）に注意。

## マルチエージェント運用（重要）

非自明な調査・設計・レビューは 3〜5 個の Explore/Plan サブエージェントを別観点で並列起動して進める（例: 根因調査 / 設計 / Web 調査 / レッドチーム / テスト設計）。**エージェントの結論は鵜呑みにしない** — 実在する重複バグをあるエージェントが「否定」したが、一次データ（ログの直接 grep）で覆った事例がある。重要な主張は必ず一次ソース（実ログ行・実コード行）で自分で確認する。実装は小さな増分でコミットし、各段で Tier-1（可能なら Tier-2）を回してから次へ進む。

## Update Policy

作業後、CLAUDE.mdに反映すべき恒久的な情報が増えたか確認する。

以下に該当する場合のみ、CLAUDE.mdの更新案を提示または更新する。

- プロジェクト目的・仕様の理解が変わった
- Tech Stack、Runtime、Framework、Database、Testing、Linter/Formatter、Package Managerが確定・変更された
- ディレクトリ構成や責務が明確になった
- 設計判断、制約、禁止事項が追加された
- ユーザーから同じミスを防ぐための明示的な指示があった
- 既存のCLAUDE.mdに誤り・古い情報・矛盾が見つかった

一時的な作業ログ、今回限りの判断、未確定情報はCLAUDE.mdに入れない。
不明点は推測せず、更新前にユーザーへ確認する。

## Other

### タスク別の起点

| やりたいこと | 触る場所 |
|---|---|
| 新モード追加 | `modes/base_mode.py` を継承 + `ui/main_window.py` にエントリ登録 |
| 応答 AI の挙動変更 | `modules/llm.py` + `prompts/{talk,game,youtube}_prompt.py` |
| VLM 検出パラメータ調整 | `config/vlm_default.yaml`（**`vlm_orig/` ではない**）または `vlm/detection/` |
| BoT-SORT トラッカー設定 | `config/botsort_reid.yaml` |
| VTube Studio パラメータ | `vts.py`（別プロセスで動作中、変更後は再起動が必要） |
| 永続状態を追加 | `modules/rag.py` / `modules/conversation_cache.py` の async write queue を踏襲 |

### YOLO 重みファイル

リポジトリ root 直下の `yolo11*.pt` / `yolov8*.pt` は YOLO 重み。デフォルトで使われるのは `yolov8n` / `yolov8m`（`config/vlm_default.yaml` の `detection.small_model` / `detection.mid_model`）。`yolo11*` は staging 用に同梱。

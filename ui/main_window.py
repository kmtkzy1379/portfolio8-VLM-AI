"""
Tkinter UIメインウィンドウ
シンプル＆かわいいデザイン + VLM統合
"""
import asyncio
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional
import threading


class _UIInjectLogHandler(logging.Handler):
    """eve.inject logger の出力を MainWindow の add_log に橋渡しするハンドラ。

    add_log は内部で root.after(0, ...) を使って UI スレッドへポストするので、
    asyncio スレッドや VLM スレッドから呼ばれてもスレッドセーフ。
    """

    def __init__(self, ui_callback):
        super().__init__(level=logging.DEBUG)
        self.ui_callback = ui_callback

    def emit(self, record):
        try:
            self.ui_callback("Inject", record.getMessage(), "debug")
        except Exception:
            # UI 側の例外で logger を落とさない
            pass


class MainWindow:
    """メインウィンドウ"""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.root = tk.Tk()
        self.root.title("Eve AI System")
        self.root.geometry("500x780")
        self.root.resizable(True, True)
        self.root.minsize(500, 700)

        # カラースキーム（シンプル＆かわいい）
        self.colors = {
            'bg': '#FFF5F5',           # 薄いピンク背景
            'frame_bg': '#FFFFFF',      # 白いフレーム
            'accent': '#FF9999',        # アクセントピンク
            'text': '#333333',          # 濃いグレーテキスト
            'user': '#4A90D9',          # ユーザーテキスト（青）
            'ai': '#E91E63',            # AIテキスト（ピンク）
            'system': '#888888',        # システムテキスト（グレー）
            'ready': '#4CAF50',         # Ready（緑）
            'processing': '#FF9800',    # Processing（オレンジ）
        }

        self.root.configure(bg=self.colors['bg'])

        # 現在のモード
        self.current_mode = None
        self._mode_task: Optional[asyncio.Task] = None
        self._is_running = False

        # VLM state
        self._vlm_bridge = None
        self._vlm_enabled = tk.BooleanVar(value=False)

        # Debug log toggle (default OFF; ON で AI への注入ログ等が UI に流れる)
        self._show_debug = tk.BooleanVar(value=False)

        # UI構築
        self._create_widgets()

        # eve.inject logger に UI ハンドラを attach（冪等: 既存の同種ハンドラを除去後に追加）
        inject_logger = logging.getLogger("eve.inject")
        inject_logger.setLevel(logging.DEBUG)
        inject_logger.handlers = [
            h for h in inject_logger.handlers
            if not isinstance(h, _UIInjectLogHandler)
        ]
        inject_logger.addHandler(_UIInjectLogHandler(self.add_log))

        # ウィンドウ閉じるイベント
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_widgets(self):
        """ウィジェット作成"""
        # スタイル設定
        style = ttk.Style()
        style.theme_use('clam')

        # ヘッダー
        header_frame = tk.Frame(self.root, bg=self.colors['accent'], height=50)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="Eve AI System",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['accent'],
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=15, pady=12)

        # ⚙ Settings ボタン（ヘッダー右端）
        self._settings_button = tk.Button(
            header_frame,
            text="⚙ Settings",
            font=('Segoe UI', 9),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            activebackground='#FFAAAA',
            command=self._open_settings,
        )
        self._settings_button.pack(side=tk.RIGHT, padx=10)

        # 「⚠ Restart needed」ラベル（Save 後に visible になる）
        self._restart_needed_label = tk.Label(
            header_frame,
            text="⚠ Restart needed",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['accent'],
            fg='#FFEB3B',
        )
        # 初期は非表示。Save 時に pack する。

        # メインコンテンツ
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # モード選択フレーム
        mode_frame = tk.LabelFrame(
            main_frame,
            text=" Mode Selection ",
            font=('Segoe UI', 10),
            bg=self.colors['frame_bg'],
            fg=self.colors['text']
        )
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.mode_var = tk.StringVar(value="talk")

        modes = [
            ("talk", "Talk Mode (対話)"),
            ("game", "Game Mode (P: OCR / O: 画面認識)"),
            ("youtube", "YouTube Mode (コメント読み)")
        ]

        for value, text in modes:
            rb = tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                font=('Segoe UI', 10),
                bg=self.colors['frame_bg'],
                fg=self.colors['text'],
                selectcolor=self.colors['accent'],
                activebackground=self.colors['frame_bg']
            )
            rb.pack(anchor=tk.W, padx=15, pady=5)

        # VLM フレーム
        vlm_frame = tk.LabelFrame(
            main_frame,
            text=" VLM Screen Recognition ",
            font=('Segoe UI', 10),
            bg=self.colors['frame_bg'],
            fg=self.colors['text']
        )
        vlm_frame.pack(fill=tk.X, pady=(0, 10))

        vlm_inner = tk.Frame(vlm_frame, bg=self.colors['frame_bg'])
        vlm_inner.pack(fill=tk.X, padx=15, pady=8)

        self._vlm_check = tk.Checkbutton(
            vlm_inner,
            text="VLM ON/OFF",
            variable=self._vlm_enabled,
            font=('Segoe UI', 10),
            bg=self.colors['frame_bg'],
            fg=self.colors['text'],
            selectcolor=self.colors['accent'],
            activebackground=self.colors['frame_bg'],
            command=self._on_vlm_toggle
        )
        self._vlm_check.pack(side=tk.LEFT)

        self._vlm_status_label = tk.Label(
            vlm_inner,
            text="Stopped",
            font=('Segoe UI', 9),
            bg=self.colors['frame_bg'],
            fg=self.colors['system']
        )
        self._vlm_status_label.pack(side=tk.RIGHT)

        # ステータスフレーム
        status_frame = tk.LabelFrame(
            main_frame,
            text=" Status ",
            font=('Segoe UI', 10),
            bg=self.colors['frame_bg'],
            fg=self.colors['text']
        )
        status_frame.pack(fill=tk.X, pady=(0, 10))

        status_inner = tk.Frame(status_frame, bg=self.colors['frame_bg'])
        status_inner.pack(fill=tk.X, padx=15, pady=10)

        self.status_indicator = tk.Canvas(
            status_inner,
            width=12,
            height=12,
            bg=self.colors['frame_bg'],
            highlightthickness=0
        )
        self.status_indicator.pack(side=tk.LEFT)
        self._update_indicator('ready')

        self.status_label = tk.Label(
            status_inner,
            text="Ready",
            font=('Segoe UI', 11),
            bg=self.colors['frame_bg'],
            fg=self.colors['ready']
        )
        self.status_label.pack(side=tk.LEFT, padx=8)

        # Show debug logs（ON で AI への注入ログ等が UI に流れる。配信中は OFF 推奨）
        self._show_debug_check = tk.Checkbutton(
            status_inner,
            text="Show debug logs",
            variable=self._show_debug,
            font=('Segoe UI', 9),
            bg=self.colors['frame_bg'],
            fg=self.colors['system'],
            selectcolor=self.colors['accent'],
            activebackground=self.colors['frame_bg']
        )
        self._show_debug_check.pack(side=tk.RIGHT)

        # ボタンフレーム
        # NOTE: pack 順は画面表示順と一致しない。side=BOTTOM を先に pack することで
        # Tkinter は available area の下端を最初に確保し、後続の log_frame が
        # expand=True でどれだけ縦を取ろうがボタンが画面外に押し出されないようにする。
        self._button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        self._button_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.start_button = tk.Button(
            self._button_frame,
            text="Start",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['ready'],
            fg='white',
            width=12,
            relief=tk.FLAT,
            cursor='hand2',
            command=self._on_start
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = tk.Button(
            self._button_frame,
            text="Stop",
            font=('Segoe UI', 11, 'bold'),
            bg='#E57373',
            fg='white',
            width=12,
            relief=tk.FLAT,
            cursor='hand2',
            command=self._on_stop,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)

        clear_button = tk.Button(
            self._button_frame,
            text="Clear Log",
            font=('Segoe UI', 9),
            bg='#BDBDBD',
            fg='white',
            width=10,
            relief=tk.FLAT,
            cursor='hand2',
            command=self._clear_log
        )
        clear_button.pack(side=tk.RIGHT)

        # ログフレーム（button_frame の上の残りスペース全部）
        self._log_frame = tk.LabelFrame(
            main_frame,
            text=" Log ",
            font=('Segoe UI', 10),
            bg=self.colors['frame_bg'],
            fg=self.colors['text']
        )
        self._log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(
            self._log_frame,
            font=('Consolas', 9),
            bg='#FAFAFA',
            fg=self.colors['text'],
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # タグ設定
        self.log_text.tag_configure('user', foreground=self.colors['user'])
        self.log_text.tag_configure('ai', foreground=self.colors['ai'])
        self.log_text.tag_configure('system', foreground=self.colors['system'])
        self.log_text.tag_configure('comment', foreground='#9C27B0')
        self.log_text.tag_configure('game', foreground='#009688')
        self.log_text.tag_configure('debug', foreground='#888888')
        self.log_text.tag_configure('inject', foreground='#0088AA')

        # VLM ログフレーム — _create_widgets では作るだけで pack しない。
        # _start_vlm() で log_frame の直後に挿入し、_stop_vlm() で pack_forget する。
        self._vlm_log_frame = tk.LabelFrame(
            main_frame,
            text=" VLM Log ",
            font=('Segoe UI', 10),
            bg=self.colors['frame_bg'],
            fg=self.colors['text']
        )

        self._vlm_log_text = scrolledtext.ScrolledText(
            self._vlm_log_frame,
            font=('Consolas', 9),
            bg='#F0F0F0',
            fg=self.colors['text'],
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=8
        )
        self._vlm_log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # VLMログタグ設定
        self._vlm_log_text.tag_configure('vlm', foreground='#00796B')
        self._vlm_log_text.tag_configure('vlm_error', foreground='#D32F2F')

    def _update_indicator(self, status: str):
        """インジケーター更新"""
        self.status_indicator.delete("all")
        color = self.colors['ready'] if status == 'ready' else self.colors['processing']
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color, outline=color)

    def set_status(self, status: str, text: str):
        """ステータス更新（スレッドセーフ）"""
        def update():
            self._update_indicator(status)
            self.status_label.config(
                text=text,
                fg=self.colors['ready'] if status == 'ready' else self.colors['processing']
            )
        self.root.after(0, update)

    def add_log(self, role: str, message: str, level: str = "info"):
        """ログ追加（スレッドセーフ）

        level="debug" のログは Show debug チェックが ON の時のみ表示する。
        AI への注入ログは role="Inject" + level="debug" で来るので専用色で出す。
        """
        # debug ログは Show debug OFF なら捨てる（UI スレッド外でチェック OK、
        # tk.BooleanVar.get() は読み取り専用なので race しない）
        if level == "debug" and not self._show_debug.get():
            return

        def update():
            self.log_text.config(state=tk.NORMAL)

            # タグ選択
            role_lower = role.lower()
            if role_lower == 'user':
                tag = 'user'
                prefix = "User:"
            elif role_lower == 'ai':
                tag = 'ai'
                prefix = "AI:"
            elif role_lower == 'comment':
                tag = 'comment'
                prefix = "[Comment]"
            elif role_lower == 'game':
                tag = 'game'
                prefix = "[Game]"
            elif role_lower == 'inject':
                tag = 'inject'
                prefix = "[Inject]"
            elif level == 'debug':
                tag = 'debug'
                prefix = f"[{role}]"
            else:
                tag = 'system'
                prefix = f"[{role}]"

            self.log_text.insert(tk.END, f"{prefix} {message}\n", tag)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(0, update)

    def add_vlm_log(self, role: str, message: str):
        """VLMログ追加（スレッドセーフ、200行制限）"""
        def update():
            self._vlm_log_text.config(state=tk.NORMAL)

            tag = 'vlm_error' if 'error' in role.lower() else 'vlm'
            prefix = f"[{role}]"

            self._vlm_log_text.insert(tk.END, f"{prefix} {message}\n", tag)

            # 200行制限
            line_count = int(self._vlm_log_text.index('end-1c').split('.')[0])
            if line_count > 200:
                self._vlm_log_text.delete('1.0', f'{line_count - 200}.0')

            self._vlm_log_text.see(tk.END)
            self._vlm_log_text.config(state=tk.DISABLED)

        self.root.after(0, update)

    def _clear_log(self):
        """ログクリア（通常 Log + VLM Log の両方）"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        self._vlm_log_text.config(state=tk.NORMAL)
        self._vlm_log_text.delete(1.0, tk.END)
        self._vlm_log_text.config(state=tk.DISABLED)

    # ── VLM Control ──

    def _on_vlm_toggle(self):
        """VLMチェックボックス変更時"""
        if self._vlm_enabled.get():
            self._start_vlm()
        else:
            self._stop_vlm()

    def _start_vlm(self):
        """VLMBridge作成+起動"""
        from modules.vlm_bridge import VLMBridge

        # VLM Log 枠を表示（log_frame の直後、button_frame の上に挟まる）
        self._vlm_log_frame.pack(
            fill=tk.BOTH, expand=True, pady=(0, 10), after=self._log_frame
        )

        self._vlm_status_label.config(text="Starting...", fg=self.colors['processing'])
        self.add_vlm_log("VLM", "Starting VLM pipeline...")

        self._vlm_bridge = VLMBridge()
        self._vlm_bridge.set_log_callback(self.add_vlm_log)
        self._vlm_bridge.start()

        # Connect to current mode if running
        if self.current_mode:
            self.current_mode.set_vlm_bridge(self._vlm_bridge)

        # Start polling
        self._poll_vlm_status()

    def _stop_vlm(self):
        """VLM停止"""
        if self._vlm_bridge:
            self.add_vlm_log("VLM", "Stopping VLM pipeline...")

            # Disconnect from current mode (via setter to clear LLM vision state)
            if self.current_mode:
                self.current_mode.set_vlm_bridge(None)

            # Stop in background thread to avoid blocking UI
            def _do_stop():
                self._vlm_bridge.stop()
                self._vlm_bridge = None

            threading.Thread(target=_do_stop, daemon=True).start()

        # VLM Log 枠を畳む（widget は破棄せず内容保持。再 ON で復帰）
        self._vlm_log_frame.pack_forget()

        self._vlm_status_label.config(text="Stopped", fg=self.colors['system'])

    def _poll_vlm_status(self):
        """500msポーリングでVLMステータスをUI更新"""
        if self._vlm_bridge and self._vlm_bridge.is_running:
            self._vlm_status_label.config(text="Running", fg=self.colors['ready'])
            self.root.after(500, self._poll_vlm_status)
        elif self._vlm_enabled.get() and self._vlm_bridge:
            # Still starting up
            self.root.after(500, self._poll_vlm_status)
        else:
            self._vlm_status_label.config(text="Stopped", fg=self.colors['system'])
            # Uncheck if it stopped unexpectedly
            if self._vlm_enabled.get() and not self._vlm_bridge:
                self._vlm_enabled.set(False)

    # ── Mode Control ──

    def _on_start(self):
        """Start ボタン押下"""
        if self._is_running:
            return

        self._is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # モード選択を無効化
        for widget in self.root.winfo_children():
            self._disable_mode_selection(widget)

        mode = self.mode_var.get()
        self.set_status('processing', 'Initializing...')
        self.add_log('System', f'Starting {mode} mode...')

        # 非同期でモードを開始
        asyncio.run_coroutine_threadsafe(self._start_mode(mode), self.loop)

    def _disable_mode_selection(self, widget):
        """モード選択のRadiobuttonを無効化"""
        if isinstance(widget, tk.Radiobutton):
            widget.config(state=tk.DISABLED)
        for child in widget.winfo_children():
            self._disable_mode_selection(child)

    def _enable_mode_selection(self, widget):
        """モード選択のRadiobuttonを有効化"""
        if isinstance(widget, tk.Radiobutton):
            widget.config(state=tk.NORMAL)
        for child in widget.winfo_children():
            self._enable_mode_selection(child)

    async def _start_mode(self, mode: str):
        """モード開始"""
        from modes import TalkMode, GameMode, YouTubeMode

        try:
            # モードインスタンス作成
            if mode == 'talk':
                self.current_mode = TalkMode(log_callback=self.add_log)
            elif mode == 'game':
                self.current_mode = GameMode(
                    log_callback=self.add_log,
                    ready_callback=lambda: self.set_status('ready', 'Ready - P(OCR)/O(Vision)')
                )
            elif mode == 'youtube':
                self.current_mode = YouTubeMode(log_callback=self.add_log)

            # 初期化
            await self.current_mode.initialize()

            # VLMが動作中なら接続
            if self._vlm_bridge and self._vlm_bridge.is_running:
                self.current_mode.set_vlm_bridge(self._vlm_bridge)

            self.set_status('ready', 'Running')

            # メインループ実行
            self._mode_task = asyncio.create_task(self.current_mode.run())
            await self._mode_task

        except asyncio.CancelledError:
            pass
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.add_log('System', f'Error: {e}')
            self.add_log('System', f'Traceback:\n{tb}')
        finally:
            await self._cleanup_mode()

    async def _cleanup_mode(self):
        """モードのクリーンアップ"""
        if self.current_mode:
            await self.current_mode.shutdown()
            self.current_mode = None

        self._is_running = False
        self.set_status('ready', 'Stopped')

        # UI更新
        def update_ui():
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            for widget in self.root.winfo_children():
                self._enable_mode_selection(widget)

        self.root.after(0, update_ui)

    def _on_stop(self):
        """Stop ボタン押下"""
        if not self._is_running:
            return

        self.set_status('processing', 'Stopping...')
        self.add_log('System', 'Stopping...')

        if self.current_mode:
            self.current_mode.stop_requested = True

        if self._mode_task:
            self._mode_task.cancel()

    def _on_closing(self):
        """ウィンドウを閉じる"""
        # VLM停止
        if self._vlm_bridge and self._vlm_bridge.is_running:
            self._vlm_bridge.stop()

        if self._is_running:
            self._on_stop()
            # 少し待ってから閉じる
            self.root.after(500, self._force_close)
        else:
            self._force_close()

    def _force_close(self):
        """強制終了"""
        self.root.quit()
        self.root.destroy()

    def _open_settings(self):
        """Settings ダイアログを開く。"""
        from ui.settings_dialog import SettingsDialog
        SettingsDialog(
            self.root,
            on_saved=self._on_settings_saved,
            colors=self.colors,
        )

    def _on_settings_saved(self):
        """Settings 保存後のコールバック。Restart needed ラベルを表示する。"""
        # 既に表示中なら何もしない（pack_info は管理対象でない場合 KeyError）
        try:
            self._restart_needed_label.pack_info()
        except tk.TclError:
            self._restart_needed_label.pack(side=tk.RIGHT, padx=(0, 10))

    def _bring_to_front_once(self):
        """起動直後に Eve UI を最前面に持ち上げる（VTube Studio フォーカス対策）。

        VTS / VOICEVOX が起動時に最前面を奪うので、Eve UI を一瞬 topmost にして
        前に出した後、すぐ topmost を解除して通常ウィンドウに戻す。
        """
        try:
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.focus_force()
            # 800ms 後に topmost を解除（常時最前面ではなく、起動時のみ最前面化）
            self.root.after(800, lambda: self.root.attributes('-topmost', False))
        except Exception:
            # ウィンドウが既に閉じられている等の競合に備えて握り潰し
            pass

    def run(self):
        """UIメインループ開始"""
        # 2.5 秒後に最前面化を発火（VTS/VOICEVOX 起動の落ち着き待ち）
        self.root.after(2500, self._bring_to_front_once)
        self.root.mainloop()

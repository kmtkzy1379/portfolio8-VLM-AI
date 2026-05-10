"""
Settings ダイアログ（Tkinter Toplevel）
頻出 5 項目だけ UI で編集し、保存時に .env を書き換える。
変更は次回起動時から反映される（実行中の Config は変えない）。
"""
import os
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Optional

from dotenv import dotenv_values, set_key


_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


# 編集対象の 5 項目。プリセットは Combobox の候補として提示するが、
# values 引数の Combobox は state="normal" のままならユーザーがカスタム値を入れられる。
AI1_PRESETS = [
    "gpt-5.4-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "llama-3.3-70b-versatile",
    "claude-opus-4-7",
]
AI2_PRESETS = [
    "claude-opus-4-7",
    "claude-opus-4-5",
    "gpt-4o",
]


class SettingsDialog:
    """設定ダイアログ。MainWindow から呼び出される。"""

    def __init__(
        self,
        parent: tk.Tk,
        on_saved: Optional[Callable[[], None]] = None,
        colors: Optional[dict] = None,
    ):
        self._parent = parent
        self._on_saved = on_saved
        self._colors = colors or {
            'bg': '#FFF5F5',
            'frame_bg': '#FFFFFF',
            'accent': '#FF9999',
            'text': '#333333',
            'system': '#888888',
        }

        # 現在の .env を読み込み（実行中 Config ではなくファイルから直読み）
        env_values = dotenv_values(_ENV_PATH) if _ENV_PATH.exists() else {}

        self._top = tk.Toplevel(parent)
        self._top.title("Settings (反映には Eve の再起動が必要)")
        self._top.geometry("480x560")
        self._top.resizable(False, False)
        self._top.configure(bg=self._colors['bg'])
        self._top.transient(parent)
        self._top.grab_set()

        self._build(env_values)

    def _build(self, env_values: dict) -> None:
        # ── 警告ヘッダー ──
        warn_frame = tk.Frame(self._top, bg='#FFF3E0', height=40)
        warn_frame.pack(fill=tk.X)
        warn_frame.pack_propagate(False)
        tk.Label(
            warn_frame,
            text="⚠ 変更しても次回起動から反映されます（実行中の AI には反映されません）",
            font=('Segoe UI', 9),
            bg='#FFF3E0',
            fg='#E65100',
            wraplength=460,
            justify=tk.LEFT,
        ).pack(padx=10, pady=8, anchor=tk.W)

        body = tk.Frame(self._top, bg=self._colors['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # ── 1. AI1 モデル ──
        self._ai1_var = tk.StringVar(value=env_values.get("AI1_MODEL_NAME", "gpt-5.4-mini"))
        self._add_combobox(
            body, "Eve の脳みそ (応答 LLM)",
            "AI1_MODEL_NAME — Eve の発話を生成する LLM",
            self._ai1_var, AI1_PRESETS,
        )

        # ── 2. AI2 モデル ──
        self._ai2_var = tk.StringVar(value=env_values.get("AI2_MODEL_NAME", "claude-opus-4-5"))
        self._add_combobox(
            body, "反省係 LLM",
            "AI2_MODEL_NAME — 内省ノートを書く LLM (フィードバックループで使用)",
            self._ai2_var, AI2_PRESETS,
        )

        # ── 3. VOICEVOX 話者 ID ──
        self._spk_var = tk.IntVar(value=int(env_values.get("VOICEVOX_SPEAKER_ID", "8") or "8"))
        self._add_spinbox(
            body, "声 (話者 ID)",
            "VOICEVOX_SPEAKER_ID — 例: 1=四国めたん, 8=ずんだもん, 14=冥鳴ひまり",
            self._spk_var, 0, 200,
        )

        # ── 4. VOICEVOX 速度 ──
        self._spd_var = tk.DoubleVar(value=float(env_values.get("VOICEVOX_SPEED", "1.1") or "1.1"))
        self._add_scale(
            body, "話速",
            "VOICEVOX_SPEED — 0.5（遅い）〜 2.0（速い）",
            self._spd_var, 0.5, 2.0, 0.05,
        )

        # ── 5. VOICEVOX 音高 ──
        self._pit_var = tk.DoubleVar(value=float(env_values.get("VOICEVOX_PITCH", "0.0") or "0.0"))
        self._add_scale(
            body, "声の高さ",
            "VOICEVOX_PITCH — -0.15（低い）〜 0.15（高い）",
            self._pit_var, -0.15, 0.15, 0.01,
        )

        # ── ボタン行 ──
        btn_frame = tk.Frame(self._top, bg=self._colors['bg'])
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        tk.Button(
            btn_frame,
            text="Open .env in Notepad",
            font=('Segoe UI', 9),
            bg='#BDBDBD',
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            command=self._open_env_in_notepad,
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_frame,
            text="Cancel",
            font=('Segoe UI', 10),
            bg='#9E9E9E',
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            width=10,
            command=self._top.destroy,
        ).pack(side=tk.RIGHT, padx=(8, 0))

        tk.Button(
            btn_frame,
            text="Save (next launch)",
            font=('Segoe UI', 10, 'bold'),
            bg=self._colors['accent'],
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            width=18,
            command=self._on_save,
        ).pack(side=tk.RIGHT)

    # ── ヘルパー: ウィジェット生成 ──

    def _add_section_header(self, parent, label: str, hint: str) -> tk.Frame:
        section = tk.Frame(parent, bg=self._colors['bg'])
        section.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            section, text=label, font=('Segoe UI', 10, 'bold'),
            bg=self._colors['bg'], fg=self._colors['text'], anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Label(
            section, text=hint, font=('Segoe UI', 8),
            bg=self._colors['bg'], fg=self._colors['system'], anchor=tk.W,
            wraplength=440, justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 4))
        return section

    def _add_combobox(self, parent, label, hint, var, values):
        section = self._add_section_header(parent, label, hint)
        cb = ttk.Combobox(section, textvariable=var, values=values, state='normal')
        cb.pack(fill=tk.X)

    def _add_spinbox(self, parent, label, hint, var, lo, hi):
        section = self._add_section_header(parent, label, hint)
        sb = ttk.Spinbox(section, textvariable=var, from_=lo, to=hi)
        sb.pack(fill=tk.X)

    def _add_scale(self, parent, label, hint, var, lo, hi, step):
        section = self._add_section_header(parent, label, hint)
        scale = tk.Scale(
            section, variable=var, from_=lo, to=hi, resolution=step,
            orient=tk.HORIZONTAL,
            bg=self._colors['bg'], fg=self._colors['text'],
            highlightthickness=0, troughcolor='#E0E0E0',
        )
        scale.pack(fill=tk.X)

    # ── アクション ──

    def _open_env_in_notepad(self):
        try:
            subprocess.Popen(["notepad.exe", str(_ENV_PATH)])
        except Exception as e:
            messagebox.showerror("Error", f"メモ帳を起動できませんでした: {e}", parent=self._top)

    def _on_save(self):
        try:
            updates = {
                "AI1_MODEL_NAME": self._ai1_var.get().strip(),
                "AI2_MODEL_NAME": self._ai2_var.get().strip(),
                "VOICEVOX_SPEAKER_ID": str(int(self._spk_var.get())),
                "VOICEVOX_SPEED": f"{float(self._spd_var.get()):.2f}",
                "VOICEVOX_PITCH": f"{float(self._pit_var.get()):.2f}",
            }
        except Exception as e:
            messagebox.showerror("Error", f"値が不正です: {e}", parent=self._top)
            return

        # 空文字は弾く（モデル名）
        for key, val in updates.items():
            if not val:
                messagebox.showerror("Error", f"{key} が空です", parent=self._top)
                return

        # .env が無ければ作成
        if not _ENV_PATH.exists():
            try:
                _ENV_PATH.touch()
            except Exception as e:
                messagebox.showerror("Error", f".env を作成できませんでした: {e}", parent=self._top)
                return

        # 1 件ずつ書き込み（quote_mode='never' で既存 .env のスタイルに合わせる）
        try:
            for key, val in updates.items():
                set_key(str(_ENV_PATH), key, val, quote_mode='never')
        except Exception as e:
            messagebox.showerror("Error", f".env 書き込み失敗: {e}", parent=self._top)
            return

        messagebox.showwarning(
            "保存しました",
            "現在動作中の AI には反映されません。\n"
            "Eve を一度閉じて再起動してください。",
            parent=self._top,
        )

        if self._on_saved:
            try:
                self._on_saved()
            except Exception:
                pass

        self._top.destroy()

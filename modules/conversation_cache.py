import asyncio
import json
import os
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional


class ConversationCache:
    """会話履歴をメモリにキャッシュして高速アクセスを提供"""
    
    def __init__(self, history_file: str, max_turns: int = 100):
        self.history_file = history_file
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)  # 最大100ターンをメモリに保持
        self._lock = asyncio.Lock()
        self._write_queue = asyncio.Queue()
        self._write_task = None
    
    async def initialize(self):
        """初期化：既存の会話履歴をロード"""
        if os.path.exists(self.history_file):
            try:
                def load_file():
                    entries = []
                    with open(self.history_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    entry = json.loads(line)
                                    entries.append(entry)
                                except json.JSONDecodeError:
                                    continue
                    return entries
                
                entries = await asyncio.to_thread(load_file)
                
                # ユーザー発話とAI応答をペアにしてターンとして保存
                async with self._lock:
                    i = 0
                    while i < len(entries):
                        if "user" in entries[i]:
                            user_text = entries[i].get("user", "")
                            user_timestamp = entries[i].get("timestamp", "")
                            ai_text = ""
                            ai_timestamp = ""
                            
                            # 次のエントリがAI応答か確認
                            if i + 1 < len(entries) and "ai" in entries[i + 1]:
                                ai_text = entries[i + 1].get("ai", "")
                                ai_timestamp = entries[i + 1].get("timestamp", "")
                                i += 2
                            else:
                                i += 1
                            
                            turn = {
                                "user": user_text,
                                "ai": ai_text,
                                "user_timestamp": user_timestamp,
                                "ai_timestamp": ai_timestamp
                            }
                            self.turns.append(turn)
                        else:
                            i += 1
                print(f"Conversation Cache loaded: {len(self.turns)} turns")
            except Exception as e:
                print(f"Conversation Cache Load Error: {e}")
        
        # 非同期書き込みタスクを開始
        self._write_task = asyncio.create_task(self._write_worker())
    
    async def _write_worker(self):
        """非同期でファイルに書き込むワーカー"""
        while True:
            try:
                entry = await self._write_queue.get()
                if entry is None:  # 終了シグナル
                    break
                
                await asyncio.to_thread(self._append_to_file, entry)
                self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Conversation Cache Write Error: {e}")
                self._write_queue.task_done()
    
    def _append_to_file(self, entry: dict):
        """ファイルに追記（同期処理）"""
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"File Write Error: {e}")
    
    async def add_user_message(self, user_text: str) -> None:
        """ユーザーメッセージを追加"""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "user": user_text
        }
        
        # 新しいターンを開始
        async with self._lock:
            self.turns.append({
                "user": user_text,
                "ai": "",
                "user_timestamp": timestamp,
                "ai_timestamp": ""
            })
        
        # ファイルに非同期書き込み
        await self._write_queue.put(entry)
    
    async def add_ai_response(self, ai_text: str) -> None:
        """AI応答を追加（最後のターンに追加）"""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "ai": ai_text
        }
        
        # 最後のターンにAI応答を追加
        async with self._lock:
            if self.turns and self.turns[-1]["ai"] == "":
                self.turns[-1]["ai"] = ai_text
                self.turns[-1]["ai_timestamp"] = timestamp
        
        # ファイルに非同期書き込み
        await self._write_queue.put(entry)
    
    async def get_recent_turns(
        self, count: int = 5, exclude_ellipsis: bool = False,
    ) -> List[Dict[str, str]]:
        """直近のターンを取得（メモリから高速に取得）

        Args:
            count: 取得するターン数。
            exclude_ellipsis: True なら user 発話が「…」だけのターンを除外する。
                沈黙時に AI 自身が出した内部見守り発話で recent_conversation が
                埋まり、直前の実会話が押し出される問題への対策。
        """
        async with self._lock:
            all_turns = list(self.turns)
            if exclude_ellipsis:
                all_turns = [t for t in all_turns if t.get("user", "").strip() != "…"]
            recent = all_turns[-count:] if len(all_turns) > count else all_turns
            return [{"user": turn["user"], "ai": turn["ai"]} for turn in recent]

    async def get_silence_summary(self) -> Dict[str, Optional[float]]:
        """直前期間の沈黙状態を計算する。

        最後の「…」以外のユーザターンを起点として、現在までの経過秒数と
        その間に発火した内部見守り (「…」入力) の件数を返す。

        Returns:
            {
                "silence_seconds": float,  # 直近実ユーザ発話からの経過秒 (0.0 if 不明)
                "ellipsis_count": int,     # その期間中の「…」入力回数
                "last_real_user_ts": Optional[str],  # ISO 8601 / None
            }
        """
        async with self._lock:
            turns = list(self.turns)

        last_real_idx: Optional[int] = None
        for i in range(len(turns) - 1, -1, -1):
            if turns[i].get("user", "").strip() != "…":
                last_real_idx = i
                break

        ellipsis_count = 0
        last_real_user_ts: Optional[str] = None
        silence_seconds: float = 0.0

        if last_real_idx is None:
            ellipsis_count = sum(
                1 for t in turns if t.get("user", "").strip() == "…"
            )
        else:
            last_real_user_ts = turns[last_real_idx].get("user_timestamp") or None
            ellipsis_count = sum(
                1 for t in turns[last_real_idx + 1:]
                if t.get("user", "").strip() == "…"
            )
            if last_real_user_ts:
                try:
                    last_dt = datetime.fromisoformat(last_real_user_ts)
                    silence_seconds = max(
                        0.0, (datetime.now() - last_dt).total_seconds()
                    )
                except (ValueError, TypeError):
                    silence_seconds = 0.0

        return {
            "silence_seconds": silence_seconds,
            "ellipsis_count": ellipsis_count,
            "last_real_user_ts": last_real_user_ts,
        }

    async def shutdown(self):
        """終了処理"""
        if self._write_task:
            await self._write_queue.put(None)  # 終了シグナル
            await self._write_task
            self._write_task = None


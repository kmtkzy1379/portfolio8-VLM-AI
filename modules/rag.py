import asyncio
import json
import os
import random
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
from openai import AsyncOpenAI
from config import Config


class RAGHandler:
    def __init__(self, rag_file: str = None):
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.embedding_model = "text-embedding-3-small"
        self.rag_file = rag_file or Config.RAG_FILE
        self.memory = deque(maxlen=3000)  # ロケット鉛筆方式：最大3000ターン
        self._write_queue = asyncio.Queue()
        self._write_task = None
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """初期化：既存のRAGファイルからメモリをロード"""
        if os.path.exists(self.rag_file):
            try:
                # ファイル読み込みは同期処理なので、asyncio.to_threadで実行
                def load_file():
                    entries = []
                    with open(self.rag_file, "r", encoding="utf-8") as f:
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
                async with self._lock:
                    for entry in entries:
                        self.memory.append(entry)
                print(f"RAG loaded: {len(entries)} entries")
            except Exception as e:
                print(f"RAG Load Error: {e}")
        
        # 非同期書き込みタスクを開始
        self._write_task = asyncio.create_task(self._write_worker())
    
    async def _write_worker(self):
        """非同期でファイルに書き込むワーカー"""
        while True:
            try:
                entry = await self._write_queue.get()
                if entry is None:  # 終了シグナル
                    break
                
                # ファイルI/Oは非同期で実行
                await asyncio.to_thread(self._append_to_file, entry)
                self._write_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"RAG Write Error: {e}")
                self._write_queue.task_done()
    
    def _append_to_file(self, entry: dict):
        """ファイルに追記（同期処理）"""
        try:
            with open(self.rag_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"RAG File Write Error: {e}")
    
    async def add_turn(self, user_text: str, ai_text: str):
        """1ターン（ユーザー発話 + AI発話）を追加"""
        # エンベディング生成用のテキスト（ユーザー発話 + AI発話）
        combined_text = f"{user_text} {ai_text}"
        
        # エンベディングを非同期で生成
        embedding = await self._generate_embedding(combined_text)
        
        entry = {
            "user": user_text,
            "ai": ai_text,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat()
        }
        
        # メモリに追加（dequeが自動的に3000件を超えたら古いものを削除）
        async with self._lock:
            self.memory.append(entry)
        
        # 非同期でファイルに書き出し
        await self._write_queue.put(entry)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """OpenAI APIでエンベディングを生成"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding Error: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """コサイン類似度を計算"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def search_similar(self, query_text: str, top_k: int = 2) -> List[dict]:
        """ユーザー発話に近い記憶を検索（最新1000件から検索して高速化）"""
        if not self.memory:
            return []
        
        # クエリのエンベディングを生成
        query_embedding = await self._generate_embedding(query_text)
        if not query_embedding:
            return []
        
        # メモリ上の最新1000件のエントリと類似度を計算（高速化）
        similarities = []
        async with self._lock:
            # dequeは最新のものが後ろにあるので、最後の1000件を取得
            recent_entries = list(self.memory)[-1000:] if len(self.memory) > 1000 else list(self.memory)
            for entry in recent_entries:
                if "embedding" in entry and entry["embedding"]:
                    similarity = self._cosine_similarity(query_embedding, entry["embedding"])
                    similarities.append((similarity, entry))
        
        # 類似度が高い順にソート
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # top_k件を返す（エンベディングは除外して返す）
        results = []
        for similarity, entry in similarities[:top_k]:
            result_entry = {
                "user": entry.get("user", ""),
                "ai": entry.get("ai", ""),
                "timestamp": entry.get("timestamp", ""),
                "similarity": similarity
            }
            results.append(result_entry)
        
        return results
    
    async def get_random_turns(self, count: int = 2) -> List[dict]:
        """ランダムに記憶を取得（無言時用）"""
        if not self.memory or len(self.memory) == 0:
            return []
        
        async with self._lock:
            available_count = min(count, len(self.memory))
            selected = random.sample(list(self.memory), available_count)
        
        # エンベディングは除外して返す
        results = []
        for entry in selected:
            result_entry = {
                "user": entry.get("user", ""),
                "ai": entry.get("ai", ""),
                "timestamp": entry.get("timestamp", "")
            }
            results.append(result_entry)
        
        return results
    
    async def shutdown(self):
        """終了処理：書き込みタスクを停止"""
        if self._write_task:
            await self._write_queue.put(None)  # 終了シグナル
            await self._write_task
            self._write_task = None


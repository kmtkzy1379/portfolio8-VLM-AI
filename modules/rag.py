import asyncio
import json
import logging
import math
import os
import random
import time
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
from openai import AsyncOpenAI
from config import Config

logger = logging.getLogger(__name__)


def _strip_embedding(entry: dict) -> dict:
    """RAG entry から embedding 配列を除去した display-safe コピーを返す (Phase 4.3)。

    生 embedding を呼び出し側に漏らすと、prompt に巨大配列が混入する事故を起こす。
    search_similar / get_random_turns の戻り値は必ずこの関数を通す。
    """
    return {k: v for k, v in entry.items() if k != "embedding"}


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
        """[DEPRECATED Phase 4.3.1] 生ターン保存。新規呼び出しは禁止。

        Phase 4.3.1 で `add_episode()` に置換された。後方互換のため残置するが、
        BaseMode からの呼び出しは撤去済み。type 無し entry は legacy_turn 扱い。
        """
        combined_text = f"{user_text} {ai_text}"
        embedding = await self._generate_embedding(combined_text)
        entry = {
            "user": user_text,
            "ai": ai_text,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat()
        }
        async with self._lock:
            self.memory.append(entry)
        await self._write_queue.put(entry)

    async def add_episode(
        self,
        summary: str,
        topic_tags: List[str],
        importance: float,
        source_turn_range: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Phase 4.3.1: feedback ループから抽出した episode summary を RAG に追加。

        Embedding 対象は短く密度高い `summary + topic_tags`。生 feedback 全文は入れない。

        dedupe 責務はこの関数内に閉じる: 直前 episode との cosine が閾値以上なら保存スキップ。
        呼び出し側は return 値 (True=保存、False=dedupe スキップ) を信頼するだけでよい。

        Returns:
            True: 保存成功 / False: dedupe スキップ または embedding 失敗
        """
        if not summary:
            return False
        embed_text = summary
        if topic_tags:
            embed_text = f"{summary} | {','.join(topic_tags)}"
        embedding = await self._generate_embedding(embed_text)
        if not embedding:
            return False

        # dedupe: 直前 N 件の episode との cosine 検証
        if await self._is_duplicate_episode(
            embedding, threshold=Config.FB_RAG_DEDUP_THRESHOLD,
        ):
            logger.info(
                "[RAG] episode dedupe skip (cosine > %.2f): %s",
                Config.FB_RAG_DEDUP_THRESHOLD, summary[:50],
            )
            return False

        entry = {
            "type": "episode",
            "summary": summary,
            "topic_tags": list(topic_tags) if topic_tags else [],
            "importance": float(importance),
            "source_turn_range": source_turn_range or "",
            "embedding": embedding,
            "timestamp": timestamp or datetime.now().isoformat(),
        }
        async with self._lock:
            self.memory.append(entry)
        await self._write_queue.put(entry)
        logger.info(
            "[RAG] episode added: importance=%.2f, tags=%s, summary=%s",
            importance, topic_tags, summary[:60],
        )
        return True

    async def add_goal(
        self,
        goal_short: str,
        goal_long: str,
        goal_change: str,
        importance: float = 0.7,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Phase 4.3.3: goal_change=update/pivot 時のみ呼ばれる、goal の RAG entry 化。

        slot は LLMHandler.current_goal_slot が SoT。RAG は想起信号 (連想検索) 用。
        """
        if not goal_short:
            return False
        embed_text = f"{goal_short} | {goal_long} | change={goal_change}"
        embedding = await self._generate_embedding(embed_text)
        if not embedding:
            return False
        entry = {
            "type": "goal",
            "summary": goal_short,
            "goal_long": goal_long,
            "goal_change": goal_change,
            "importance": float(importance),
            "embedding": embedding,
            "timestamp": timestamp or datetime.now().isoformat(),
        }
        async with self._lock:
            self.memory.append(entry)
        await self._write_queue.put(entry)
        logger.info(
            "[RAG] goal added: change=%s, short=%s",
            goal_change, goal_short[:40],
        )
        return True

    async def _is_duplicate_episode(
        self, embedding: List[float], threshold: float, lookback: int = 20,
    ) -> bool:
        """Phase 4.3.1: 直前 lookback 件の episode と embedding cosine 比較。

        threshold 以上の類似度があれば True (重複と判定して保存スキップ)。
        episode 以外 (legacy_turn / goal) は対象外。
        """
        async with self._lock:
            recent = list(self.memory)[-lookback:] if len(self.memory) > lookback else list(self.memory)
        for entry in reversed(recent):
            if entry.get("type") != "episode":
                continue
            other = entry.get("embedding")
            if not other:
                continue
            sim = self._cosine_similarity(embedding, other)
            if sim >= threshold:
                return True
        return False
    
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
    
    async def search_similar(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Phase 4.3.2: 2 段階 ranking (base_score + MMR) + duplicate hard-cut + display-safe return。

        Stage 1: base_score = w_imp × importance + w_rec × recency + w_rel × relevance
                 type 別 importance: episode=entry.importance, goal=entry.importance, legacy_turn=0.3
        Stage 2a: 完全一致 hard-cut (cosine > FB_RAG_DUPLICATE_HARDCUT)
        Stage 2b: MMR (λ × base_score - (1-λ) × max_sim_to_selected)
        Stage 3: embedding を strip した display-safe な dict を返す。

        Args:
            top_k: None なら Config.FB_RAG_TOP_K を使う (BaseMode の hardcoded 撤去対応)。
        """
        if not self.memory:
            return []
        if top_k is None:
            top_k = Config.FB_RAG_TOP_K

        query_embedding = await self._generate_embedding(query_text)
        if not query_embedding:
            return []

        # ── Stage 1: base_score 計算 ──────────────────────────────────
        candidates = []
        now = time.time()
        async with self._lock:
            recent = list(self.memory)[-1000:] if len(self.memory) > 1000 else list(self.memory)
        for entry in recent:
            emb = entry.get("embedding")
            if not emb:
                continue
            relevance = self._cosine_similarity(query_embedding, emb)
            t = entry.get("type", "legacy_turn")
            if t == "episode":
                importance = float(entry.get("importance", 0.5))
            elif t == "goal":
                importance = float(entry.get("importance", 0.7))
            else:  # legacy_turn (type 無し旧 entry も含む)
                importance = 0.3
            try:
                ts_iso = entry.get("timestamp", "")
                ts_epoch = datetime.fromisoformat(ts_iso).timestamp() if ts_iso else now
            except (ValueError, TypeError):
                ts_epoch = now
            recency = math.exp(-(now - ts_epoch) / Config.FB_RAG_RECENCY_TAU)
            base_score = (
                Config.FB_RAG_W_IMP * importance
                + Config.FB_RAG_W_REC * recency
                + Config.FB_RAG_W_REL * relevance
            )
            candidates.append({
                "base_score": base_score,
                "relevance": relevance,
                "entry": entry,
            })
        candidates.sort(key=lambda x: -x["base_score"])
        candidates = candidates[:max(top_k * 4, 20)]  # 候補プール

        # ── Stage 2a: 完全一致 hard-cut (refinement #5) ────────────────
        # MMR alone は重複完全排除を保証しない (small base_score 差で top-k に入る可能性)
        cutoff = Config.FB_RAG_DUPLICATE_HARDCUT
        deduped: list = []
        for cand in candidates:
            cand_emb = cand["entry"].get("embedding")
            if not cand_emb:
                continue
            is_dup = any(
                self._cosine_similarity(cand_emb, d["entry"]["embedding"]) > cutoff
                for d in deduped
            )
            if not is_dup:
                deduped.append(cand)
        candidates = deduped

        # ── Stage 2b: MMR ─────────────────────────────────────────────
        selected: list = []
        lam = Config.FB_RAG_MMR_LAMBDA
        while len(selected) < top_k and candidates:
            best_idx = -1
            best_mmr = -float("inf")
            for i, cand in enumerate(candidates):
                if not selected:
                    mmr = cand["base_score"]
                else:
                    cand_emb = cand["entry"]["embedding"]
                    max_sim = max(
                        self._cosine_similarity(cand_emb, s["entry"]["embedding"])
                        for s in selected
                    )
                    mmr = lam * cand["base_score"] - (1 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))

        # ── Stage 3: display-safe return (embedding strip) ─────────────
        results = []
        for s in selected:
            safe = _strip_embedding(s["entry"])
            safe["similarity"] = s["relevance"]  # 互換: 旧 API も similarity を返していた
            safe["base_score"] = s["base_score"]
            results.append(safe)
        return results

    async def get_random_turns(self, count: int = 2) -> List[dict]:
        """Phase 4.3.2: 無言時用ランダム取得。type 別 display-safe 対応。

        既存の user/ai 抽出を撤廃し、entry をそのまま (embedding 抜き) 返す。
        呼び出し側 (llm.py 等) で type 別表示分岐する。
        """
        if not self.memory:
            return []
        async with self._lock:
            available = min(count, len(self.memory))
            if available == 0:
                return []
            selected = random.sample(list(self.memory), available)
        return [_strip_embedding(e) for e in selected]
    
    async def shutdown(self):
        """終了処理：書き込みタスクを停止"""
        if self._write_task:
            await self._write_queue.put(None)  # 終了シグナル
            await self._write_task
            self._write_task = None


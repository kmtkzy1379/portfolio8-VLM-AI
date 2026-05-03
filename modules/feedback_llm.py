"""
FeedbackLLMClient - Anthropic / OpenAI を統合する薄い抽象レイヤ。

フィードバック (内省) 専用。応答LLM (LLMHandler) とは分離している。

主な特徴:
- モデル名プレフィックスで Provider を自動判定
  (claude- -> Anthropic, gpt- / o1- / o3- / o4- -> OpenAI)
- Anthropic では system をブロックリストで渡し、末尾に cache_control を付けて
  プロンプトキャッシュを活かす。
- 3段フォールバック: primary -> fallback_model -> gpt-4o
- anthropic SDK が未インストールでもアプリ起動は成功する (遅延 import)。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ANTHROPIC_PREFIXES = ("claude-",)
_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")
_REASONING_PREFIXES = ("o1-", "o3-", "o4-")


def detect_provider(model: str) -> str:
    """モデル名から provider 名を返す。"""
    name = (model or "").lower()
    if any(name.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(name.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    # デフォルトは Anthropic (プラン上のデフォルトモデルが Claude Opus のため)
    return "anthropic"


def _is_reasoning_model(model: str) -> bool:
    name = (model or "").lower()
    return any(name.startswith(p) for p in _REASONING_PREFIXES)


class FeedbackLLMClient:
    """フィードバック専用の LLM クライアント (Anthropic / OpenAI 統合)。"""

    # 最終フォールバック先。ここまでは必ず試す。
    _LAST_RESORT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str,
        api_key_openai: Optional[str],
        api_key_anthropic: Optional[str],
        fallback_model: Optional[str] = None,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self._openai_key = api_key_openai
        self._anthropic_key = api_key_anthropic
        self._openai_client = None
        self._anthropic_client = None
        self._init_clients()

        logger.info(
            "FeedbackLLMClient: primary=%s (%s), fallback=%s",
            detect_provider(model), model, fallback_model,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_clients(self) -> None:
        if self._openai_key:
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(api_key=self._openai_key)
            except Exception as e:
                logger.warning("Failed to init OpenAI client: %s", e)

        if self._anthropic_key:
            try:
                from anthropic import AsyncAnthropic
                self._anthropic_client = AsyncAnthropic(api_key=self._anthropic_key)
            except ImportError:
                logger.warning(
                    "anthropic SDK not installed. Claude models unavailable. "
                    "Run: pip install anthropic>=0.40.0"
                )
            except Exception as e:
                logger.warning("Failed to init Anthropic client: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def complete(
        self,
        *,
        system_blocks: list[dict],
        user_text: str,
        max_tokens: int = 4000,
        temperature: float = 0.6,
    ) -> str:
        """system_blocks + user_text を投げてテキストを返す。

        provider 差分を吸収する。primary が失敗したら fallback -> 最終 gpt-4o の順で試す。
        """
        # ルート1: primary
        try:
            return await self._complete_with_model(
                self.model, system_blocks, user_text, max_tokens, temperature
            )
        except Exception as primary_err:
            logger.warning("Feedback primary (%s) failed: %s", self.model, primary_err)

        # ルート2: fallback
        if self.fallback_model and self.fallback_model != self.model:
            try:
                return await self._complete_with_model(
                    self.fallback_model, system_blocks, user_text, max_tokens, temperature
                )
            except Exception as fb_err:
                logger.warning(
                    "Feedback fallback (%s) failed: %s", self.fallback_model, fb_err
                )

        # ルート3: last resort
        if self.model != self._LAST_RESORT_MODEL and self.fallback_model != self._LAST_RESORT_MODEL:
            return await self._complete_with_model(
                self._LAST_RESORT_MODEL, system_blocks, user_text, max_tokens, temperature
            )

        # ここに来たら全ての候補が失敗
        raise RuntimeError("All feedback LLM candidates failed")

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------
    async def _complete_with_model(
        self,
        model: str,
        system_blocks: list[dict],
        user_text: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        provider = detect_provider(model)
        if provider == "anthropic":
            if not self._anthropic_client:
                raise RuntimeError("Anthropic client unavailable (missing SDK or API key)")
            return await self._complete_anthropic(
                model, system_blocks, user_text, max_tokens, temperature
            )
        if not self._openai_client:
            raise RuntimeError("OpenAI client unavailable (missing API key)")
        return await self._complete_openai(
            model, system_blocks, user_text, max_tokens, temperature
        )

    async def _complete_anthropic(
        self,
        model: str,
        system_blocks: list[dict],
        user_text: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        resp = await self._anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_blocks,
            messages=[{"role": "user", "content": user_text}],
        )

        # キャッシュヒット情報をログに出す
        usage = getattr(resp, "usage", None)
        if usage is not None:
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            logger.info(
                "[Feedback] anthropic usage: in=%d out=%d cache_read=%d cache_create=%d",
                input_tokens, output_tokens, cache_read, cache_create,
            )

        # content は list of block object
        parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts).strip()

    async def _complete_openai(
        self,
        model: str,
        system_blocks: list[dict],
        user_text: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        # system を単一 text に合流
        system_text = "\n\n".join(
            b.get("text", "") for b in system_blocks if b.get("type") == "text"
        )

        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        }
        if _is_reasoning_model(model):
            # o1 / o3 系: temperature 指定不可 (デフォルト1固定)、max_completion_tokens を使う
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        resp = await self._openai_client.chat.completions.create(**kwargs)

        # 使用トークン (キャッシュ情報は OpenAI 側でもレスポンスに入ることがある)
        try:
            usage = resp.usage
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            logger.info(
                "[Feedback] openai(%s) usage: prompt=%d completion=%d cached=%d",
                model, usage.prompt_tokens, usage.completion_tokens, cached,
            )
        except Exception:
            pass

        return (resp.choices[0].message.content or "").strip()

"""
ゲームモード - ノベルゲームOCR実況
Pキーでスクリーンキャプチャ→OCR→AI応答
Oキーで画面全体をキャプチャ→画像解析→AI応答
"""
import asyncio
import base64
import warnings
import os
from io import BytesIO
from typing import Callable, Optional

import numpy as np
from PIL import Image
import mss
import keyboard
from openai import OpenAI

from config import Config
from prompts import GAME_SYSTEM_PROMPT
from .base_mode import BaseMode

# 警告を抑制
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class GameMode(BaseMode):
    """ゲームモード：Pキー + OCR + 無言処理なし"""
    
    def __init__(self, log_callback: Optional[Callable[[str, str, str], None]] = None,
                 ready_callback: Optional[Callable[[], None]] = None):
        """
        Args:
            log_callback: ログ出力コールバック
            ready_callback: 処理完了時のコールバック（UIにReady表示用）
        """
        super().__init__(GAME_SYSTEM_PROMPT, log_callback)
        self.ready_callback = ready_callback
        
        # OCR用OpenAIクライアント
        self.ocr_client: Optional[OpenAI] = None
        
        # スクリーン情報
        self.screen_width = 0
        self.screen_height = 0
        
        # キーボードフック
        self._keyboard_hooked = False
        
        # 処理待ちフラグ
        self._pending_ocr = False
        self._pending_vision = False
    
    async def initialize(self):
        """ゲームモード固有の初期化"""
        await super().initialize()
        
        # OCR用OpenAIクライアント初期化
        self.ocr_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # スクリーン解像度取得
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            self.screen_width = monitor['width']
            self.screen_height = monitor['height']
        
        self.log("System", f"Screen: {self.screen_width}x{self.screen_height}", level="debug")
        
        # Pキー・Oキーのフック設定
        keyboard.on_press_key('p', lambda _: self._on_p_pressed())
        keyboard.on_press_key('o', lambda _: self._on_o_pressed())
        self._keyboard_hooked = True
        
        self.log("System", "Game Mode Ready - Press P (OCR) / O (Vision)")
        self._notify_ready()
    
    async def shutdown(self):
        """ゲームモード固有の終了処理"""
        # キーボードフック解除
        if self._keyboard_hooked:
            keyboard.unhook_all()
            self._keyboard_hooked = False
        
        await super().shutdown()
    
    def _on_p_pressed(self):
        """Pキー押下時のコールバック"""
        if not self.is_processing and not self._pending_ocr and not self._pending_vision:
            self._pending_ocr = True
    
    def _on_o_pressed(self):
        """Oキー押下時のコールバック"""
        if not self.is_processing and not self._pending_ocr and not self._pending_vision:
            self._pending_vision = True
    
    def _notify_ready(self):
        """Ready状態を通知"""
        if self.ready_callback:
            self.ready_callback()
    
    def on_response_complete(self):
        """応答完了時のコールバック"""
        super().on_response_complete()
        self._notify_ready()
    
    def capture_bottom_half(self) -> np.ndarray:
        """画面下半分をキャプチャ"""
        monitor = {
            'left': 0,
            'top': self.screen_height // 2,
            'width': self.screen_width,
            'height': self.screen_height // 2
        }
        
        with mss.mss() as sct:
            screenshot = sct.grab(monitor)
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            return np.array(img, dtype=np.uint8)
    
    def capture_full_screen(self) -> np.ndarray:
        """画面全体をキャプチャ"""
        monitor = {
            'left': 0,
            'top': 0,
            'width': self.screen_width,
            'height': self.screen_height
        }
        
        with mss.mss() as sct:
            screenshot = sct.grab(monitor)
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            return np.array(img, dtype=np.uint8)
    
    async def perform_ocr(self, image: np.ndarray) -> str:
        """GPT-4 Vision OCR実行"""
        try:
            # 画像をBase64エンコード
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # OCRリクエスト（同期APIを非同期で実行）
            def _ocr_request():
                return self.ocr_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "画像内のすべての日本語テキストを抽出して、そのまま出力してください。説明や前置きは一切不要です。テキストのみを出力してください。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }],
                    max_tokens=300
                )
            
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _ocr_request)
            
            text = response.choices[0].message.content.strip()
            
            # コスト計算（ログ用）
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            input_cost = input_tokens * 2.50 / 1_000_000
            output_cost = output_tokens * 10.00 / 1_000_000
            total_cost_jpy = (input_cost + output_cost) * 150
            
            self.log("System", f"OCR Cost: ¥{total_cost_jpy:.4f}", level="debug")
            
            return text if text else ""
            
        except Exception as e:
            self.log("System", f"OCR Error: {e}")
            return ""
    
    async def perform_vision_analysis(self, image: np.ndarray) -> str:
        """画像解析実行（画面全体の詳細分析）

        VLM ON時: VisionBuffer + Llama 4 Maverick（高速・安価）
        VLM OFF時: GPT-4o（従来通り）
        """
        # VLM ON → Maverick経由
        if self.vlm_bridge and self.vlm_bridge.is_running:
            return await self._perform_maverick_analysis(image)

        # VLM OFF → 従来のGPT-4o
        return await self._perform_gpt4o_analysis(image)

    async def _perform_maverick_analysis(self, image: np.ndarray) -> str:
        """Llama 4 Maverick による高速画像解析"""
        try:
            from modules.vision_analyzer import VisionAnalyzer

            # 画像をJPEGバイトに変換
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # 960pxにリサイズ
            w, h = pil_image.size
            max_dim = 960
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                pil_image = pil_image.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )

            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=70)
            jpeg_bytes = buffered.getvalue()

            analyzer = VisionAnalyzer()
            prompt = """この画像の内容を詳しく日本語で説明してください:
1. 画面に表示されているテキスト
2. キャラクター/人物の見た目
3. UI要素、背景、レイアウト
4. 色使い、雰囲気
具体的に詳しく説明してください。"""

            description = await analyzer.analyze(jpeg_bytes, prompt)

            self.log("System", "Vision: Maverick (fast)", level="debug")
            return description if description else ""

        except Exception as e:
            self.log("System", f"Maverick Vision Error: {e}")
            return ""

    async def _perform_gpt4o_analysis(self, image: np.ndarray) -> str:
        """GPT-4o による画像解析（従来方式）"""
        try:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            def _vision_request():
                return self.ocr_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """この画像の内容を詳しく日本語で説明してください。以下の要素を含めてください：

1. 画面に表示されているテキスト（すべて）
2. UI要素（ボタン、メニュー、ウィンドウなど）
3. キャラクター、人物、アバター
4. キャラクター１人１人の見た目を詳しく説明してください。
5. 背景、風景
6. 色使い、雰囲気
7. レイアウト、配置
8. その他の視覚的要素

できるだけ具体的に、詳しく説明してください。"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }],
                    max_tokens=1000
                )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _vision_request)

            description = response.choices[0].message.content.strip()

            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            input_cost = input_tokens * 2.50 / 1_000_000
            output_cost = output_tokens * 10.00 / 1_000_000
            total_cost_jpy = (input_cost + output_cost) * 150

            self.log("System", f"Vision Cost: ¥{total_cost_jpy:.4f}", level="debug")

            return description if description else ""

        except Exception as e:
            self.log("System", f"Vision Error: {e}")
            return ""
    
    async def process_capture(self):
        """キャプチャ→OCR→AI応答の一連処理"""
        self.is_processing = True
        
        try:
            self.log("System", "Capturing screen...", level="debug")
            
            # キャプチャ（非同期で実行）
            loop = asyncio.get_running_loop()
            image = await loop.run_in_executor(None, self.capture_bottom_half)
            
            self.log("System", "Running OCR...", level="debug")
            
            # OCR実行
            ocr_text = await self.perform_ocr(image)
            
            if not ocr_text:
                self.log("System", "No text detected")
                return
            
            self.log("Game", ocr_text)
            
            # AI応答生成
            await self.process_input(ocr_text)
            
        except Exception as e:
            self.log("System", f"Capture Error: {e}")
        finally:
            self.is_processing = False
            self._pending_ocr = False
    
    async def process_full_screen_vision(self):
        """画面全体キャプチャ→画像解析→AI応答の一連処理（Oキー）"""
        self.is_processing = True
        
        try:
            self.log("System", "Capturing full screen...", level="debug")
            
            # 画面全体をキャプチャ（非同期で実行）
            loop = asyncio.get_running_loop()
            image = await loop.run_in_executor(None, self.capture_full_screen)
            
            self.log("System", "Running Vision Analysis...", level="debug")
            
            # 画像解析実行
            description = await self.perform_vision_analysis(image)
            
            if not description:
                self.log("System", "Vision analysis failed")
                return
            
            self.log("Game", f"[Vision] {description[:200]}...")
            
            # AI応答生成（解析結果を入力として渡す）
            await self.process_input(f"[画面解析結果]\n{description}")
            
        except Exception as e:
            self.log("System", f"Vision Error: {e}")
        finally:
            self.is_processing = False
            self._pending_vision = False
    
    async def run(self):
        """ゲームモードのメインループ"""
        self.log("System", "Starting Game Mode main loop...", level="debug")
        self.log("System", "Press P (OCR) / O (Vision)")
        
        try:
            while self.running and not self.stop_requested:
                # Pキー押下待ち（OCR）
                if self._pending_ocr:
                    await self.process_capture()
                # Oキー押下待ち（Vision）
                elif self._pending_vision:
                    await self.process_full_screen_vision()
                else:
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            self.log("System", "Game Mode cancelled", level="debug")
        except Exception as e:
            self.log("System", f"Error in Game Mode: {e}")

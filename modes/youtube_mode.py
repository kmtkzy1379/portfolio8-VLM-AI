"""
YouTubeモード - ライブ配信コメント読み上げ
音声合成完了後に最新コメントを取得してループ
"""
import asyncio
from typing import Callable, Optional, Set

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import Config
from prompts import YOUTUBE_SYSTEM_PROMPT
from .base_mode import BaseMode

# 通貨記号マッピング
CURRENCY_SYMBOLS = {
    'JPY': '¥',
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'AUD': 'A$',
    'CAD': 'C$',
}


class YouTubeMode(BaseMode):
    """YouTubeモード：コメント読み上げ"""
    
    def __init__(self, log_callback: Optional[Callable[[str, str, str], None]] = None):
        super().__init__(YOUTUBE_SYSTEM_PROMPT, log_callback)
        
        # YouTube API
        self.youtube = None
        self.live_chat_id: Optional[str] = None
        self.video_id: Optional[str] = None
        
        # 表示済みコメントID
        self.displayed_comment_ids: Set[str] = set()
        
        # 待機フラグ（音声再生完了待ち）
        self._waiting_for_audio = False
    
    async def initialize(self):
        """YouTubeモード固有の初期化"""
        await super().initialize()
        
        # YouTube API初期化
        self.youtube = build('youtube', 'v3', developerKey=Config.YOUTUBE_API_KEY)
        
        # ライブ配信を検索
        self.log("System", f"Searching for live stream on channel: {Config.TARGET_CHANNEL_ID}", level="debug")

        self.video_id = await self._get_live_video_id()
        if not self.video_id:
            self.log("System", "No live stream found")
            return

        self.log("System", f"Found live stream: {self.video_id}", level="debug")

        # ライブチャットID取得
        self.live_chat_id = await self._get_live_chat_id()
        if not self.live_chat_id:
            self.log("System", "Failed to get live chat ID")
            return

        self.log("System", f"Live chat ID: {self.live_chat_id}", level="debug")
        self.log("System", "YouTube Mode Ready")
    
    async def _get_live_video_id(self) -> Optional[str]:
        """ライブ配信のvideo IDを取得"""
        try:
            def _request():
                return self.youtube.search().list(
                    part='id',
                    channelId=Config.TARGET_CHANNEL_ID,
                    eventType='live',
                    type='video',
                    maxResults=1
                ).execute()
            
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _request)
            
            if not response.get('items'):
                return None
            
            return response['items'][0]['id']['videoId']
            
        except HttpError as e:
            self.log("System", f"YouTube API Error: {e}")
            return None
    
    async def _get_live_chat_id(self) -> Optional[str]:
        """ライブチャットIDを取得"""
        if not self.video_id:
            return None
        
        try:
            def _request():
                return self.youtube.videos().list(
                    part='liveStreamingDetails',
                    id=self.video_id
                ).execute()
            
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _request)
            
            if not response.get('items'):
                return None
            
            details = response['items'][0].get('liveStreamingDetails', {})
            return details.get('activeLiveChatId')
            
        except HttpError as e:
            self.log("System", f"YouTube API Error: {e}")
            return None
    
    async def _get_latest_comment(self) -> Optional[dict]:
        """最新のコメントを取得"""
        if not self.live_chat_id:
            return None
        
        try:
            def _request():
                return self.youtube.liveChatMessages().list(
                    liveChatId=self.live_chat_id,
                    part='snippet,authorDetails',
                    maxResults=10
                ).execute()
            
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _request)
            
            items = response.get('items', [])
            
            # 未表示のコメントをフィルタリング
            new_comments = [
                item for item in items
                if item['id'] not in self.displayed_comment_ids
            ]
            
            if not new_comments:
                return None
            
            # 最新のコメントを取得
            latest = max(new_comments, key=lambda x: x['snippet']['publishedAt'])
            self.displayed_comment_ids.add(latest['id'])
            
            return latest
            
        except HttpError as e:
            error_str = str(e).lower()
            if 'disabled' in error_str or 'not found' in error_str:
                self.log("System", "Live stream ended")
                self.stop_requested = True
            else:
                self.log("System", f"YouTube API Error: {e}")
            return None
    
    def _format_comment(self, message: dict) -> str:
        """コメントをAI入力用にフォーマット"""
        snippet = message['snippet']
        
        # 投稿者名
        author_name = "不明"
        if 'authorDetails' in message:
            author_name = message['authorDetails'].get('displayName', '不明')
        
        # スーパーチャット判定
        is_super_chat = snippet.get('type') == 'superChatEvent'
        
        # コメント内容
        if is_super_chat and 'superChatDetails' in snippet:
            comment_text = snippet['superChatDetails'].get('userComment', '(メッセージなし)')
        else:
            comment_text = snippet.get('displayMessage', 
                snippet.get('textMessageDetails', {}).get('messageText', ''))
        
        # フォーマット作成
        formatted = f"リスナー名: {author_name}\n"
        formatted += f"コメント: {comment_text}\n"
        formatted += f"スーパーチャット: {str(is_super_chat).lower()}"
        
        # スーパーチャットの金額
        if is_super_chat and 'superChatDetails' in snippet:
            details = snippet['superChatDetails']
            amount_micros = details.get('amountMicros')
            currency = details.get('currency')
            
            if amount_micros and currency:
                amount = int(amount_micros) / 1_000_000
                symbol = CURRENCY_SYMBOLS.get(currency, currency + ' ')
                if amount == int(amount):
                    formatted += f"\n金額: {symbol}{int(amount)}"
                else:
                    formatted += f"\n金額: {symbol}{amount:.2f}"
        
        return formatted
    
    async def _wait_for_audio_complete(self):
        """音声再生完了を待つ"""
        while self.player.is_playing or not self.player.queue.empty():
            if self.stop_requested:
                break
            await asyncio.sleep(0.1)
    
    async def run(self):
        """YouTubeモードのメインループ"""
        if not self.live_chat_id:
            self.log("System", "Cannot start - no live chat ID")
            return
        
        self.log("System", "Starting YouTube Mode main loop...", level="debug")
        
        try:
            while self.running and not self.stop_requested:
                # 最新コメント取得
                comment = await self._get_latest_comment()
                
                if comment:
                    # コメントをフォーマットしてAIに送信
                    formatted = self._format_comment(comment)
                    self.log("Comment", formatted)
                    
                    # AI応答生成
                    await self.process_input(formatted)
                    
                    # 音声再生完了を待つ
                    await self._wait_for_audio_complete()
                else:
                    # コメントがない場合は少し待つ
                    await asyncio.sleep(2)
                    
        except asyncio.CancelledError:
            self.log("System", "YouTube Mode cancelled", level="debug")
        except Exception as e:
            self.log("System", f"Error in YouTube Mode: {e}")

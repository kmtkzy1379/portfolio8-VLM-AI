import pyaudio
import asyncio
import io
import soundfile as sf
from typing import Optional
from config import Config


class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.queue = asyncio.Queue()
        self.is_playing = False
        self.interrupt_signal = False

        # Step 2: 多重 interrupt レース + wait レース対策
        # - interrupt_done: Dispatcher が「割り込み完了」を待つための Event（初期は完了状態）
        # - _reset_task: 200ms 後の signal リセットタスクを保持（多重 interrupt 時に cancel + 再作成）
        # - _loop: メインループへの参照（別スレッドからの interrupt() 呼び出しに対応）
        self.interrupt_done: asyncio.Event = asyncio.Event()
        self.interrupt_done.set()
        self._reset_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """BaseMode 起動時にメインループを渡す（別スレッドからの interrupt 用）。"""
        self._loop = loop

    async def play_worker(self):
        while True:
            try:
                wav_bytes = await self.queue.get()
                if self.interrupt_signal:
                    self.queue.task_done()
                    continue


                self.is_playing = True
                await self._play_audio(wav_bytes)
                self.is_playing = False
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Player Error: {e}")


    async def _play_audio(self, wav_bytes):
        with io.BytesIO(wav_bytes) as wav_io:
            data, samplerate = sf.read(wav_io, dtype='float32')


        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=samplerate,
            output=True
        )


        chunk_size = 1024
        loop = asyncio.get_running_loop()

        for i in range(0, len(data), chunk_size):
            if self.interrupt_signal:
                break
            chunk = data[i:i + chunk_size]
            await loop.run_in_executor(None, lambda: stream.write(chunk.tobytes()))


        stream.stop_stream()
        stream.close()


    def add_to_queue(self, audio_data: bytes):
        self.queue.put_nowait(audio_data)


    async def interrupt_async(self):
        """同一イベントループ内から呼び出す割り込み（推奨）。

        clear → drain → reset task 作成 → wait の順序を同期保証。
        Dispatcher 等から `await player.interrupt_async()` の形で呼べば、
        戻った時点で interrupt は完全に処理済み（200ms 経過後）。
        """
        print(">> Player Interrupted!")
        self.interrupt_signal = True
        self.interrupt_done.clear()
        # 既存 reset task を cancel（多重 interrupt レース対策）
        if self._reset_task and not self._reset_task.done():
            self._reset_task.cancel()
        # 待機中の TTS チャンクを破棄
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        # 新規 reset task を起動
        self._reset_task = asyncio.create_task(self._reset_interrupt())
        await self.interrupt_done.wait()

    def interrupt(self):
        """fire-and-forget 版。どのスレッドから呼んでも安全。

        - 同一ループ内: create_task で interrupt_async を起動
        - 別スレッドから: run_coroutine_threadsafe で委譲
        ※ 「完了を待ちたい」用途は interrupt_async() を使うこと。
        """
        if self._loop is not None and self._loop.is_running():
            try:
                running = asyncio.get_running_loop()
                if running is self._loop:
                    asyncio.create_task(self.interrupt_async())
                else:
                    asyncio.run_coroutine_threadsafe(self.interrupt_async(), self._loop)
            except RuntimeError:
                # 現スレッドにループが無い → 別スレッド扱い
                asyncio.run_coroutine_threadsafe(self.interrupt_async(), self._loop)
        else:
            # フォールバック（loop 未設定、テスト時等）
            self.interrupt_signal = True


    async def _reset_interrupt(self):
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            return  # 新しい interrupt が来たので静かに退場
        self.interrupt_signal = False
        self.interrupt_done.set()

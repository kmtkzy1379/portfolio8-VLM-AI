#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eve AI System - マルチモード統合アプリケーション

使用方法:
    python app.py

モード:
    - Talk Mode: マイク入力による対話
    - Game Mode: ノベルゲームOCR実況（Pキー押下）
    - YouTube Mode: ライブコメント読み上げ
"""
import asyncio
import os
import sys
import threading
import colorama
from colorama import Fore

# Windows用イベントループポリシー設定
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import Config
from ui import MainWindow


def run_async_loop(loop: asyncio.AbstractEventLoop):
    """非同期ループを別スレッドで実行"""
    asyncio.set_event_loop(loop)
    loop.run_forever()


def main():
    """メインエントリポイント"""
    colorama.init(autoreset=True)
    
    print(Fore.CYAN + "=" * 50)
    print(Fore.CYAN + "  Eve AI System - Multi-Mode Application")
    print(Fore.CYAN + "=" * 50)
    
    # 設定検証
    try:
        Config.validate()
    except ValueError as e:
        print(Fore.RED + f"Config Error: {e}")
        sys.exit(1)
    
    print(Fore.GREEN + "Configuration validated.")

    # VOICEVOX接続確認
    try:
        import requests as _req
        resp = _req.get(f"{Config.VV_URL}/version", timeout=2)
        if resp.status_code == 200:
            print(Fore.GREEN + f"VOICEVOX connected (version: {resp.text.strip().strip(chr(34))})")
        else:
            print(Fore.YELLOW + f"VOICEVOX responded with status {resp.status_code}")
    except Exception:
        print(Fore.YELLOW + "VOICEVOX not reachable - TTS may not work")

    # 非同期ループを別スレッドで起動
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=run_async_loop, args=(loop,), daemon=True)
    loop_thread.start()
    
    print(Fore.GREEN + "Async loop started.")
    print(Fore.YELLOW + "Starting UI...")
    print()
    
    # UIウィンドウを起動
    window = MainWindow(loop)
    
    try:
        window.run()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nInterrupted by user.")
    finally:
        # ループを停止
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        
        print(Fore.GREEN + "Application closed.")


if __name__ == "__main__":
    main()

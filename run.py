#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eve AI System - 統合ランチャー

起動順序:
1. VTube Studio & VOICEVOX を起動（launcher.py）
2. VOICEVOX接続待機
3. VTS制御スクリプトを起動（vts.py）
4. メインアプリを起動（app.py）

使用方法:
    python run.py
"""
import os
import sys
import time
import subprocess
import atexit
import signal
from pathlib import Path

# スクリプトのディレクトリをパスに追加
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# 起動したサブプロセスを管理
_subprocesses = []


def cleanup_subprocesses():
    """終了時にサブプロセスをクリーンアップ"""
    for proc in _subprocesses:
        if proc.poll() is None:  # まだ動いている場合
            print(f"サブプロセス (PID: {proc.pid}) を終了しています...")
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception as e:
                print(f"サブプロセス終了エラー: {e}")


def signal_handler(signum, frame):
    """シグナルハンドラ"""
    print("\n終了シグナルを受信しました...")
    cleanup_subprocesses()
    sys.exit(0)


def main():
    """メインエントリポイント"""
    print("=" * 60)
    print("  Eve AI System - 統合ランチャー")
    print("=" * 60)
    print()

    # 終了時のクリーンアップを登録
    atexit.register(cleanup_subprocesses)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ----------------------------------------
    # Step 1: VTube Studio & VOICEVOX を起動
    # ----------------------------------------
    print("[Step 1/3] 外部アプリケーションを起動中...")
    try:
        from launcher import launch_external_apps_sync, wait_for_voicevox
        results = launch_external_apps_sync()
        print(f"  VTube Studio: {'OK' if results.get('vts') else 'SKIP'}")
        print(f"  VOICEVOX: {'OK' if results.get('voicevox') else 'SKIP'}")

        # VOICEVOX接続待機
        if results.get("voicevox"):
            print("  VOICEVOX HTTP API 待機中...")
            wait_for_voicevox(timeout=30)
    except Exception as e:
        print(f"  警告: 外部アプリ起動エラー: {e}")

    print()

    # ----------------------------------------
    # Step 2: VTS制御スクリプトを起動
    # ----------------------------------------
    print("[Step 2/3] VTS制御スクリプトを起動中...")
    vts_script = SCRIPT_DIR / "vts.py"

    if vts_script.exists():
        try:
            # vts.pyをバックグラウンドで起動（stdout/stderrをDEVNULLに）
            vts_proc = subprocess.Popen(
                [sys.executable, str(vts_script)],
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            _subprocesses.append(vts_proc)
            print(f"  VTS制御スクリプト起動 (PID: {vts_proc.pid})")
        except Exception as e:
            print(f"  警告: VTS制御スクリプト起動エラー: {e}")
    else:
        print(f"  警告: vts.py が見つかりません: {vts_script}")

    print()

    # VTube Studioとの接続確立を待つ
    print("  VTube Studio接続待機中 (2秒)...")
    time.sleep(2)

    # ----------------------------------------
    # Step 3: メインアプリを起動
    # ----------------------------------------
    print("[Step 3/3] メインアプリを起動中...")
    print()
    print("=" * 60)

    try:
        # app.pyのmain()を直接呼び出し
        from app import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\nユーザーによる中断")
    except Exception as e:
        print(f"アプリケーションエラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print()
        print("=" * 60)
        print("  Eve AI System を終了しています...")
        print("=" * 60)
        cleanup_subprocesses()


if __name__ == "__main__":
    main()

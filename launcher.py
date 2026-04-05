"""
外部アプリケーションランチャー
VTube Studio と VOICEVOX を起動する（同期版）
"""
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


# .envを読み込み
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


def launch_app(path: str, name: str) -> bool:
    """アプリケーションを起動

    Args:
        path: アプリケーションのパス
        name: アプリケーション名（ログ用）

    Returns:
        起動成功ならTrue、失敗ならFalse
    """
    if not path:
        print(f"警告: {name} のパスが設定されていません")
        return False

    if os.path.exists(path):
        os.startfile(path)
        print(f"{name} を起動しました")
        return True
    else:
        print(f"エラー: {name} のパスが見つかりません: {path}")
        return False


def wait_for_voicevox(url: str = None, timeout: float = 30.0) -> bool:
    """VOICEVOXのHTTP APIが応答するまでポーリング待機

    Args:
        url: VOICEVOX URL (default: env or http://127.0.0.1:50021)
        timeout: 最大待機秒数

    Returns:
        接続成功ならTrue
    """
    if url is None:
        url = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")

    version_url = f"{url}/version"
    start = time.monotonic()
    attempt = 0

    while time.monotonic() - start < timeout:
        attempt += 1
        try:
            resp = requests.get(version_url, timeout=2)
            if resp.status_code == 200:
                version = resp.text.strip().strip('"')
                print(f"  VOICEVOX 接続確認 (version: {version})")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        # 最初は短く、徐々に間隔を延ばす
        time.sleep(min(1.0, 0.3 + attempt * 0.1))

    print(f"  警告: VOICEVOX が {timeout}秒以内に応答しませんでした")
    return False


def launch_external_apps_sync() -> dict:
    """VTube Studio と VOICEVOX を同期的に起動

    Returns:
        各アプリの起動結果を含む辞書
    """
    vts_path = os.getenv("VTS_PATH")
    voicevox_path = os.getenv("VOICEVOX_PATH")

    results = {}
    results["vts"] = launch_app(vts_path, "VTube Studio")
    results["voicevox"] = launch_app(voicevox_path, "VOICEVOX")

    return results


if __name__ == "__main__":
    results = launch_external_apps_sync()
    print(f"起動結果: {results}")
    if results.get("voicevox"):
        wait_for_voicevox()

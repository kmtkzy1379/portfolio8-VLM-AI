import asyncio
import json
import random
import websockets
import os
import time

# --- VTube Studio 設定 ---
VTS_API_URL = "ws://localhost:8001"
PLUGIN_NAME = "NaturalBodyMover"
PLUGIN_DEVELOPER = "kmtkzy"
AUTH_TOKEN_FILE = "vts_auth_token.txt"

# ランダム動作パラメータと範囲
PARAMS_TO_RANDOMIZE = {
    "FaceAngleX": {"min": -15, "max": 15},
    "FaceAngleY": {"min": -10, "max": 10},
    "FaceAngleZ": {"min": -20, "max": 20},
}

# --- 瞬き設定 ---
BLINK_PARAM_NAMES = ["EyeOpenLeft","EyeOpenRight"]  # モデルに合わせて調整
BLINK_INTERVAL_MIN_SEC = 2.0
BLINK_INTERVAL_MAX_SEC = 6.0
BLINK_DURATION_SEC = 0.14       # 閉じ→開きの合計時間
BLINK_HOLD_CLOSED_SEC = 0.03    # 閉じたままの時間

# 自然な動きの設定
UPDATE_INTERVAL_SECONDS = 0.10
EASE_FACTOR = 0.1
TARGET_UPDATE_INTERVAL_SECONDS = 3.0
SLEEP_JITTER_SECONDS = 0.02  # 微小なタイミングずれ
NOISE_MAGNITUDE = 0.25       # 微小ノイズ強度
TARGET_INTERVAL_JITTER = 0.3 # ターゲット更新間隔のばらつき

# --- デフォルト値への復帰設定 ---
RETURN_TO_DEFAULT_SPEED = 0.05

# パラメータ状態管理
param_state = {}

# 瞬き状態管理
blink_state = {
    "current_value": 1.0,            # 1.0 開、0.0 閉
    "status": "IDLE",                # IDLE, CLOSING, HOLDING, OPENING
    "last_status_change_time": 0.0,
    "next_blink_trigger_time": 0.0,
}

# --- VTube Studio API 通信関数 ---

def create_request(request_type, data=None):
    """APIリクエスト用のJSONペイロードを作成"""
    return json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": f"req-{random.randint(10000, 99999)}",
        "messageType": request_type,
        "data": data or {}
    })

async def send_request(websocket, request_type, data=None):
    """WebSocket経由でリクエスト送信（タイムアウト付き）"""
    try:
        await websocket.send(create_request(request_type, data))
        if request_type == "InjectParameterDataRequest":
            # パラメータ注入はレスポンスが返らない場合がある
            return {"messageType": "InjectParameterDataResponse", "data": {}}
        raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        return json.loads(raw)
    except asyncio.TimeoutError:
        return None
    except websockets.exceptions.ConnectionClosed as e:
        raise e
    except Exception as e:
        return None

# --- 認証処理 ---

def load_auth_token():
    """認証トークンをファイルから読み込む"""
    try:
        with open(AUTH_TOKEN_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def save_auth_token(token):
    """認証トークンをファイルに保存"""
    with open(AUTH_TOKEN_FILE, 'w') as f:
        f.write(token)
    print(f"認証トークンを {AUTH_TOKEN_FILE} に保存しました。")

async def authenticate(websocket):
    """接続と認証処理"""
    print("認証処理を開始します...")
   
    token = load_auth_token()
   
    if not token:
        # トークンがなければ新規取得
        request_data = {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_DEVELOPER
        }
        response = await send_request(websocket, "AuthenticationTokenRequest", request_data)
       
        if response and response.get("messageType") == "AuthenticationTokenResponse":
            new_token = response["data"]["authenticationToken"]
            save_auth_token(new_token)
            print("認証トークンを取得しました。VTube Studioでプラグインを承認してください。")
            await asyncio.sleep(5)  # ユーザーの承認待ち
            token = new_token
        else:
            print("エラー: 認証トークンの取得に失敗しました。")
            return None

    # トークンで認証
    request_data = {
        "pluginName": PLUGIN_NAME,
        "pluginDeveloper": PLUGIN_DEVELOPER,
        "authenticationToken": token
    }
    response = await send_request(websocket, "AuthenticationRequest", request_data)
   
    if response and response.get("messageType") == "AuthenticationResponse" and response["data"].get("authenticated"):
        print("認証に成功しました！")
        return True
    else:
        reason = None
        if response and isinstance(response, dict):
            reason = response.get("data", {}).get("reason")
        print(f"エラー: 認証に失敗しました。理由: {reason or '不明'}")
        if os.path.exists(AUTH_TOKEN_FILE):
            os.remove(AUTH_TOKEN_FILE)
            print("無効なトークンを削除しました。再起動してください。")
        return False

# --- パラメータ操作 ---

def initialize_param_state():
    """パラメータ状態を初期化"""
    global param_state
    now = time.time()
    for param_name, config in PARAMS_TO_RANDOMIZE.items():
        initial_value = random.uniform(config["min"], config["max"])
        param_state[param_name] = {
            "current": initial_value,
            "target": initial_value,
            "min": config["min"],
            "max": config["max"],
            "last_target_update_time": now,
            "next_target_update_interval": TARGET_UPDATE_INTERVAL_SECONDS
        }
    print("パラメータ状態を初期化しました。")
    # 瞬きも初期化
    now2 = time.time()
    blink_state["current_value"] = 1.0
    blink_state["status"] = "IDLE"
    blink_state["last_status_change_time"] = now2
    blink_state["next_blink_trigger_time"] = now2 + random.uniform(BLINK_INTERVAL_MIN_SEC, BLINK_INTERVAL_MAX_SEC)

def update_parameters():
    """パラメータ状態を計算（通信はしない）"""
    current_time = time.time()
    for param_name, state in param_state.items():
        # ターゲット更新タイミングのランダム化
        if current_time - state["last_target_update_time"] > state.get("next_target_update_interval", TARGET_UPDATE_INTERVAL_SECONDS):
            state["target"] = random.uniform(state["min"], state["max"])
            state["last_target_update_time"] = current_time
            # 次の更新間隔もランダム化
            jitter = 1.0 + random.uniform(-TARGET_INTERVAL_JITTER, TARGET_INTERVAL_JITTER)
            state["next_target_update_interval"] = max(0.8, TARGET_UPDATE_INTERVAL_SECONDS * jitter)

        state["current"] += (state["target"] - state["current"]) * EASE_FACTOR
        # 微小ノイズを加える
        state["current"] += random.uniform(-NOISE_MAGNITUDE, NOISE_MAGNITUDE) * EASE_FACTOR
        state["current"] = max(state["min"], min(state["max"], state["current"]))

    # --- 瞬き状態の更新 ---
    t = current_time
    elapsed = t - blink_state["last_status_change_time"]
    if blink_state["status"] == "IDLE":
        if t >= blink_state["next_blink_trigger_time"]:
            blink_state["status"] = "CLOSING"
            blink_state["last_status_change_time"] = t
    elif blink_state["status"] == "CLOSING":
        progress = min(1.0, elapsed / (BLINK_DURATION_SEC / 2.0))
        blink_state["current_value"] = 1.0 - progress
        if blink_state["current_value"] <= 0.0:
            blink_state["current_value"] = 0.0
            blink_state["status"] = "HOLDING"
            blink_state["last_status_change_time"] = t
    elif blink_state["status"] == "HOLDING":
        if elapsed >= BLINK_HOLD_CLOSED_SEC:
            blink_state["status"] = "OPENING"
            blink_state["last_status_change_time"] = t
    elif blink_state["status"] == "OPENING":
        progress = min(1.0, elapsed / (BLINK_DURATION_SEC / 2.0))
        blink_state["current_value"] = progress
        if blink_state["current_value"] >= 1.0:
            blink_state["current_value"] = 1.0
            blink_state["status"] = "IDLE"
            blink_state["last_status_change_time"] = t
            blink_state["next_blink_trigger_time"] = t + random.uniform(BLINK_INTERVAL_MIN_SEC, BLINK_INTERVAL_MAX_SEC)

async def send_parameters_task(websocket):
    """定期的にパラメータをVTSへ送信するタスク"""
    while True:
        update_parameters()
        param_values_to_inject = []
        for param_name, state in param_state.items():
            param_values_to_inject.append({"id": param_name, "value": state["current"]})

        # 瞬きパラメータも追加
        for param_name in BLINK_PARAM_NAMES:
            param_values_to_inject.append({"id": param_name, "value": blink_state["current_value"]})
        data = {"parameterValues": param_values_to_inject}
        await send_request(websocket, "InjectParameterDataRequest", data)
        # 微小なタイミングずれを加えて負荷分散
        await asyncio.sleep(UPDATE_INTERVAL_SECONDS + random.uniform(0.0, SLEEP_JITTER_SECONDS))

async def app_heartbeat_task(websocket):
    """アプリ側のハートビートタスク"""
    while True:
        try:
            # 定期的なpingで接続維持
            await websocket.ping()
        except Exception:
            # 接続異常は上位で再接続
            raise
        await asyncio.sleep(15)

async def main():
    retry_delay_sec = 3
    while True:
        try:
            print(f"{VTS_API_URL} に接続中...")
            async with websockets.connect(
                VTS_API_URL,
                ping_interval=30,   # 定期ping
                ping_timeout=25,    # pong待ちタイムアウト
                open_timeout=5,     # 接続確立タイムアウト
                close_timeout=3,    # クローズ待ちタイムアウト
                max_queue=None      # バックプレッシャ回避
            ) as websocket:
                print("WebSocket接続に成功しました！")

                if not await authenticate(websocket):
                    print("認証に失敗しました。5秒後に再試行します...")
                    await asyncio.sleep(5)
                    retry_delay_sec = min(20, retry_delay_sec * 2)
                    continue

                initialize_param_state()
                print("\n動作を開始しました。AIアシスタント側で終了してください。")

                # パラメータ送信とハートビートを並列実行
                sender = asyncio.create_task(send_parameters_task(websocket))
                heartbeat = asyncio.create_task(app_heartbeat_task(websocket))
                done, pending = await asyncio.wait(
                    {sender, heartbeat},
                    return_when=asyncio.FIRST_EXCEPTION
                )
                for task in pending:
                    task.cancel()

        except websockets.exceptions.ConnectionClosed as e:
            print(f"接続が切断されました。理由: {e}")
            print(f"{retry_delay_sec}秒後に再接続します...")
            await asyncio.sleep(retry_delay_sec)
            retry_delay_sec = min(20, retry_delay_sec * 2)
            continue
        except ConnectionRefusedError:
            print("\n-------------------------------------------------------------")
            print("エラー: VTube Studioに接続できません。")
            print("原因: VTube Studioが起動していないか、APIが有効化されていません。")
            print("-------------------------------------------------------------")
            print(f"{retry_delay_sec}秒後に再試行します...")
            await asyncio.sleep(retry_delay_sec)
            retry_delay_sec = min(20, retry_delay_sec * 2)
            continue
        # KeyboardInterruptの処理を削除
        except Exception as e:
            print(f"予期しないエラーが発生しました: {e}")
            print(f"{retry_delay_sec}秒後に再試行します...")
            await asyncio.sleep(retry_delay_sec)
            retry_delay_sec = min(20, retry_delay_sec * 2)
            continue

if __name__ == "__main__":
    asyncio.run(main())


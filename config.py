import os
from pathlib import Path
from dotenv import load_dotenv


# この config.py と同じディレクトリの .env を必ず読む
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)


class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    # YouTube
    TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID")

    # Models
    AI1_MODEL = os.getenv("AI1_MODEL_NAME", "llama-3.3-70b-versatile")
    AI2_MODEL = os.getenv("AI2_MODEL_NAME", "gpt-4o")
    VISION_ANALYSIS_MODEL = os.getenv(
        "VISION_ANALYSIS_MODEL",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # Vision Buffer
    VISION_BUFFER_SIZE = int(os.getenv("VISION_BUFFER_SIZE", "20"))


    # Voicevox
    VV_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")
    VV_SPEAKER = int(os.getenv("VOICEVOX_SPEAKER_ID", "1"))
    VV_SPEED = float(os.getenv("VOICEVOX_SPEED", "1.1"))
    VV_PITCH = float(os.getenv("VOICEVOX_PITCH", "0.0"))


    # Audio & VAD
    SAMPLE_RATE = 16000
    CHANNELS = 1
    VAD_THRESHOLD = 0.5
    SILENCE_LIMIT = 0.5


    # Paths
    HISTORY_FILE = "conversation_history.txt"
    SUMMARY_FILE = "memory_summary.txt"
    RAG_FILE = "rag_memory.jsonl"


    @staticmethod
    def validate():
        # デバッグ用に現在読み込まれているキーの情報を少しだけ表示
        if Config.GROQ_API_KEY:
            print(f"GROQ_API_KEY(head)={Config.GROQ_API_KEY[:8]}..., len={len(Config.GROQ_API_KEY)}")
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in .env")
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in .env")




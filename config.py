from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
INPUT_DIR = Path("data/pdfs")
OUTPUT_DIR = Path("data/outputs")
TXT_DIR = Path("data/txts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

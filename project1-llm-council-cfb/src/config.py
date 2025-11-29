from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

CONTEXT_DIR = BASE_DIR / "context"
PROMPT_DIR = BASE_DIR / "prompts"

GAME_DATA_FILE = CONTEXT_DIR / "game_data.toon"
AGENT_SYSTEM_PROMPT_FILE = PROMPT_DIR / "agent_system_prompt.txt"
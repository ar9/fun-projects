import json
from pathlib import Path
import pandas as pd
from typing import Optional, Any

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


def load_text(path: str | Path) -> str:
    """
    Load and return the full contents of a text-based file.

    Parameters
    ----------
    path : str or Path
        Path to the file to load. Supports .txt, .toon, .json, .yaml, etc.

    Returns
    -------
    str
        Raw text contents of the file.
    """
    return Path(path).read_text(encoding="utf-8")



def build_agent_prompt_text(agent_system_prompt: str, game_data_text: str) -> str:
    """
    Construct the full text prompt that will be sent to each LLM agent.

    Parameters
    ----------
    agent_system_prompt : str
        The base system prompt instructing the agent on behavior.
    game_data_text : str
        Contents of the GAME_DATA (TOON) file.

    Returns
    -------
    str
        A full assembled prompt including system prompt, game data,
        and final instructions on expected JSON output.
    """
    return (
        agent_system_prompt
        + "\n\n"
        + "----- GAME_DATA (TOON) -----\n"
        + game_data_text
        + "\n\n"
        + "You are part of a council predicting the outcome of this game.\n"
          "Return ONLY a single JSON object matching the agreed schema."
    )


def safe_json_load(s: str) -> Optional[Any]:
    """
    Safely attempt to parse a JSON string.

    Behavior:
    - First, try a normal json.loads().
    - If that fails and json_repair is installed, attempt to repair malformed JSON.
    - If all attempts fail, return None.

    Parameters
    ----------
    s : str
        Raw JSON-like string returned by an LLM.

    Returns
    -------
    dict or list or None
        Parsed JSON structure, repaired JSON, or None if parsing fails.
    """
    # First attempt: standard JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Second attempt: repaired JSON, if available
    if repair_json is not None:
        try:
            repaired = repair_json(s)
            return json.loads(repaired)
        except Exception:
            return None

    return None



def parse_council_df(council_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse a council dataframe containing raw model outputs into structured fields.

    Expected input:
    Each row of `council_raw` MUST contain at least:
        - completion_text : str  (raw JSON-ish output from the model)
        - model           : str  (optional; name of the model)

    Expected JSON structure returned by each model:
        {
          "winner_team_id": 1,
          "winner_team_name": "Texas",
          "projected_score": {"1": 27, "2": 24},
          "confidence_winner": 0.63,
          "confidence_score_band": "medium",
          "key_factors": [...],
          "risk_factors": [...]
        }

    Parameters
    ----------
    council_raw : pd.DataFrame
        Raw model outputs from the LLM council.

    Returns
    -------
    pd.DataFrame
        A cleaned, structured dataframe where each row corresponds to a model's
        prediction with validated fields, default fallbacks, and JSON-parsing status.
    """

    records = []

    for _, row in council_raw.iterrows():
        raw = row.get("completion_text", "")
        model_name = row.get("model", None)

        data = safe_json_load(raw)

        if data is None or not isinstance(data, dict):
            # JSON invalid — store minimal row
            records.append({
                "model": model_name,
                "raw": raw,
                "valid_json": False,
            })
            continue

        # Extract values with safe fallbacks
        projected_score = data.get("projected_score", {}) or {}

        records.append({
            "model": model_name,
            "valid_json": True,
            "winner_team_id": data.get("winner_team_id"),
            "winner_team_name": data.get("winner_team_name"),
            "score_team_1": projected_score.get("1"),
            "score_team_2": projected_score.get("2"),
            "confidence_winner": data.get("confidence_winner"),
            "confidence_score_band": data.get("confidence_score_band"),
            "key_factors": data.get("key_factors", []),
            "risk_factors": data.get("risk_factors", []),
        })

    return pd.DataFrame(records)
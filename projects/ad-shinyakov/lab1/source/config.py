import os
from pathlib import Path


class Config:
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    ollama_models = ["glm4:9b", "gemma3:12b", "granite4:3b"]
    modes = {
        "baseline": {},
        "precise": {"temperature": 0.1, "top_p": 0.5, "max_tokens": 700},
        "creative": {"temperature": 1.2, "top_p": 0.95, "max_tokens": 700},
        "hot_narrow": {"temperature": 1.5, "top_p": 0.2, "max_tokens": 700},
        "anti_repeat": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 700,
            "frequency_penalty": 0.8,
            "presence_penalty": 0.5,
        },
        "short_seeded": {
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 100,
            "seed": 42,
        },
    }
    lab_dir = Path(__file__).resolve().parent.parent
    results_path = lab_dir / "results" / "runs.json"
    plots_dir = lab_dir / "plots"

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from prompts import PROMPTS
from storage import load_results


MODE_LABELS = {
    "baseline": "Базовый",
    "precise": "Точный",
    "creative": "Креативный",
    "hot_narrow": "Горячий + узкий top_p",
    "anti_repeat": "Штрафы за повторы",
    "short_seeded": "Короткий + seed",
}
PROMPT_LABELS = {
    "generation": "Генерация",
    "classification": "Классификация",
    "extraction": "Извлечение",
}


def plot_metric(results, metric, ylabel, filename, prompt_name):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(Config.ollama_models))
    width = 0.8 / len(Config.modes)
    indexed = {
        (run["model"], run["prompt_name"], run["mode"]): run
        for run in results
        if run["status"] == "ok"
    }
    for index, mode in enumerate(Config.modes):
        offset = (index - (len(Config.modes) - 1) / 2) * width
        values = []
        for model in Config.ollama_models:
            run = indexed.get((model, prompt_name, mode))
            value = run.get(metric) if run else None
            values.append(value if value is not None else np.nan)
        bars = ax.bar(x + offset, values, width, label=MODE_LABELS[mode])
        labels = [
            f"{value:.1f}" if np.isfinite(value) else ""
            for value in values
        ]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_title(PROMPT_LABELS[prompt_name])
    ax.set_xticks(x)
    ax.set_xticklabels(Config.ollama_models, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    Config.plots_dir.mkdir(parents=True, exist_ok=True)
    path = Config.plots_dir / prompt_name / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"График сохранён: {path}")


def main():
    results = load_results(Config.results_path)
    for prompt_name in PROMPTS:
        plot_metric(
            results, "output_tokens", "Выходные токены",
            "output_tokens.png", prompt_name,
        )
        plot_metric(
            results, "elapsed_seconds", "Полное время запроса, с",
            "response_time.png", prompt_name,
        )


if __name__ == "__main__":
    main()

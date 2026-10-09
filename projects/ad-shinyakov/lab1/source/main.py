from config import Config
from generation import generate
from prompts import PROMPTS
from storage import save_results


def main():
    results = []
    for model in Config.ollama_models:
        print(f"\nПрогрев: {model}")
        try:
            generate(model, "Ответь одним словом: привет.", {})
        except Exception as error:
            print(f"Ошибка прогрева: {error}")

        for prompt_name, prompt in PROMPTS.items():
            for mode, parameters in Config.modes.items():
                print(f"Запрос: {model} | {prompt_name} | {mode}")
                result = {
                    "model": model,
                    "prompt_name": prompt_name,
                    "prompt": prompt,
                    "mode": mode,
                    "parameters": parameters.copy(),
                }
                try:
                    result.update(generate(model, prompt, parameters))
                    result["status"] = "ok"
                    print(result["answer"])
                except Exception as error:
                    result["status"] = "error"
                    result["error"] = str(error)
                    print(f"Ошибка: {error}")
                results.append(result)
                save_results(results, Config.results_path)
    print(f"\nРезультаты сохранены: {Config.results_path}")


if __name__ == "__main__":
    main()

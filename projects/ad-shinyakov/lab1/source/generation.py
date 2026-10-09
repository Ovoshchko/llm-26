from time import perf_counter

from openai import OpenAI

from config import Config


client = OpenAI(
    api_key=Config.ollama_api_key,
    base_url=Config.ollama_base_url,
    max_retries=0,
)


def generate(model, prompt, parameters):
    started = perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **parameters,
    )
    elapsed = perf_counter() - started
    choice = response.choices[0]
    usage = response.usage
    return {
        "answer": choice.message.content,
        "finish_reason": choice.finish_reason,
        "elapsed_seconds": elapsed,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }

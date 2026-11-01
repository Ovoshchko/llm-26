import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.llm import GenerationConfig, OllamaGenerator
from rag.prompt import REVIEW
from rag.schemas import EvidenceReview


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate GLM evidence review on fixed quotations")
    parser.add_argument("--cases", type=Path, default=root / "tests/fixtures/evidence_review_cases.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="glm4:9b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()
    output = args.output or root / "data/experiments/evidence_review" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_review_prompt")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    generator = OllamaGenerator(GenerationConfig(model=args.model, url=args.ollama_url, verify_evidence=True))
    try:
        for case in json.loads(args.cases.read_text()):
            messages = [{"role": "system", "content": REVIEW},
                        {"role": "user", "content": json.dumps(case["payload"], ensure_ascii=False)}]
            started = perf_counter()
            record = {**case, "messages": messages}
            try:
                content, usage = generator.generate(messages, EvidenceReview)
                review = EvidenceReview.model_validate_json(content)
                sources = {s["source_id"] for s in case["payload"]["sources"]}
                valid = (review.requirement_id == case["payload"]["requirement_id"]
                         and set(review.source_ids).issubset(sources)
                         and (review.status != "supported" or bool(review.source_ids and review.answer and review.answer.strip())))
                record.update(review=review.model_dump(), usage=usage,
                              passed=valid and (review.status == "supported") == case["expected_supported"])
            except (ValueError, RuntimeError) as error:
                record.update(error=str(error), passed=False)
            record["seconds"] = perf_counter() - started
            results.append(record)
            (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
            print(case["id"], "PASS" if record["passed"] else "FAIL", flush=True)
    finally:
        generator.close()
    print(output)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
import unittest
from unittest.mock import Mock, patch

import httpx

from rag.llm import GenerationConfig, OllamaGenerator
from rag.tokenizer import ChatTokenCounter


class TokenCounterTests(unittest.TestCase):
    def test_loads_tokenizer_once_and_counts_full_chat_template(self):
        messages = [{"role": "system", "content": "Инструкция"},
                    {"role": "user", "content": "Вопрос и источники"}]
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [151331, 151333, 42, 151337]
        with patch("rag.tokenizer.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
            counter = ChatTokenCounter()
            self.assertEqual(counter.count(messages), 4)
            self.assertEqual(counter.count(messages), 4)
            load.assert_called_once_with(counter.model, revision=counter.revision, trust_remote_code=False)
            tokenizer.apply_chat_template.assert_called_with(
                messages, tokenize=True, add_generation_prompt=True,
            )

    def test_context_boundary_and_mismatched_server_count(self):
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            self.assertFalse(payload["truncate"])
            self.assertFalse(payload["shift"])
            requests.append(payload)
            return httpx.Response(200, json={"done": True, "message": {"content": "{}"},
                                             "prompt_eval_count": 31744})

        with httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler)) as client:
            generator = OllamaGenerator(GenerationConfig(), client)
            with patch.object(generator.counter, "count", return_value=31744):
                generator.generate([])
            with patch.object(generator.counter, "count", return_value=31745):
                with self.assertRaises(ValueError):
                    generator.generate([])
            self.assertEqual(len(requests), 1)
            with patch.object(generator.counter, "count", return_value=31743):
                with self.assertRaises(RuntimeError):
                    generator.generate([])


if __name__ == "__main__":
    unittest.main()

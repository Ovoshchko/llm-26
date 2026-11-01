from transformers import AutoTokenizer


class ChatTokenCounter:
    def __init__(self, model="zai-org/glm-4-9b-chat-hf",
                 revision="8599336fc6c125203efb2360bfaf4c80eef1d1bf"):
        self.model = model
        self.revision = revision
        self.tokenizer = None

    def count(self, messages):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, revision=self.revision, trust_remote_code=False,
            )
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        ))

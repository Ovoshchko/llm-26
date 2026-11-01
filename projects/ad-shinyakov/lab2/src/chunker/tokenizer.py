class TokenCounter:
    def __init__(self, model):
        from transformers import AutoTokenizer

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

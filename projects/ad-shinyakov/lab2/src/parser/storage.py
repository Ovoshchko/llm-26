from pathlib import Path


class FileStorage:
    def __init__(self, root):
        self.root = Path(root)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

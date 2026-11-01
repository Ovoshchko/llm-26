import hashlib
import subprocess
import tempfile
from pathlib import Path


class RepositoryDownloader:
    repository_url = "https://github.com/kubernetes/website.git"

    def __init__(self, cache):
        self.cache = Path(cache).resolve()

    def download(self, ref):
        key = hashlib.sha256(ref.encode()).hexdigest()[:16]
        destination = self.cache / key
        if (destination / ".git").is_dir():
            return destination
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="clone-", dir=self.cache) as temporary:
            checkout = Path(temporary) / "website"
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--single-branch", "--branch", ref,
                 "--", self.repository_url, str(checkout)],
                capture_output=True, text=True,
            )
            if result.returncode:
                raise ValueError(f"Git clone failed: {result.stderr.strip()}")
            checkout.rename(destination)
        return destination

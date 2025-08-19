import requests
import logging

logger = logging.getLogger(__name__)

class TEITokenizer:
    def __init__(self, global_config, base_url: str = "http://embeddings:8080", timeout: int = 30):
        logger.debug("Global Config: {global_config}")
        self.global_config = global_config
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def encode(self, text: str) -> list[int]:
        """
        Tokenize a single string and return list of token IDs.
        """
        url = f"{self.base_url}/tokenize"
        try:
            resp = requests.post(url, json={"inputs": text}, timeout=self.timeout)
            resp.raise_for_status()
            tokens = resp.json()
            if tokens and isinstance(tokens[0], list):
                return [tok["id"] for tok in tokens[0]]
            return []
        except Exception as e:
            logger.error(f"TEI tokenization failed: {e}")
            return []

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """
        Tokenize multiple strings and return list of lists of token IDs.
        """
        url = f"{self.base_url}/tokenize"
        try:
            resp = requests.post(url, json={"inputs": texts}, timeout=self.timeout)
            resp.raise_for_status()
            all_tokens = resp.json()
            return [[tok["id"] for tok in seq] for seq in all_tokens]
        except Exception as e:
            logger.error(f"TEI batch tokenization failed: {e}")
            return [[] for _ in texts]

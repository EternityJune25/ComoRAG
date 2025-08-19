import requests
import logging

logger = logging.getLogger(__name__)

class TEITokenizer:
    def __init__(self, base_url: str = "http://embeddings:8080", timeout: int = 30):
        """
        base_url: Base URL of the TEI service (e.g. http://embeddings:8080 if running locally)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def encode(self, text: str) -> list[int]:
        """Tokenize a single string and return list of token IDs"""
        url = f"{self.base_url}/tokenize"
        payload = {"inputs": text}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            tokens = response.json()  # TEI returns tokenized ids
            return tokens[0] if isinstance(tokens, list) and tokens else []
        except Exception as e:
            logger.error(f"TEI tokenization failed: {e}")
            return []

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Tokenize multiple strings at once"""
        url = f"{self.base_url}/tokenize"
        payload = {"inputs": texts}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()  # list of token ID lists
        except Exception as e:
            logger.error(f"TEI batch tokenization failed: {e}")
            return [[] for _ in texts]

import numpy as np
import requests
import logging

logger = logging.getLogger(__name__)

class HFTEIEmbedding:
    def __init__(self, global_config, base_url: str = "http://embeddings:8080", embedding_model_name = "", timeout: int = 30):
        logger.debug("Global Config: {global_config}")
        logger.warning("input name: {embedding_model_name}. Make sure you did load it with HF TEI!")
        self.global_config = global_config
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def batch_encode(self, texts: list[str], instruction: str = None, norm: bool = True) -> np.ndarray:
        """
        Encode a batch of texts using Hugging Face TEI `/embed`.
        Returns numpy array of shape (batch_size, embedding_dim).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        url = f"{self.base_url}/embed"
        payload = {"inputs": texts, "normalize": norm}
        if instruction:
            payload["prompt_name"] = instruction

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            embeddings = resp.json()  # [[float, ...], [float, ...]]
            embeddings = [np.array(e, dtype=np.float32) for e in embeddings if e]
            if not embeddings:
                raise ValueError("Empty embeddings returned from TEI")
            return np.vstack(embeddings)
        except Exception as e:
            logger.error(f"TEI embedding failed: {e}")
            return np.zeros((len(texts), 1), dtype=np.float32)

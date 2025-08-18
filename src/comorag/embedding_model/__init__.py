
from .base import EmbeddingConfig, BaseEmbeddingModel
from .BGEEmbedding import BGEEmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "None"):
    if "bge-" in embedding_model_name.lower():
        return BGEEmbeddingModel
    elif "text-embedding-3-small" in embedding_model_name:
        return OpenAIEmbeddingModel
    else:
        logger.info(f"Unknown embedding model name: {embedding_model_name}, using BGEEmbeddingModel as default")
        return 

import os
from .HF_TEI import HFTEIEmbedding
from .BGEEmbedding import BGEEmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from ..utils.logging_utils import get_logger
from ..utils.config_utils import str_to_bool
logger = get_logger(__name__)

LOCAL_DOCKER = str_to_bool(os.getenv("LOCAL_DOCKER", "false"))
logger.info(f"LOCAL_DOCKER: {LOCAL_DOCKER} {type(LOCAL_DOCKER)}")


def _get_embedding_model_class(embedding_model_name: str = "None"):
    if "bge-" in embedding_model_name.lower():
        return BGEEmbeddingModel
    elif "text-embedding-3-small" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif LOCAL_DOCKER:
        return HFTEIEmbedding
    else:
        logger.info(f"Unknown embedding model name: {embedding_model_name}, using TEI as default")
        return

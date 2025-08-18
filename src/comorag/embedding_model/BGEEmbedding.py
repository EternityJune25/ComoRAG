from copy import deepcopy
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

def mean_pooling(token_embeddings, mask):
    """
    Perform mean pooling on token embeddings using attention mask
    
    Args:
        token_embeddings: Embeddings from the model
        mask: Attention mask from tokenizer
        
    Returns:
        Pooled sentence embeddings
    """
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class BGEEmbeddingModel(BaseEmbeddingModel):
    """
    Implementation of BGE embedding model based on the BGE-M3 architecture.
    BGE-M3 is a multilingual embedding model that supports 100+ languages
    with strong cross-lingual alignment and understanding.
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_model.eval()
        self.embedding_dim = self.embedding_model.config.hidden_size
        
        # Enable caching if specified in the config
        if hasattr(self.global_config, 'embedding_cache_enabled') and self.global_config.embedding_cache_enabled:
            cache_path = getattr(self.global_config, 'embedding_cache_path', 'bge_embeddings_cache.db')
            self.encode = make_cache_embed(self._encode, cache_path, self.embedding_model.device)
        else:
            self.encode = self._encode

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                'device_map': "auto",  # Use multiple GPUs if available
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                # BGE-specific prefixes
                "query_instruction": "Generate a representation for this sentence to retrieve relevant articles:",  # Instruction for queries
                "passage_instruction": "Generate a representation for this sentence to retrieve relevant articles:",  # Instruction for passages
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def _encode(self, prompts: List[str], **kwargs) -> torch.Tensor:
        """
        Internal encode method that interfaces directly with the model.
        
        Args:
            prompts: List of text prompts to encode
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Tensor of embeddings
        """
        if isinstance(prompts, str): 
            prompts = [prompts]
        
        # Process instructions if specified
        instruction = kwargs.get("instruction", "")
        if instruction:
            prompts = [instruction + text for text in prompts]
            
        with torch.no_grad():
            inputs = self.tokenizer(
                prompts, 
                padding=True, 
                truncation=True, 
                max_length=kwargs.get("max_length", self.embedding_config.encode_params.get("max_length", 512)),
                return_tensors='pt'
            ).to(self.embedding_model.device)
            
            outputs = self.embedding_model(**inputs)
            
            # BGE models typically use the last hidden state for embeddings
            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize embeddings if specified
            if kwargs.get("normalize", True):
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
        return embeddings

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            **kwargs: Additional parameters to override defaults
            
        Returns:
            NumPy array of embeddings
        """
        if isinstance(texts, str): 
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: 
            params.update(kwargs)

        # Handle query vs passage format
        if "is_query" in kwargs and kwargs["is_query"]:
            # Use query instruction for queries
            params["instruction"] = params.get("query_instruction", "Generate a representation for this sentence to retrieve relevant articles:")
        else:
            # Use passage instruction for documents
            params["instruction"] = params.get("passage_instruction", "Generate a representation for this sentence to retrieve relevant articles:")

        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        
        # Process in batches if needed
        if len(texts) <= batch_size:
            params["prompts"] = texts
            results = self.encode(**params)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                params["prompts"] = batch_texts
                batch_results = self.encode(**params)
                results.append(batch_results)
                pbar.update(len(batch_texts))
            pbar.close()
            results = torch.cat(results, dim=0)

        # Convert to NumPy and normalize if needed
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()

        # Normalize if specified in config and not already done in encode
        if self.embedding_config.norm and not kwargs.get("normalize", True):
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode queries with the appropriate instruction prefix.
        
        Args:
            queries: Query or list of queries to encode
            **kwargs: Additional parameters
            
        Returns:
            NumPy array of query embeddings
        """
        kwargs["is_query"] = True
        return self.batch_encode(queries, **kwargs)
    
    def encode_passages(self, passages: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode passages with the appropriate instruction prefix.
        
        Args:
            passages: Passage or list of passages to encode
            **kwargs: Additional parameters
            
        Returns:
            NumPy array of passage embeddings
        """
        kwargs["is_query"] = False
        return self.batch_encode(passages, **kwargs)
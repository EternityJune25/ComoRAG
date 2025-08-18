from typing import List, Tuple, Optional
import torch
from tqdm import tqdm
import numpy as np
from ..embedding_store import EmbeddingStore


def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000,
                 key_batch_size=10000):
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.
    
    Args:
        query_ids: List of query identifiers
        key_ids: List of key identifiers
        k: Number of top-k nearest neighbors to retrieve
        query_batch_size: Batch size for processing queries
        key_batch_size: Batch size for processing keys

    Returns:
        Dictionary mapping query_id to (top_k_key_ids, similarity_scores)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(key_vecs) == 0: return {}

    query_vecs = torch.tensor(query_vecs, dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

    key_vecs = torch.tensor(key_vecs, dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries"
    ):
        query_batch = query_batch.clone().detach()
        query_batch = query_batch.to(device)

        batch_topk_sim_scores = []
        batch_topk_indices = []

        offset_keys = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            key_batch = key_batch.to(device)
            actual_key_batch_size = key_batch.size(0)

            similarity = torch.mm(query_batch, key_batch.T)

            topk_sim_scores, topk_indices = torch.topk(similarity, min(k, actual_key_batch_size), dim=1, largest=True,
                                                       sorted=True)

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            key_batch = key_batch.cpu()
            torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size
        # End for each key batch

        batch_topk_sim_scores = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(batch_topk_sim_scores,
                                                               min(k, batch_topk_sim_scores.size(1)), dim=1,
                                                               largest=True, sorted=True)
        final_topk_indices = final_topk_indices.cpu()
        final_topk_sim_scores = final_topk_sim_scores.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]

            final_topk_indices_i = final_topk_indices[i]
            final_topk_sim_scores_i = final_topk_sim_scores[i]

            query_to_topk_key_relative_ids = batch_topk_indices[i][final_topk_indices_i]
            query_to_topk_key_ids = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
            results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

        query_batch = query_batch.cpu()
        torch.cuda.empty_cache()
    # End for each query batch

    return results

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0,1] range"""
    if len(scores) == 0:
        return scores
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)

def get_similar_summaries(
    query: str,
    level_store: EmbeddingStore,
    embedding_model,
    top_k: int = 3,
    instruction: Optional[str] = None
) -> Tuple[List[str], List[float]]:
    """
    Get summaries most similar to the query
    
    Args:
        query: Query text
        level_store: EmbeddingStore instance storing summaries for this level
        embedding_model: Embedding model
        top_k: Number of most similar summaries to return
        instruction: Optional instruction text
        
    Returns:
        Tuple[List[str], List[float]]: (List of most similar summaries, corresponding similarity scores)
    """
    # Get all summary IDs and texts for this level
    level_ids = level_store.get_all_ids()
    
    # Check if summaries are available
    if not level_ids:
        return [], []
    
    level_texts = [level_store.hash_id_to_text[id] for id in level_ids]
    
    # Get embedding vectors for all summaries at this level
    summary_embeddings = level_store.get_embeddings(level_ids)
    
    # Check if summary_embeddings is empty
    if len(summary_embeddings) == 0:
        return [], []
    
    # Get query embedding vector
    query_embedding = embedding_model.batch_encode(
        query,
        instruction='Given a question, retrieve relevant documents that best answer the question.',
        norm=True
    )
    
    # Calculate similarity scores
    similarity_scores = np.dot(summary_embeddings, query_embedding.T)
    similarity_scores = np.squeeze(similarity_scores) if similarity_scores.ndim == 2 else similarity_scores
    similarity_scores = min_max_normalize(similarity_scores)
    
    # Get top_k most similar summaries
    sorted_indices = np.argsort(similarity_scores)[::-1][:top_k]
    sorted_scores = similarity_scores[sorted_indices]
    
    return [level_texts[i] for i in sorted_indices], sorted_scores.tolist()
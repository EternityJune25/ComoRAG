import numpy as np
import logging
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set, Any
import umap
from sklearn.mixture import GaussianMixture
from copy import deepcopy
import pickle
from tqdm import tqdm

from .summarization_utils import (BaseSummarizationModel, GPT4SummarizationModel)

logger = logging.getLogger(__name__)

# Set random seed
RANDOM_SEED = 224
np.random.seed(RANDOM_SEED)

class SoftCluster:
    """Soft clustering cluster data structure"""
    def __init__(self, cluster_id: int, centroid: np.ndarray = None):
        self.id = cluster_id
        self.centroid = centroid
        self.members: Dict[str, float] = {}  # hash_id -> membership_score
        self.source_hash_ids: Set[str] = set()  # Store original chunks' hash_id
        
    def add_member(self, hash_id: str, membership_score: float):
        """Add member to cluster"""
        self.members[hash_id] = membership_score
        self.source_hash_ids.add(hash_id)
            
    def get_members_above_threshold(self, threshold: float) -> List[Tuple[str, float]]:
        """Get members with membership scores above threshold"""
        return [(hash_id, score) for hash_id, score in self.members.items() if score >= threshold]
    
    def __len__(self):
        return len(self.members)
    
    def __repr__(self):
        return f"SoftCluster(id={self.id}, members={len(self.members)})"


class ChunkSoftClustering:
    """
    RAG Chunk soft clustering implementation based on EmbeddingStore
    """
    def __init__(
        self, 
        embedding_store, 
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        max_clusters: int = 50,
        verbose: bool = True,
        db_filename: str = None,
        summarization_length=None,
        summarization_model=None,
        namespace: str = "soft_clusters",
        llm_model_name=None,
        llm_base_url="https://api.example.com/v1",
        llm_api_key=None
    ):
        """
        Initialize soft clustering
        
        Args:
            embedding_store: EmbeddingStore instance for getting and storing embeddings
            reduction_dimension: Dimension after reduction
            threshold: Membership threshold for assigning nodes to clusters
            max_clusters: Maximum allowed number of clusters
            verbose: Whether to output detailed logs
            db_filename: Directory path for storing clustering results
            namespace: Namespace for clustering results
        """
        self.embedding_store = embedding_store
        self.reduction_dimension = reduction_dimension
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.verbose = verbose
        self.namespace = namespace

        # Summary model
        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        if summarization_model is None:
            # Create summarization_model using passed configuration parameters
            summarization_model = GPT4SummarizationModel(
                model=llm_model_name,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key
            )
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        self.summarization_model = summarization_model
        
        # Store clustering state
        self.clusters: List[SoftCluster] = []
        self.hash_id_to_cluster_memberships: Dict[str, Dict[int, float]] = {}

        # Set result storage path
        if db_filename:
            self.db_filename = db_filename
            if not os.path.exists(self.db_filename):
                logger.info(f"Creating clusters directory: {self.db_filename}")
                os.makedirs(self.db_filename, exist_ok=True)
            
            self.clusters_file = os.path.join(
                self.db_filename, f"soft_clusters_{self.namespace}.pkl"
            )
            self.memberships_file = os.path.join(
                self.db_filename, f"cluster_memberships_{self.namespace}.parquet"
            )
            
            # Try to load existing clustering results
            self._load_clustering_results()
        else:
            self.db_filename = None

    def _load_clustering_results(self):
        """Load existing clustering results"""
        try:
            if os.path.exists(self.clusters_file):
                with open(self.clusters_file, 'rb') as f:
                    self.clusters = pickle.load(f)
                logger.info(f"Loaded {len(self.clusters)} clusters from {self.clusters_file}")
            
            if os.path.exists(self.memberships_file):
                memberships_df = pd.read_parquet(self.memberships_file)
                self.hash_id_to_cluster_memberships = {}
                
                for _, row in memberships_df.iterrows():
                    hash_id = row['hash_id']
                    cluster_memberships = {}
                    # Parse stored cluster memberships
                    memberships_dict = row['cluster_memberships']
                    for cluster_id, score in memberships_dict.items():
                        cluster_memberships[int(cluster_id)] = float(score)
                    
                    self.hash_id_to_cluster_memberships[hash_id] = cluster_memberships
                
                logger.info(f"Loaded cluster memberships for {len(self.hash_id_to_cluster_memberships)} chunks")
        except Exception as e:
            logger.warning(f"Error loading clustering results: {e}")
            self.clusters = []
            self.hash_id_to_cluster_memberships = {}
    
    def _save_clustering_results(self):
        """Save clustering results"""
        if not self.db_filename:
            return
        
        try:
            # Save cluster information
            with open(self.clusters_file, 'wb') as f:
                pickle.dump(self.clusters, f)
            # Save membership information
            memberships_data = []
            for hash_id, cluster_memberships in self.hash_id_to_cluster_memberships.items():
                memberships_data.append({
                    'hash_id': hash_id,
                    'cluster_memberships': cluster_memberships
                })
            
            memberships_df = pd.DataFrame(memberships_data)
            memberships_df.to_parquet(self.memberships_file, index=False)
            
            logger.info(f"Saved {len(self.clusters)} clusters and memberships for {len(self.hash_id_to_cluster_memberships)} chunks")
        except Exception as e:
            logger.warning(f"Error saving clustering results: {e}")
    
    def _get_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Determine optimal number of clusters"""
        max_clusters = min(self.max_clusters, len(embeddings) - 1)
        n_clusters = np.arange(1, max_clusters + 1)
        bics = []
        
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=RANDOM_SEED)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            
        optimal_clusters = n_clusters[np.argmin(bics)]
        if self.verbose:
            logger.info(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Use UMAP for dimension reduction"""
        n_neighbors = min(30, max(5, int(len(embeddings) * 0.2)))
        dim = min(self.reduction_dimension, len(embeddings) - 2)
        
        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=dim,
                metric="cosine",
                random_state=RANDOM_SEED
            )
            
            reduced_embeddings = reducer.fit_transform(embeddings)
            if self.verbose:
                logger.info(f"Reduced dimensions from {embeddings.shape[1]} to {dim}")
            return reduced_embeddings
        except Exception as e:
            logger.warning(f"Error during dimension reduction: {e}. Using original embeddings.")
            # If dimension reduction fails, return original embeddings
            return embeddings
        
    def perform_clustering(self, hash_ids: Optional[List[str]] = None) -> List[SoftCluster]:
        """
        Perform two-level soft clustering (global + local)
        
        Args:
            hash_ids: Chunk hash_ids to cluster, if None then use all chunks in embedding_store
            
        Returns:
            Clustering results, each cluster contains its members and membership scores
        """
        if hash_ids is None or len(hash_ids) == 0:
            hash_ids = self.embedding_store.get_all_ids()
        
        if len(hash_ids) <= 1:
            logger.warning("Insufficient data to perform clustering")
            if len(hash_ids) == 1:
                cluster = SoftCluster(0)
                cluster.add_member(hash_ids[0], 1.0)
                self.clusters = [cluster]
                self.hash_id_to_cluster_memberships = {hash_ids[0]: {0: 1.0}}
            return self.clusters
        
        # Get embeddings
        embeddings = np.array(self.embedding_store.get_embeddings(hash_ids))
        
        # Level 1: Global clustering
        
        # Dimension reduction for global embeddings
        if embeddings.shape[1] > self.reduction_dimension:
            try:
                reduced_embeddings_global = self._reduce_dimensions(embeddings)
            except Exception as e:
                logger.warning(f"Global dimension reduction error: {e}")
                reduced_embeddings_global = embeddings
        else:
            reduced_embeddings_global = embeddings
        
        # Global clustering
        n_global_clusters = self._get_optimal_clusters(reduced_embeddings_global)
        global_gmm = GaussianMixture(
            n_components=n_global_clusters,
            random_state=RANDOM_SEED,
            covariance_type='full'
        )
        global_gmm.fit(reduced_embeddings_global)
        
        # Get global membership scores
        global_membership_scores = global_gmm.predict_proba(reduced_embeddings_global)
        
        # Create global cluster assignments
        global_clusters = []
        for i in range(len(hash_ids)):
            # Get global clusters that current point belongs to
            global_cluster_ids = np.where(global_membership_scores[i] >= self.threshold)[0]
            global_clusters.append(global_cluster_ids)
        
        if self.verbose:
            logger.info(f"Global cluster count: {n_global_clusters}")
        
        # Initialize results
        self.clusters = []
        self.hash_id_to_cluster_memberships = {}
        total_clusters = 0
        
        # Level 2: Local clustering for each global cluster
        for i in range(n_global_clusters):
            # Get indices of points belonging to current global cluster
            global_cluster_indices = np.array([j for j, gc in enumerate(global_clusters) if i in gc])
            
            if len(global_cluster_indices) == 0:
                continue
            
            # Get all embeddings in current global cluster
            global_cluster_embeddings = embeddings[global_cluster_indices]
            global_cluster_hash_ids = [hash_ids[j] for j in global_cluster_indices]
            
            if self.verbose:
                logger.info(f"Node count in global cluster {i}: {len(global_cluster_embeddings)}")
            
            # Handle special case: too few nodes
            if len(global_cluster_embeddings) <= self.reduction_dimension + 1:
                # If only a few nodes, create a single local cluster
                local_cluster = SoftCluster(total_clusters)
                for hash_id in global_cluster_hash_ids:
                    local_cluster.add_member(hash_id, 1.0)
                    if hash_id not in self.hash_id_to_cluster_memberships:
                        self.hash_id_to_cluster_memberships[hash_id] = {}
                    self.hash_id_to_cluster_memberships[hash_id][total_clusters] = 1.0
                    
                self.clusters.append(local_cluster)
                total_clusters += 1
                continue
            
            # Local dimension reduction for global cluster
            try:
                reduced_embeddings_local = self._reduce_dimensions(global_cluster_embeddings)
            except Exception as e:
                logger.warning(f"Local dimension reduction error: {e}")
                reduced_embeddings_local = global_cluster_embeddings
            
            # Local clustering
            n_local_clusters = self._get_optimal_clusters(reduced_embeddings_local)
            local_gmm = GaussianMixture(
                n_components=n_local_clusters,
                random_state=RANDOM_SEED,
                covariance_type='full'
            )
            local_gmm.fit(reduced_embeddings_local)
            
            # Get local membership scores
            local_membership_scores = local_gmm.predict_proba(reduced_embeddings_local)
            
            if self.verbose:
                logger.info(f"Local cluster count in global cluster {i}: {n_local_clusters}")
            
            # Create local clusters
            for j in range(n_local_clusters):
                local_cluster = SoftCluster(total_clusters, local_gmm.means_[j])
                
                # Add members to local cluster
                for k, hash_id in enumerate(global_cluster_hash_ids):
                    if local_membership_scores[k, j] >= self.threshold:
                        local_cluster.add_member(hash_id, local_membership_scores[k, j])
                        
                        if hash_id not in self.hash_id_to_cluster_memberships:
                            self.hash_id_to_cluster_memberships[hash_id] = {}
                        self.hash_id_to_cluster_memberships[hash_id][total_clusters] = local_membership_scores[k, j]
                
                # Only add non-empty clusters
                if len(local_cluster.members) > 0:
                    self.clusters.append(local_cluster)
                
                total_clusters += 1
        
        if self.verbose:
            logger.info(f"Total cluster count: {total_clusters}")
            for i, cluster in enumerate(self.clusters):
                logger.info(f"Cluster {cluster.id}: {len(cluster.members)} members")
        
        # Save clustering results
        if self.db_filename:
            pass
            # self._save_clustering_results()
        
        return self.clusters

    def get_cluster_membership(self, hash_id: str) -> Dict[int, float]:
        """
        Get all clusters and membership scores for the specified chunk
        
        Args:
            hash_id: Chunk's hash_id
            
        Returns:
            {cluster_id: membership_score, ...}
        """
        return self.hash_id_to_cluster_memberships.get(hash_id, {})
    
    def get_cluster_by_id(self, cluster_id: int) -> Optional[SoftCluster]:
        """Get cluster by specified ID"""
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None
    
    def get_cluster_members(self, cluster_id: int, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Get all members of the specified cluster and their membership scores
        
        Args:
            cluster_id: Cluster ID
            threshold: Minimum membership threshold, if None then use the threshold set during initialization
            
        Returns:
            [(hash_id, membership_score), ...], sorted by membership score in descending order
        """
        if threshold is None:
            threshold = self.threshold
            
        cluster = self.get_cluster_by_id(cluster_id)
        if not cluster:
            return []
            
        members = [(hash_id, score) for hash_id, score in cluster.members.items() if score >= threshold]
        return sorted(members, key=lambda x: x[1], reverse=True)
    
    def get_related_chunks(self, hash_id: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Get chunks related to the specified chunk (through shared clusters)
        
        Args:
            hash_id: Target chunk's hash_id
            threshold: Minimum relatedness threshold, if not provided then use clustering threshold
            
        Returns:
            [(related_hash_id, relatedness_score), ...], sorted by relatedness in descending order
        """
        if threshold is None:
            threshold = self.threshold
            
        # Get clusters that current chunk belongs to
        cluster_memberships = self.get_cluster_membership(hash_id)
        if not cluster_memberships:
            return []
            
        related_chunks: Dict[str, float] = {}
        
        # For each cluster, find other members
        for cluster_id, membership_score in cluster_memberships.items():
            cluster = self.get_cluster_by_id(cluster_id)
            if not cluster:
                continue
                
            # Calculate relatedness between each member and target chunk
            for other_id, other_score in cluster.members.items():
                if other_id != hash_id:
                    # Relatedness is the weighted geometric mean of two membership scores
                    relatedness = (membership_score * other_score) ** 0.5
                    if relatedness >= threshold:
                        if other_id in related_chunks:
                            related_chunks[other_id] = max(related_chunks[other_id], relatedness)
                        else:
                            related_chunks[other_id] = relatedness
        
        # Sort by relatedness
        sorted_related = sorted(related_chunks.items(), key=lambda x: x[1], reverse=True)
        return sorted_related
    
    def get_cluster_texts(self, cluster_id: int, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Get member texts and their membership scores for the specified cluster
        
        Args:
            cluster_id: Cluster ID
            top_k: Maximum number of members to return, if None then return all members
            threshold: Minimum membership threshold, if None then use the threshold set during initialization
            
        Returns:
            [(text, membership_score), ...], sorted by membership score in descending order
        """
        members = self.get_cluster_members(cluster_id, threshold)
        if not members:
            return []
            
        if top_k is not None:
            members = members[:top_k]
            
        texts_with_scores = []
        for hash_id, score in members:
            try:
                text = self.embedding_store.get_row(hash_id)['content']
                texts_with_scores.append((text, score))
            except KeyError:
                logger.warning(f"Text for hash_id {hash_id} not found in embedding store")
                
        return texts_with_scores
    
    def get_cluster_stats(self) -> Dict:
        """Get clustering statistics"""
        stats = {
            'total_chunks': len(self.hash_id_to_cluster_memberships),
            'total_clusters': len(self.clusters),
            'avg_membership_per_chunk': 0,
            'cluster_sizes': {}
        }
        
        # Calculate average number of clusters each chunk belongs to
        if self.hash_id_to_cluster_memberships:
            clusters_per_chunk = [len(memberships) for memberships in self.hash_id_to_cluster_memberships.values()]
            stats['avg_membership_per_chunk'] = sum(clusters_per_chunk) / len(clusters_per_chunk)
            
        # Count cluster sizes
        for cluster in self.clusters:
            size = len(cluster.members)
            stats['cluster_sizes'][cluster.id] = size
        
        return stats
    
    def get_hash_ids_in_same_clusters(self, hash_id: str, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Get all chunks in the same clusters as the specified chunk and their association strength
        
        Args:
            hash_id: Target chunk's hash_id
            threshold: Minimum association strength threshold
            
        Returns:
            {related_hash_id: relatedness_score, ...}
        """
        related = self.get_related_chunks(hash_id, threshold)
        return {r_id: score for r_id, score in related}
    
    def find_chunks_by_similarity(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float, List[int]]]:
        """
        Find chunks based on text similarity and return their cluster information
        
        Args:
            query_text: Query text
            top_k: Maximum number of results to return
            
        Returns:
            [(hash_id, similarity_score, [cluster_ids]), ...]
        """
        # First get similar chunks in embedding store
        query_embedding = self.embedding_store.embedding_model.get_embedding(query_text)
        query_embedding = np.array(query_embedding)
        
        # Get all embeddings
        all_hash_ids = self.embedding_store.get_all_ids()
        all_embeddings = np.array(self.embedding_store.get_embeddings(all_hash_ids))
        
        # Calculate cosine similarity
        norm_query = np.linalg.norm(query_embedding)
        norm_embeddings = np.linalg.norm(all_embeddings, axis=1)
        cosine_similarities = np.dot(all_embeddings, query_embedding) / (norm_embeddings * norm_query)
        
        # Get top_k similar chunks
        top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            hash_id = all_hash_ids[idx]
            similarity = cosine_similarities[idx]
            
            # Get belonging clusters
            cluster_memberships = self.get_cluster_membership(hash_id)
            cluster_ids = list(cluster_memberships.keys())
            
            results.append((hash_id, similarity, cluster_ids))
            
        return results
    
    def create_cluster_summary(self, cluster_id: int) -> str:
        """
        Create summary for the specified cluster
        
        Args:
            cluster_id: Cluster ID
            summary_model: Model used for generating summaries
            
        Returns:
            Summary of cluster content
        """
        texts = self.get_cluster_texts(cluster_id)
        if not texts:
            return ""
        
        # Get texts after sorting by membership score
        sorted_texts = [text for text, _ in texts]
        
        # Concatenate texts
        combined_text = ""
        for text in sorted_texts:
            combined_text += f"{' '.join(text.splitlines())}\n\n"
            
        # Generate summary
        summary = self.summarization_model.summarize(combined_text, self.summarization_length)
        return summary
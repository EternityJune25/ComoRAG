from argparse import ArgumentTypeError
from dataclasses import dataclass, field
from hashlib import md5
from pickle import NONE
import time
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging
import torch

from wandb import summary

from .typing_utils import Triple
from .misc_utils import compute_mdhash_id

class NodeType:
    VER = "veridical"  # Original text fragments
    SEM = "semantical"  # semantical summaries
    EPI = "episodic"  # episodic summaries
    FUSION = "fusion"  # Fusion nodes

@dataclass
class MemoryNode:
    probe: str = None
    node_type: str = None
    original_content: List[str] = None
    content_hash: List[str] = None
    cue: str = None
    embedding: np.ndarray = None  
    
    def __post_init__(self):
        """Calculate hash values during initialization"""
        if self.original_content:
            self.update_hashes()
    
    def update_hashes(self):
        """Update hash values for all content using node_type as namespace"""
        if self.original_content:
            self.content_hash = [
                compute_mdhash_id(content, prefix=self.node_type + "-")
                for content in self.original_content
            ]
    
    def add_content(self, content: str):
        """Add new content and update hash values"""
        if self.original_content is None:
            self.original_content = []
        if self.content_hash is None:
            self.content_hash = []
        
        self.original_content.append(content)
        self.content_hash.append(compute_mdhash_id(content, prefix=self.node_type + "-"))
    
    def get_content_hashes(self) -> List[str]:
        """Get hash values for all content"""
        if not self.content_hash and self.original_content:
            self.update_hashes()
        return self.content_hash if self.content_hash is not None else []
    
    def get_full_content(self) -> str:
        """Get complete node content for generating embeddings"""
        content_parts = []
        if self.probe:
            content_parts.append(f"probe: {self.probe}")
        if self.cue:
            content_parts.append(f"Note: {self.cue}")
        if self.original_content:
            content_parts.append(f"Content: {' '.join(self.original_content)}")
        return " ".join(content_parts)

@dataclass
class MemoryPool:
    pool: List[MemoryNode] = None
    temp_pool: List[MemoryNode] = None  
    embedding_model: Any = None  
    agent: Any = None  
    
    def __init__(self, embedding_model=None, agent=None):
        self.pool = []
        self.temp_pool = []  # Initialize temporary memory pool
        self.embedding_model = embedding_model
        self.agent = agent
    
    def add_node(self, node: MemoryNode):
        """Add node to memory pool"""
        self.pool.append(node)
    
    def add_to_temp_pool(self, node: MemoryNode):
        """Add node to temporary memory pool"""
        self.temp_pool.append(node)
    
    def clear_temp_pool(self):
        """Clear temporary memory pool"""
        self.temp_pool = []
    
    def merge_temp_to_main(self):
        """Merge nodes from temporary memory pool to main memory pool"""
        self.pool.extend(self.temp_pool)
        # Count how many temp nodes were merged
        print(f"Merged {len(self.temp_pool)} temporary memories")
        print(f"Node count in main memory pool after merge: {len(self.pool)}")
        self.clear_temp_pool()
    
    def get_temp_nodes_by_type(self, node_type: str) -> List[MemoryNode]:
        """Get all nodes of specified type from temporary memory pool"""
        return [node for node in self.temp_pool if node.node_type == node_type]
    
    def get_temp_nodes(self) -> List[MemoryNode]:
        """Get all nodes from temporary memory pool"""
        return self.temp_pool
    
    def get_nodes_by_hash(self, hash_value: str) -> List[MemoryNode]:
        """Get nodes by hash value"""
        return [
            node for node in self.pool 
            if hash_value in node.get_content_hashes()
        ]
    
    def get_nodes_by_type(self, node_type: str) -> List[MemoryNode]:
        """Get all nodes of specified type"""
        return [node for node in self.pool if node.node_type == node_type]
    
    def get_all_nodes(self) -> List[MemoryNode]:
        """Get all nodes"""
        return self.pool
    
    def get_all_hashes(self) -> Dict[str, List[str]]:
        """Get hash values for all nodes, grouped by node type
        
        Returns:
            Dict[str, List[str]]: Keys are node types, values are lists of hash values for all nodes of that type
        """
        hash_dict = {}
        for node in self.pool:
            if node.node_type not in hash_dict:
                hash_dict[node.node_type] = []
            hash_dict[node.node_type].extend(node.get_content_hashes())
        return hash_dict
    
    def get_all_probes(self) -> List[str]:
        """Get probe content from all nodes and remove duplicates
        
        Returns:
            List[str]: List of deduplicated probe content
        """
        return list(set(node.probe for node in self.pool if node.probe))
    
    def compute_probe_note_embeddings(self, force_recompute: bool = False):
        """Compute embedding values for probe + atomic notes for all nodes
        
        Args:
            force_recompute: Whether to force recomputation of nodes that already have embeddings
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not provided")
        
        nodes_to_compute = []
        for node in self.pool:
            if node.embedding is None or force_recompute:
                nodes_to_compute.append(node)
        
        if nodes_to_compute:
            # Batch get probe + atomic note content
            contents = []
            for node in nodes_to_compute:
                content_parts = []
                if node.probe:
                    content_parts.append(node.probe)
                if node.cue:
                    content_parts.append(node.cue)
                # If neither exists, use empty string
                contents.append(" ".join(content_parts) if content_parts else "")
            
            # Batch generate embeddings
            embeddings = self.embedding_model.encode(contents)
            
            # Ensure embeddings are on CPU and convert to numpy arrays
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            
            # Update node embeddings
            for node, embedding in zip(nodes_to_compute, embeddings):
                node.embedding = embedding
                
        logging.info(f"Computed embeddings for {len(nodes_to_compute)} nodes")
    
    def retrieve_similar_nodes(self, current_probe: str, top_percent: float = 0.5) -> List[MemoryNode]:
        """Retrieve similar historical nodes based on current probe
        
        Args:
            current_probe: Current round's probe
            top_percent: Return top percentage of nodes with highest similarity (0-1)
            
        Returns:
            List[MemoryNode]: List of nodes with highest similarity
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not provided")
        
        # Ensure all nodes have embeddings
        self.compute_probe_note_embeddings()
        
        # Generate embedding for current probe
        probe_embedding = self.embedding_model.encode([current_probe])[0]
        
        # Ensure probe_embedding is on CPU
        if isinstance(probe_embedding, torch.Tensor):
            probe_embedding = probe_embedding.detach().cpu().numpy()
        
        # Calculate similarity with all historical nodes
        similarities = []
        for node in self.pool:
            if node.embedding is not None:
                # Ensure node.embedding is on CPU
                node_embedding = node.embedding
                if isinstance(node_embedding, torch.Tensor):
                    node_embedding = node_embedding.detach().cpu().numpy()
                
                # Calculate cosine similarity
                similarity = np.dot(probe_embedding, node_embedding) / (
                    np.linalg.norm(probe_embedding) * np.linalg.norm(node_embedding)
                )
                similarities.append((node, similarity))
        
        # Sort by similarity (high to low)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate number of nodes to return
        k = max(1, int(len(similarities) * top_percent))
        selected_nodes = [node for node, sim in similarities[:k]]
        
        logging.info(f"Retrieved {len(selected_nodes)} similar nodes (total {len(self.pool)} nodes)")
        
        return selected_nodes
    
 
    def create_fusion_content(self, probe: str, top_k_percent: float = 0.2) -> str:
        """Fuse related nodes based on probe
        
        Args:
            probe: Probe used for retrieval
            top_k_percent: Use top percentage of nodes with highest similarity for fusion
            
        Returns:
            str: Fused content
        """
        if not self.agent:
            raise ValueError("Agent not provided for fusion")
        
        # 1. Retrieve similar nodes
        similar_nodes = self.retrieve_similar_nodes(probe, top_k_percent)
        
        if not similar_nodes:
            return "No relevant memory nodes found for the given probe."
        
        # 2. Prepare node content
        nodes_content = []
        for i, node in enumerate(similar_nodes, 1):
            content_parts = []
            if node.cue:
                content_parts.append(f"Note: {node.cue}")
            
            nodes_content.append(f"Node {i}:\n" + "\n".join(content_parts))
        
        # 3. Prepare content
        content = "\n\n".join(nodes_content)
        
        # 4. Call agent for fusion
        fused_content = self.agent.fuse_memory_nodes(
            query=probe,
            content=content,
            max_completion_tokens=1000
        )
        
        return fused_content
    
    def add_fused_node(self, probe: str, fused_content: str, source_nodes: List[MemoryNode]):
        """Add fused content as new node to memory pool
        
        Args:
            probe: Probe used for fusion
            fused_content: Fused content
            source_nodes: List of source nodes
        """
        # Node fusion
        fused_node = MemoryNode(
            probe=probe,
            node_type=NodeType.FUSION,  
            original_content=None,  
            cue=fused_content  
        )
        
        # Compute embedding
        if self.embedding_model:
            embedding_content = f"{fused_content}"
            fused_node.embedding = self.embedding_model.encode([embedding_content])[0]
        
        # Add to temporary memory pool
        self.add_to_temp_pool(fused_node)
        
        logging.info(f"Node count in main memory pool after adding fused_node: {len(self.pool)}")
        
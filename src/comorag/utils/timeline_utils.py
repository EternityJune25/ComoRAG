import numpy as np
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..embedding_store import EmbeddingStore
from .summarization_utils import BaseSummarizationModel
import os
import tiktoken
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TimelineSummarizer:
    def __init__(
        self,
        chunk_embedding_store,
        summary_embedding_store,
        summarization_model: BaseSummarizationModel,
        window_size: Optional[int] = None,
        max_workers: int = 8
    ):
        """
        Initialize timeline summarizer
        
        Args:
            chunk_embedding_store: Vector database storing text chunks
            summary_embedding_store: Vector database storing summaries
            summarization_model: Model used for generating summaries
            window_size: Size of sliding window, if None then auto-calculated
            max_workers: Maximum number of threads, default is 8
        """
        self.chunk_store = chunk_embedding_store
        self.summary_store = summary_embedding_store
        self.summarization_model = summarization_model
        self.max_workers = max_workers
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Encoder used by GPT-4
        
        # Calculate appropriate window size
        all_ids = self.chunk_store.get_all_ids()
        total_chunks = len(all_ids)
        if window_size is None:
            # Dynamically calculate window size based on total chunks
            if total_chunks <= 5:
                window_size = 2
            elif total_chunks <= 20:
                window_size = 3
            elif total_chunks <= 50:
                window_size = 5
            elif total_chunks <= 100:
                window_size = 8
            elif total_chunks <= 200:
                window_size = 10
            else:
                # For larger datasets, use logarithmic scaling
                window_size = min(20, max(10, int(np.log2(total_chunks) * 2)))
        
        self.window_size = window_size
        logging.info(f"Total chunks: {total_chunks}, using window size: {window_size}, max workers: {max_workers}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text
        
        Args:
            text: Input text
            
        Returns:
            int: Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics
        
        Returns:
            Dict[str, Any]: Dictionary containing statistics
        """
        stats = {
            "total_levels": 0,
            "levels": [],
            "total_tokens": 0,
            "total_nodes": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "window_size": self.window_size,
                "max_workers": self.max_workers,
                "total_chunks": len(self.chunk_store.get_all_ids())
            }
        }
        
        # 获取所有层级的总结
        level = 0
        while True:
            try:
                summaries = self.get_summary_by_level(level)
                level_stats = {
                    "level": level,
                    "node_count": len(summaries),
                    "total_tokens": sum(self._count_tokens(s) for s in summaries),
                    "avg_tokens_per_node": float(np.mean([self._count_tokens(s) for s in summaries])),
                    "min_tokens": min(self._count_tokens(s) for s in summaries),
                    "max_tokens": max(self._count_tokens(s) for s in summaries),
                    "total_chars": sum(len(s) for s in summaries),
                    "avg_chars_per_node": float(np.mean([len(s) for s in summaries]))
                }
                
                stats["levels"].append(level_stats)
                stats["total_tokens"] += level_stats["total_tokens"]
                stats["total_nodes"] += level_stats["node_count"]
                level += 1
            except ValueError:
                break
        
        stats["total_levels"] = level
        stats["avg_tokens_per_level"] = float(stats["total_tokens"] / level if level > 0 else 0)
        stats["avg_nodes_per_level"] = float(stats["total_nodes"] / level if level > 0 else 0)
        
        return stats
    
    def save_summary_statistics(self, output_dir: str = "./summary_statistics"):
        """
        Save summary statistics to file
        
        Args:
            output_dir: Output directory
        """
        stats = self.get_summary_statistics()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename (using timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(output_dir, f"summary_stats_{timestamp}.json")
        txt_file = os.path.join(output_dir, f"summary_stats_{timestamp}.txt")
        
        # Save statistics in JSON format
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Save human-readable statistics
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=== Summary Statistics ===\n")
            f.write(f"Generation Time: {stats['timestamp']}\n")
            f.write(f"Total Levels: {stats['total_levels']}\n")
            f.write(f"Total Nodes: {stats['total_nodes']}\n")
            f.write(f"Total Tokens: {stats['total_tokens']}\n")
            f.write(f"Average Tokens per Level: {stats['avg_tokens_per_level']:.2f}\n")
            f.write(f"Average Nodes per Level: {stats['avg_nodes_per_level']:.2f}\n")
            
            f.write("\n=== Model Configuration ===\n")
            f.write(f"Window Size: {stats['model_info']['window_size']}\n")
            f.write(f"Max Worker Threads: {stats['model_info']['max_workers']}\n")
            f.write(f"Initial Text Chunks: {stats['model_info']['total_chunks']}\n")
            
            f.write("\n=== Level Details ===\n")
            for level in stats["levels"]:
                f.write(f"\nLevel {level['level']}:\n")
                f.write(f"  Node Count: {level['node_count']}\n")
                f.write(f"  Total Tokens: {level['total_tokens']}\n")
                f.write(f"  Average Tokens per Node: {level['avg_tokens_per_node']:.2f}\n")
                f.write(f"  Min Tokens: {level['min_tokens']}\n")
                f.write(f"  Max Tokens: {level['max_tokens']}\n")
                f.write(f"  Total Characters: {level['total_chars']}\n")
                f.write(f"  Average Characters per Node: {level['avg_chars_per_node']:.2f}\n")
        
        # Save summaries for each level to separate files
        for level in range(stats["total_levels"]):
            summaries = self.get_summary_by_level(level)
            level_dir = os.path.join(output_dir, f"level_{level}")
            os.makedirs(level_dir, exist_ok=True)
            
            # Save all summaries for this level to one file
            level_file = os.path.join(level_dir, f"summaries_{timestamp}.txt")
            with open(level_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Level {level} Summary Content ===\n")
                f.write(f"Generation Time: {stats['timestamp']}\n")
                f.write(f"Node Count: {len(summaries)}\n\n")
                for i, summary in enumerate(summaries):
                    f.write(f"Node {i+1}:\n")
                    f.write(f"{summary}\n")
                    f.write("\n" + "="*50 + "\n\n")
        
        logger.info(f"Statistics saved to: {json_file} and {txt_file}")
        logger.info(f"Level summaries saved to: {os.path.join(output_dir, 'level_*')}")
        return json_file, txt_file
    

    
    def _create_summary_prompt(self, texts: List[str], is_final_summary: bool = False) -> str:
        """
        Create summary prompt
        
        Args:
            texts: List of texts to be summarized
            is_final_summary: Whether this is the final summary
            
        Returns:
            str: Formatted prompt
        """
        combined_text = "\n\n".join(texts)
        if is_final_summary:
            return f"""Please provide a comprehensive summary of the following text, creating a coherent and complete narrative:

{combined_text}

Please provide a complete summary that ensures:
1. Maintains chronological order and logical coherence
2. Highlights important events, turning points, and key figures
3. Preserves all important details and causal relationships
4. Uses clear and fluent language
5. Forms a complete narrative rather than a simple list of points
6. Ensures the completeness and coherence of the summary, allowing readers to fully understand the entire story
"""
        else:
            return f"""Please summarize the following text, maintaining timeline coherence, highlighting key events while preserving important information:

{combined_text}

Please provide a coherent summary that ensures:
1. Maintains chronological order
2. Highlights important events and turning points
3. Preserves key details
4. Uses clear language
"""
    
    def _summarize_window(self, texts: List[str], is_final_summary: bool = False) -> str:
        """
        Summarize text within a window
        
        Args:
            texts: List of texts within the window
            is_final_summary: Whether this is the final summary
            
        Returns:
            str: Generated summary
        """
        prompt = self._create_summary_prompt(texts, is_final_summary)
        return self.summarization_model.summarize(prompt)
    
    def _process_window(self, window_texts: List[str], window_index: int, is_final_summary: bool = False) -> tuple[int, str]:
        """
        Process text within a single window
        
        Args:
            window_texts: List of texts within the window
            window_index: Window index
            is_final_summary: Whether this is the final summary
            
        Returns:
            tuple[int, str]: (window index, summary text)
        """
        if len(window_texts) > 1:
            summary = self._summarize_window(window_texts, is_final_summary)
            return window_index, summary
        return window_index, window_texts[0]
    
    def _generate_final_summary(self, texts: List[str]) -> str:
        """
        Generate final summary
        
        Args:
            texts: List of texts to be summarized
            
        Returns:
            str: Generated final summary
        """
        # If text count is small, generate summary directly
        if len(texts) <= 3:
            return self._summarize_window(texts, is_final_summary=True)
        
        # Otherwise, first generate preliminary summaries, then final summary
        # Use larger window size for preliminary summaries
        window_size = min(5, len(texts))
        preliminary_summaries = []
        
        for i in range(0, len(texts), window_size):
            window_texts = texts[i:i + window_size]
            summary = self._summarize_window(window_texts, is_final_summary=False)
            preliminary_summaries.append(summary)
        
        # Generate final summary from preliminary summaries
        return self._summarize_window(preliminary_summaries, is_final_summary=True)
    
    def generate_timeline_summary(self) -> Dict[str, Any]:
        """
        Generate timeline summary, only generate one level of summary
        
        Returns:
            Dict[str, Any]: Dictionary containing summaries
        """
        # Get all text chunks
        all_ids = self.chunk_store.get_all_ids()
        all_texts = [self.chunk_store.hash_id_to_text[id] for id in all_ids]
        total_chunks = len(all_texts)
        
        # Store summaries
        summaries_by_level = []
        window_tasks = []
        
        # Use fixed window size
        window_size = self.window_size
        
        # Prepare window tasks
        for i in range(0, len(all_texts), window_size):
            window_texts = all_texts[i:i + window_size]
            window_tasks.append((window_texts, i // window_size))
        
        # Use thread pool to process windows in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_window = {
                executor.submit(self._process_window, texts, idx, True): (texts, idx)
                for texts, idx in window_tasks
            }
            
            # Collect results
            window_results = []
            for future in as_completed(future_to_window):
                try:
                    window_idx, summary = future.result()
                    window_results.append((window_idx, summary))
                except Exception as e:
                    logger.error(f"Window processing failed: {e}")
                    texts, idx = future_to_window[future]
                    window_results.append((idx, texts[0] if len(texts) == 1 else ""))
        
        # Sort results by window index
        window_results.sort(key=lambda x: x[0])
        level_summaries = [summary for _, summary in window_results]
        
        summaries_by_level.append(level_summaries)
        
        # Store summaries
        level_store = EmbeddingStore(
            embedding_model=self.summary_store.embedding_model,
            db_filename=os.path.dirname(self.summary_store.filename),
            batch_size=self.summary_store.batch_size,
            namespace="level_0"
        )
        level_store.insert_strings(level_summaries)
        
        return {
            "total_levels": 1,
            "summaries_by_level": summaries_by_level
        }

    def get_summary_by_level(self, level: int) -> List[str]:
        """
        Get summaries for specified level
        
        Args:
            level: Level index
            
        Returns:
            List[str]: List of summaries for that level
        """
        # Create EmbeddingStore instance for corresponding level
        level_store = EmbeddingStore(
            embedding_model=self.summary_store.embedding_model,
            db_filename=os.path.dirname(self.summary_store.filename),
            batch_size=self.summary_store.batch_size,
            namespace=f"level_{level}"
        )
        
        level_ids = level_store.get_all_ids()
        if not level_ids:
            raise ValueError(f"Level {level} exceeds maximum level")
        
        return [level_store.hash_id_to_text[id] for id in level_ids]
    
    def get_level_embedding_store(self, level: int, output_dir: str = None):
        """
        Get EmbeddingStore instance for specified level
        
        Args:
            level: Level index
            output_dir: Output directory, if None then use summary_store's directory
        """
        if output_dir is None:
            output_dir = os.path.dirname(self.summary_store.filename)
            
        return EmbeddingStore(
            embedding_model=self.summary_store.embedding_model,
            db_filename=output_dir,
            batch_size=self.summary_store.batch_size,
            namespace=f"level_{level}"
        )

    def load_and_validate_summaries(self, output_dir: str) -> bool:
        """Try to load and validate summary statistics"""
        try:
            if not os.path.exists(output_dir):
                logger.warning(f"Timeline embedding directory does not exist: {output_dir}")
                return False
                
            # Check and load summaries for each level
            parquet_files = [f for f in os.listdir(output_dir) if f.startswith("vdb_level_") and f.endswith(".parquet")]
            if not parquet_files:
                logger.warning(f"No timeline embedding files found in: {output_dir}")
                return False
                
            # Validate summaries for each level
            for parquet_file in parquet_files:
                level = int(parquet_file.split("_")[2].split(".")[0])
                level_store = self.get_level_embedding_store(level, output_dir)
                if not level_store or not level_store.get_all_ids():
                    logger.warning(f"Timeline embedding validation failed for level {level}")
                    return False
                    
            logger.info(f"Successfully loaded and validated timeline embeddings for {len(parquet_files)} levels")
            return True
            
        except Exception as e:
            logger.error(f"Error occurred while loading timeline embeddings: {str(e)}")
            return False
    
    def calculate_expected_summaries(self, total_chunks: int) -> int:
        """
        Calculate expected total number of summaries, only calculate one level
        
        Args:
            total_chunks: Number of text chunks in current level
            
        Returns:
            int: Expected total number of summaries
        """
        window_size = self.window_size
        return (total_chunks + window_size - 1) // window_size

    def load_all_summaries(self) -> Dict[int, List[str]]:
        """
        Load summary content for all levels
        
        Returns:
            Dict[int, List[str]]: Dictionary with level as key and summary list for that level as value
        """
        summaries_by_level = {}
        level = 0
        
        while True:
            try:
                # Get summaries for current level
                level_summaries = self.get_summary_by_level(level)
                summaries_by_level[level] = level_summaries
                level += 1
            except ValueError:
                # When level doesn't exist, it means all levels have been loaded
                break
        
        # Store loaded summaries in instance variables for later use
        self.summaries_by_level = summaries_by_level
        self.total_levels = len(summaries_by_level)
        
        # Print loading information
        logger.info(f"Successfully loaded summaries for {self.total_levels} levels")
        for level, summaries in summaries_by_level.items():
            logger.info(f"Level {level}: {len(summaries)} summaries")
        
        return summaries_by_level 

    def try_load_or_generate_summaries(self, output_dir: str) -> bool:
        """
        Try to load existing summaries, if they don't exist or are invalid then generate new ones
        
        Args:
            output_dir: Output directory path
            
        Returns:
            bool: Whether successfully loaded or generated summaries
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Try to load existing summaries
            logger.info(f"Attempting to load timeline embeddings from {output_dir}...")
            if self.load_and_validate_summaries(output_dir):
                logger.info("Successfully loaded existing timeline embeddings")
                return True
                
            # If loading fails, generate new summaries
            logger.info("Loading validation failed, will generate new summaries")
            logger.info("Starting to generate new summaries...")
            return self._generate_and_save_summaries(output_dir)
            
        except Exception as e:
            logger.error(f"Error occurred while processing timeline summaries: {str(e)}")
            return False
    
    def _generate_and_save_summaries(self, output_dir: str):
        """
        Generate new summaries and save them
        
        Args:
            output_dir: Summary storage directory
        """
        logger.info("Starting to generate new summaries...")
        
        # Generate timeline summary
        timeline_result = self.generate_timeline_summary()
        
        # Save summary statistics
        self.save_summary_statistics(output_dir)
        
        # Ensure summaries for each level are correctly saved
        for level, summaries in enumerate(timeline_result["summaries_by_level"]):
            level_store = self.get_level_embedding_store(level, output_dir)
            level_store.insert_strings(summaries)
        
        logger.info(f"Summary generation completed, saved to {output_dir}")
        
        return timeline_result 
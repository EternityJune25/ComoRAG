import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from networkx import is_matching
from numpy import record

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: Optional[str] = field(
        default="https://api.example.com/v1",
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    llm_api_key: Optional[str] = field(
        default="your-llm-api-key-here",
        metadata={"help": "API key for the LLM model."}
    )
    embedding_base_url: Optional[str] = field(
        default="https://api.example.com/v1",
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    embedding_api_key: Optional[str] = field(
        default="your-embedding-api-key-here",
        metadata={"help": "API key for the embedding model."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    azure_endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Azure OpenAI endpoint. e.g. https://YOUR_RESOURCE_NAME.openai.azure.com/"}
    )
    azure_embedding_endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Azure OpenAI embedding endpoint. e.g. https://YOUR_RESOURCE_NAME.openai.azure.com/"}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )

    rerank_dspy_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    need_cluster: bool = field(
        default=False,
        metadata={"help": "If set to True, will perform clustering on the documents."}
    )
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: Optional[int] = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    
    
    # QA specific attributes
    max_meta_loop_max_iterations : int = field(
        default=5,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    max_iterations: int = field(
        default=3,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    max_qa_steps: int = field(
        default=3,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=50,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )

    qa_ver_top_k: int = field(
        default=50,
        metadata={"help": "Feeding top k veridicals to the QA model for reading."}
    )
    qa_sem_top_k: int = field(
        default=50,
        metadata={"help": "Feeding top k semantics to the QA model for reading."}
    )
    qa_epi_top_k: int = field(
        default=50,
        metadata={"help": "Feeding top k episodics to the QA model for reading."}
    )
    
    is_mc: bool = field(
        default=False,
        metadata={"help": "Whether the question is a multiple choice question."}
    )

    use_ver: bool = field(
        default=True,
        metadata={"help": "Whether to use the veridical index of the documents in the QA process."}
    )
    
    use_sem: bool = field(
        default= True,
        metadata={"help": "Whether to use the semantic index of the documents in the QA process."}
    )

    use_epi: bool = field(
        default=True,
        metadata={"help": "Whether to use the episodic index of the documents in the QA process."}
    )

    max_tokens_ver: int = field(
        default=3000,
        metadata={"help": "Max tokens for the chunk of the documents in the QA process."}
    )
    max_tokens_sem: int = field(
        default=1000,
        metadata={"help": "Max tokens for the summary of the documents in the QA process."}
    )
    max_tokens_epi: int = field(
        default=1000,
        metadata={"help": "Max tokens for the timeline of the documents in the QA process."}
    )
    # record the number of steps in the QA process
    record_steps: bool = field(
        default=False,
        metadata={"help": "Whether to record the number of steps in the QA process."}
    )
    # Save dir (highest level directory)
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    # Output directory for QA results
    output_dir: str = field(
        default="./outputs/qa_results",
        metadata={"help": "Directory to save QA results."}
    )
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")

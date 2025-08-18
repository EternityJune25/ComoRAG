<h1 align="center">ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning</h1>
<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![CUDA](https://img.shields.io/badge/CUDA-12.x-green)](https://developer.nvidia.com/cuda-zone) [![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)](https://kernel.org/) [![RAG](https://img.shields.io/badge/RAG-Graph%20Memory-orange)](#project-introduction) [![LLM](https://img.shields.io/badge/LLM-OpenAI%2FvLLM-purple)](#main-modules) [![Status](https://img.shields.io/badge/Status-Active-success)](#) [![arXiv](https://img.shields.io/badge/arXiv-2508.10419-b31b1b.svg)](https://arxiv.org/abs/2508.10419) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![DeepWiki](https://img.shields.io/badge/DeepWiki-ComoRAG-purple)](https://deepwiki.com/EternityJune25/ComoRAG)

[English](README.md) | [中文](README_zh.md)

</div>

<p align="center">
  <img src="assert/img/overview.png" alt="ComoRAG Overview" width="100%">
</p>

## 📖 Paper Information

This is the **official implementation** of the paper:

**[ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](https://arxiv.org/abs/2508.10419)**


**Citation:**
```bibtex
@article{wang2025comorag,
  title={ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning},
  author={Wang, Juyuan and Zhao, Rongchen and Wei, Wei and Wang, Yufeng and Yu, Mo and Zhou, Jie and Xu, Jin and Xu, Liyan},
  journal={arXiv preprint arXiv:2508.10419},
  year={2025}
}
```

---

## Project Introduction
ComoRAG is a Retrieval-Augmented Generation (RAG) system for long documents and multi-document QA, information extraction, and knowledge graph construction. It integrates various LLMs, embedding models, graph-based reasoning, and evaluation tools, suitable for both research and practical applications.

🔥 What makes ComoRAG different?

Narrative comprehension on long stories and novels is hard due to intricate plotlines and evolving character/entity relations. LLMs struggle with extended context and cost, so retrieval stays crucial. However, classic RAG is often stateless and single-step, missing the dynamic nature of long-range, interconnected reasoning.

ComoRAG takes a cognition-inspired approach: narrative reasoning is not one-shot, but a dynamic, evolving interplay between new evidence acquisition and consolidation of past knowledge — analogous to memory processes in the brain. 🧠

- 🔁 Iterative Reasoning Cycles: When hitting an impasse, ComoRAG launches cycles that interact with a dynamic memory workspace.
- 🕵️ Probing Queries: Each cycle generates targeted probes to explore new evidence paths.
- 🧳 Global Memory Pool: Newly retrieved evidence is integrated into a shared memory pool to progressively build coherent context for the query.

🚀 Benchmarks & Gains: On four challenging long-context narrative benchmarks (200K+ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to 11% over the strongest baseline. It particularly shines on complex queries requiring global comprehension, enabling principled, cognitively motivated, stateful retrieval-based reasoning. 📈

Key idea in one line: Reason → Probe → Retrieve → Consolidate → Resolve. 🧩

---

## Key Features ✨
- 🧠 Support for various LLMs and local/remote embedding models
- 🕸️ Graph-augmented retrieval and reasoning
- 🔧 Flexible data preprocessing and chunking
- 📊 Multiple evaluation metrics (F1, EM, etc.)
- 🧱 Modular and extensible design

---

## Directory Structure 📂
```
ComoRAG/
├── main_openai.py                       # Main program using OpenAI API
├── main_vllm.py                         # Main program using local vLLM server
├── script/                              # Data processing and evaluation scripts
│   ├── chunk_doc_corpus.py              # Document chunking script
│   └── eval_qa.py                       # QA evaluation script
├── dataset/                             # Dataset directory
│   └── ...
├── src/comorag/                        # Core code
│   ├── ComoRAG.py                       # Main class and core logic
│   ├── utils/                           # Utility modules
│   ├── embedding_model/                 # Embedding model related
│   ├── llm/                             # LLM related
│   ├── prompts/                         # Prompt templates
│   ├── information_extraction/          # Information extraction
│   └── rerank.py, embedding_store.py    # Other core modules
├── requirements.txt                     # Dependencies
└── README.md                            # Project documentation
```

---

## Installation & Environment 🛠️
1. 🐍 **Python version**: Python 3.10 or above recommended
2. 📦 **Install dependencies**:
```bash
pip install -r requirements.txt
```
3. 🔑 **Environment variables**: Set your OpenAI API Key or local LLM/embedding paths as needed
4. ⚙️ **GPU (optional but recommended)**: CUDA 12.x supported by many dependencies in requirements.txt

---

## Data Preparation & Format 📄
- 📚 **Corpus file corpus.jsonl**: Each line is a document, with fields like `id`, `doc_id`, `title`, `contents`
- ❓ **QA file qas.jsonl**: Each line is a question, with fields like `id`, `question`, `golden_answers`

Example:

corpus.jsonl:
```json
{"id": 0, "doc_id": 1, "title": "...", "contents": "..."}
```
qas.jsonl:
```json
{"id": "1", "question": "...", "golden_answers": ["..."]}
```

---

## Quick Start ⚡

### Method 1: Using OpenAI API (main_openai.py) 🚀

1. Configure dataset path and model parameters in the script:
```python
config = BaseConfig(
    llm_base_url='https://api.example.com/v1',  # OpenAI API
    llm_name='gpt-4o-mini',
    dataset='cinderella',
    embedding_model_name='/path/to/your/embedding/model',
    embedding_batch_size=32,
    need_cluster=True,  # Enable Semantic/Episodic enhancement
    output_dir='result/cinderella',
    save_dir='outputs/cinderella',
    max_meta_loop_max_iterations=5,
    is_mc=False,  # Multiple-choice?
    max_tokens_ver=2000,  # Veridical layer tokens
    max_tokens_sem=2000,  # Semantic layer tokens
    max_tokens_epi=2000   # Episodic layer tokens
)
```
2. Run the main program ▶️:
```bash
python main_openai.py
```

### Method 2: Using Local vLLM Server (main_vllm.py) ⚡

#### 1. Start vLLM Server 🚀

First, start the vLLM OpenAI-compatible API server:

```bash
# Method 1: Using vllm serve command
vllm serve /path/to/your/model \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95

# Method 2: Using python -m vllm.entrypoints.openai.api_server
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --served-model-name your-model-name \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --dtype auto
```

**Parameter descriptions:**
- `--model`: Model path (e.g., `/path/to/your/model`)
- `--tensor-parallel-size`: Number of GPU parallel processes
- `--max-model-len`: Maximum model length
- `--gpu-memory-utilization`: GPU memory utilization rate

#### 2. Configure main_vllm.py 📝

Modify the configuration in `main_vllm.py`:

```python
# vLLM server configuration
vllm_base_url = 'http://localhost:8000/v1'  # vLLM server address
served_model_name = '/path/to/your/model'    # Model path

config = BaseConfig(
    llm_base_url=vllm_base_url,
    llm_name=served_model_name,
    llm_api_key="your-api-key-here",  # Any value, local server doesn't need real API key
    dataset='cinderella',
    embedding_model_name='/path/to/your/embedding/model',
    embedding_batch_size=4,
    need_cluster=True,
    output_dir='result/cinderella_vllm',
    save_dir='outputs/cinderella_vllm',
    max_meta_loop_max_iterations=5,
    is_mc=False,
    max_tokens_ver=2000,
    max_tokens_sem=2000,
    max_tokens_epi=2000
)
```

#### 3. Run the Program ▶️

```bash
python main_vllm.py
```

#### 4. Check Server Status 🔍

Ensure the vLLM server is running properly:

```bash
# Check if port is occupied
netstat -tlnp | grep 8000

# Test API connection
curl http://localhost:8000/v1/models
```

### Comparison of Two Methods 📊

| Feature | OpenAI API (main.py) | vLLM Local (main_vllm.py) |
|---------|---------------------|---------------------------|
| Cost | Pay per token | One-time model download |
| Speed | Network latency | Local inference, faster |
| Privacy | Data sent to cloud | Completely local processing |
| Setup | Simple, just API key | Requires GPU and model files |
| Stability | Network dependent | Local control |

3. 📁 Results will be saved under `result/<dataset>/<subset>/`

---

## Main Modules
- 🏛️ `ComoRAG.py`: The main class, responsible for retrieval, graph construction, reasoning, and QA
- 🧰 `utils/`: Configuration, logging, embedding, clustering, summarization, memory, agents, and other utilities
- 🧲 `embedding_model/`: Embedding model adaptation and loading
- 🤖 `llm/`: LLM adaptation
- 🗒️ `prompts/`: Prompt template management
- 📦 `embedding_store.py`: Embedding vector storage and retrieval

---

## Data Processing & Evaluation Scripts 🧪
- ✂️ `script/chunk_doc_corpus.py`: Document chunking, supports token/word/sentence/recursive methods
- 📈 `script/eval_qa.py`: Automatic QA result evaluation, supports EM, F1, and other metrics

Example usage:

Chunking documents ✂️:
```bash
python script/chunk_doc_corpus.py \
  --input_path dataset/<name>/<subset>/corpus.jsonl \
  --output_path dataset/<name>/<subset>/corpus_chunked.jsonl \
  --chunk_by token \
  --chunk_size 512 \
  --tokenizer_name_or_path /path/to/your/tokenizer
```

Evaluate QA results 📊:
```bash
python script/eval_qa.py /path/to/result/<dataset>/<subset>
```
This produces files like ``details`、`results.json`, etc.

---

## Contact & Contribution 🤝
For questions or suggestions, feel free to submit an Issue or PR.

---

## Acknowledgement 🙏
We refer to the repository of [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) as a skeleton code.
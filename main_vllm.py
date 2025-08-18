import os
import json
import copy

from src.comorag.ComoRAG import ComoRAG
from src.comorag.utils.config_utils import BaseConfig
from src.comorag.utils.misc_utils import get_gold_answers


def process_dataset(dataset_path, config):
    dataset_name = os.path.basename(dataset_path)
    corpus_path = os.path.join(dataset_path, "corpus.jsonl")
    qas_path = os.path.join(dataset_path, "qas.jsonl")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    docs = [doc['contents'] for doc in corpus]

    with open(qas_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    all_queries = [s['question'] for s in samples]
    config.corpus_len = len(corpus)

    comorag = ComoRAG(global_config=config)
    comorag.index(docs)
    solutions = comorag.try_answer(all_queries)

    gold_answers = get_gold_answers(samples)
    for idx, q in enumerate(solutions):
        q.gold_answers = list(gold_answers[idx])

    result_list = []
    for idx, (q, solution) in enumerate(zip(all_queries, solutions)):
        result_list.append({
            "idx": idx,
            "question": q,
            "golden_answers": solution.gold_answers,
            "output": solution.answer
        })

    folder_path = os.path.join(config.output_dir)
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)


def main():
    # 1) Start a vLLM OpenAI-compatible server separately, e.g.:
    # vllm serve /path/to/your/model --tensor-parallel-size 1 --max_model_len 4096*2 --gpu-memory-utilization 0.95 

    # 2) Then run this script. We will call the server via base_url.
    # Ensure an API key is present for OpenAI-compatible clients
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # Optional: select visible GPUs for the client side
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


    base_path = './dataset/cinderella'
    dataset_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    dataset_dirs.sort()
    dataset_paths = [os.path.join(base_path, d) for d in dataset_dirs]

    
    vllm_base_url = 'http://localhost:8000/v1'
    served_model_name = '/path/to/your/model'

    config = BaseConfig(
        llm_base_url=vllm_base_url,
        llm_name=served_model_name,
        llm_api_key=os.environ["OPENAI_API_KEY"],
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

    for dataset_path in dataset_paths:
        tempconfig = copy.deepcopy(config)
        tempconfig.output_dir += f'/{os.path.basename(dataset_path)}'
        tempconfig.save_dir += f'/{os.path.basename(dataset_path)}'
        process_dataset(dataset_path, tempconfig)


if __name__ == "__main__":
    main()


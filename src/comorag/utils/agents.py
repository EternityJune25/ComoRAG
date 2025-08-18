# Merged from baseagent.py, poolagent.py, probesagent.py
import logging
import os
from abc import ABC, abstractmethod
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from ..prompts.prompt_template_manager import PromptTemplateManager
from .config_utils import BaseConfig
from typing import Optional, List, Dict
import re

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class BaseAgent(ABC):
    def __init__(self, model="gpt-4o-mini", llm_base_url="https://api.example.com/v1",llm_api_key=None):
        """
        Initialize base Agent class
        Args:
            model: Model name, defaults to "gpt-4o-mini"
            llm_base_url: Base URL for LLM API, defaults to "https://api.example.com/v1"
        """
        self.model = model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.client = OpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        self.prompt_template_manager = PromptTemplateManager()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _call_llm(self, messages, max_completion_tokens=500, stop_sequence=None, temperature=0, top_p=1):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_completion_tokens,
                stop=stop_sequence,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            return str(e)

    @abstractmethod
    def process(self, *args, **kwargs):
        pass

# PoolAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
class PoolAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini", llm_base_url="https://api.example.com/v1",llm_api_key=None):
        super().__init__(model, llm_base_url,llm_api_key)
        self.max_workers = 3

    def process(self, query, chunks=None, semantic_summaries=None, timeline_summaries=None, content=None, task='fusion', max_completion_tokens=500, stop_sequence=None):
        if task == 'memory_fusion':
            messages = self.prompt_template_manager.render(
                name='memory_fusion',
                query=query,
                content=content
            )
            return self._call_llm(
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                stop_sequence=stop_sequence
            )
        if task == 'node_fusion':
            messages = self.prompt_template_manager.render(
                name='node_fusion',
                query=query,
                content=content
            )
            return self._call_llm(
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                stop_sequence=stop_sequence
            )

    def memory_fusion(self, query, content):
        res = self.process(query, content=content, task='memory_fusion')
        return res

    def fusion(self, query, vers, sems, epis):
        results = {'chunk': None, 'summary': None, 'timeline': None}
        def process_content(content_type, content):
            try:
                if not content:
                    logging.warning(f"Empty content for {content_type}")
                    return content_type, ""
                result = self.memory_fusion(query=query, content=content)
                if result is None:
                    logging.warning(f"LLM returned None for {content_type}")
                    return content_type, ""
                return content_type, result
            except Exception as e:
                logging.error(f"Error processing {content_type}: {str(e)}")
                return content_type, ""
        tasks = [
            ('chunk', vers),
            ('summary', sems),
            ('timeline', epis)
        ]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_content = {
                executor.submit(process_content, content_type, content): content_type
                for content_type, content in tasks
            }
            for future in as_completed(future_to_content):
                content_type, result = future.result()
                results[content_type] = result
        return results['chunk'], results['summary'], results['timeline']

    def fuse_memory_nodes(self, query: str, content: str, max_completion_tokens: int = 1000) -> str:
        try:
            fused_content = self.process(
                query=query,
                content=content,
                task='node_fusion',
                max_completion_tokens=max_completion_tokens
            )
            logging.info(f"Successfully fused memory nodes, query clue: {query[:50]}...")
            return fused_content
        except Exception as e:
            logging.error(f"Error occurred while fusing memory nodes: {str(e)}")
            return f"Error during fusion. Original content:\n{content}"

# ProbesAgent
class ProbeAgent(BaseAgent):
    def __init__(self, model="gpt-4o-mini", llm_base_url="https://api.example.com/v1",llm_api_key=None):
        super().__init__(model, llm_base_url,llm_api_key)

    def process(self, query: str, context: str = None, previous_probes: str = None, max_completion_tokens: int = 500) -> str:
        messages = self.prompt_template_manager.render(
            name='agent_probe',
            query=query,
            context=context if context else "",
            previous_probes=previous_probes if previous_probes else "",
        )
        probes = self._call_llm(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=0
        )
        return probes.strip()

    def find_probes(self, query: str, context: str = None, previous_probes: str = None, max_completion_tokens: int = 500) -> List[str]:
        try:
            probes = self.process(query, context, previous_probes, max_completion_tokens)
            import json
            probe_dict = json.loads(probes)
            probes = [v for k, v in sorted(probe_dict.items()) if k.startswith("probe_")]
            return probes
        except Exception as e:
            try:
                import json
                if isinstance(e, json.JSONDecodeError):
                    return []
            except Exception:
                pass
            print(f"Error in parsing probes: {str(e)}")
            return []
import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT4SummarizationModel(BaseSummarizationModel):
    def __init__(self, model=None, llm_base_url="https://api.example.com/v1",llm_api_key="your-api-key-here"):
        """
        Initialize class with support for custom model and API base URL.
        
        :param model: Model name
        :param llm_base_url: Base URL for LLM API
        """
        self.model = model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        # Set up OpenAI client with custom base URL
        self.client = OpenAI(base_url=self.llm_base_url,api_key=self.llm_api_key,)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_completion_tokens=500, stop_sequence=None):
        """
        Generate text summary using GPT-4o-mini model

        :param context: Text that needs to be summarized
        :param max_tokens: Maximum number of summary tokens
        :param stop_sequence: Optional stop sequence
        :return: Generated summary
        """
        try:
            # Call OpenAI API interface to generate summary
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}",
                    },
                ],
                max_completion_tokens=max_completion_tokens,
                stop=stop_sequence,
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=1,
            )

            # Return generated summary
            return response.choices[0].message.content

        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)
        
"""
Client class for handling various LLM API providers:
    - oai models (also self-hosted vLLM servers)
    - claude models
"""

import os
import time
import tiktoken
import anthropic
from openai import OpenAI
from dataclasses import dataclass
import google.generativeai as genai
from typing import Dict, List, Optional, Union, Any
from google.generativeai.types import GenerationConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)

# Constants
API_KEY_MAP = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
}

TOKENIZER = "o200k_base"  # TODO:
MAX_TOKENS = 16384
TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
TOKEN_OVERHEAD = 3

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float
    top_p: float
    top_k: Optional[int] = None
    random_seed: Optional[int] = None
    stop: Optional[str] = None

class Client:
    """a unified client for interacting with various api"""

    def __init__(self, config: Any) -> None:
        """Initialize the client with the specified configuration.
        
        Args:
            config: Configuration object containing model settings and parameters.
        
        Raises:
            ValueError: If the configuration is invalid.
        """
        try:
            self.client_type = config.client_type
            self.max_length = config.max_seq_len
        except AttributeError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
        
        self.endpoint = "base" if self.client_type == "openai_base" else "chat"
        if self.client_type == "openai_base":
            self.client_type = "openai"

        self.api_key = os.environ.get(API_KEY_MAP[self.client_type])
        self.model_name = config.model_name_or_path
        self.think = config.think
        self.api_url = config.api_url
        self.api_port = config.api_port
        self.generation_kwargs = vars(config.generation_kwargs)
        self.encoding = tiktoken.get_encoding(TOKENIZER)
        self.timeout = config.timeout
        
        self._create_client()

    @retry(
        wait=wait_random_exponential(min=60, max=120),
        stop=stop_after_attempt(5)
    )
    def _create_client(self) -> None:
        """Create the appropriate client based on the provider type."""
        match self.client_type:
            case "openai":
                if self.api_key is not None:
                    self.client = OpenAI(
                        api_key=self.api_key,
                        timeout=self.timeout,
                    )
                else:
                    time.sleep(10)  # Wait for vLLM server to be ready
                    self.client = OpenAI(
                        api_key="EMPTY",
                        base_url=f"http://{self.api_url}:{self.api_port}/v1",
                        timeout=self.timeout,
                    )
                    models = self.client.models.list()
                    self.model_name = models.data[0].id
            
            case "anthropic":
                self.client = anthropic.Anthropic(api_key=self.api_key)
            
            case _:
                raise ValueError(f"Unsupported client type: {self.client_type}")

    def __call__(self, prompt: str, tokens_to_generate: int) -> Dict[str, List[str]]:
        """Generate text using the configured model.
        
        Args:
            prompt: The input prompt.
            tokens_to_generate: Maximum number of tokens to generate.
            
        Returns:
            Dict containing the generated text.
        """
        match self.client_type:
            case "openai":
                return self._call_openai(prompt, tokens_to_generate)
            case "anthropic":
                return self._call_anthropic(prompt, tokens_to_generate)
            case _:
                raise ValueError(f"Unsupported client type: {self.client_type}")

    @retry(
        wait=wait_random_exponential(min=15, max=60),
        stop=stop_after_attempt(3),
        retry_error_callback=lambda retry_state: print(f"Retrying... Attempt {retry_state.attempt_number}")
    )
    def _send_request_openai(self, request: Dict[str, Any]) -> Any:
        """Send a request to the OpenAI API.
        
        Args:
            request: Dictionary containing request parameters.
            
        Returns:
            API response.
        """
        kwargs = {
            "model": self.model_name,
            "messages": request["msgs"],
            "temperature": request["temperature"],
            "seed": request["random_seed"],
            "top_p": request["top_p"],
            "stop": request["stop"],
        }
        
        if "o1" in self.model_name:
            kwargs["max_completion_tokens"] = request["tokens_to_generate"]
        else:
            kwargs["max_tokens"] = request["tokens_to_generate"]
        
        if self.endpoint == "chat":
            return self.client.chat.completions.create(**kwargs)
        elif self.endpoint == "base":
            kwargs["prompt"] = "".join([m["content"] for m in kwargs.pop("messages")])
            return self.client.completions.create(**kwargs)
        else:
            raise NotImplementedError(f"Unsupported endpoint: {self.endpoint}")

    def _call_openai(self, prompt: str, tokens_to_generate: int) -> Dict[str, List[str]]:
        """Generate text using OpenAI models.
        
        Args:
            prompt: The input prompt.
            tokens_to_generate: Maximum number of tokens to generate.
            
        Returns:
            Dict containing the generated text.
        """
        msgs = [{"role": "user", "content": prompt}]
        
        # Calculate token count
        msgs_len = sum(
            TOKENS_PER_MESSAGE + sum(len(self.encoding.encode(value)) for value in message.values())
            + (TOKENS_PER_NAME if "name" in message else 0)
            for message in msgs
        ) + TOKEN_OVERHEAD

        request = self.generation_kwargs.copy()
        request["tokens_to_generate"] = min(tokens_to_generate, MAX_TOKENS)
        request["msgs"] = msgs
        
        try:
            outputs = self._send_request_openai(request)
            if outputs is None:
                response_text = ""
            else:
                response_text = (
                    outputs.choices[0].message.content
                    if self.endpoint == "chat"
                    else outputs.choices[0].text
                )
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            response_text = ""
            
        return {"text": [response_text]}

    @retry(
        wait=wait_random_exponential(min=15, max=60),
        stop=stop_after_attempt(3)
    )
    def _send_request_anthropic(self, request: Dict[str, Any]) -> Any:
        """Send a request to the Anthropic API.
        
        Args:
            request: Dictionary containing request parameters.
            
        Returns:
            API response.
        """
        kwargs = {
            "model": self.model_name,
            "messages": request["msgs"],
            "temperature": request["temperature"],
            "top_p": request["top_p"],
            "top_k": request["top_k"],
            "max_tokens": request["tokens_to_generate"]
        }
        return self.client.messages.create(**kwargs)

    def _call_anthropic(self, prompt: str, tokens_to_generate: int) -> Dict[str, List[str]]:
        """Generate text using Anthropic models.
        
        Args:
            prompt: The input prompt.
            tokens_to_generate: Maximum number of tokens to generate.
            
        Returns:
            Dict containing the generated text.
        """
        msgs = [{"role": "user", "content": prompt}]
        
        # Estimate token count
        alpha = 1.3  # TODO: 
        if len(prompt) / alpha >= self.max_length * 0.9:
            msgs_len = self.client.count_tokens(
                model=self.model_name,
                messages=msgs
            )["input_tokens"]
        else:
            msgs_len = len(prompt) / alpha

        request = self.generation_kwargs.copy()
        tokens_to_generate_new = self.max_length - msgs_len
        
        if tokens_to_generate_new < tokens_to_generate:
            print(f"Reducing generate tokens from {tokens_to_generate} to {tokens_to_generate_new}")
            request["tokens_to_generate"] = tokens_to_generate_new
        else:
            request["tokens_to_generate"] = tokens_to_generate

        request["msgs"] = msgs
        
        try:
            outputs = self._send_request_anthropic(request)
            response_text = outputs.content[0].text
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            response_text = ""
            
        return {"text": [response_text]}
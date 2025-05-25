\
import os
from openai import OpenAI, APIError
from typing import List, Dict, Any, Iterator, Optional


class ModelScopeClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = 'https://api-inference.modelscope.cn/v1/'):
        if api_key is None:
            api_key = os.getenv("MODELSCOPE_API_KEY") # Allow fallback to environment variable
        if not api_key:
            raise ValueError("ModelScope API key must be provided either as an argument or via MODELSCOPE_API_KEY environment variable.")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Iterator[Dict[str, Any]] | Dict[str, Any]:
        request_params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        if extra_body:
            # The openai library passes extra_body directly at the top level of the request JSON.
            # It might not be nested under `extra_body` itself in the final JSON sent to the server,
            # but rather its contents are merged. However, the `create` method handles this.
            request_params["extra_body"] = extra_body

        try:
            response = self.client.chat.completions.create(**request_params)
            
            if stream:
                # For streaming, the response object is already an iterator of ChatCompletionChunk
                return response 
            else:
                # For non-streaming, convert to dict for consistent return type with typical non-streaming OpenAI client usage
                return response.model_dump() # Converts the Pydantic model to a dict
        except APIError as e:
            print(f"ModelScope API error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred in ModelScopeClient: {e}")
            raise
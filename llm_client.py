from types import Dict, Any, List, Iterator
import json
from fastapi import HTTPException
import requests



class NebuClient:
    def __init__(self, api_key: str, base_url: str = "https://inference.nebulablock.com/v1"):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _make_request(self, endpoint: str, payload: Dict[str, Any], stream: bool = False) -> requests.Response:
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, json=payload, stream=stream)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            # Consider how to propagate this error. For now, re-raising or returning None/error marker.
            # For a FastAPI app, it might be better to raise an HTTPException here or let the caller handle it.
            raise HTTPException(status_code=503, detail=f"Error communicating with upstream LLM service: {str(e)}")


    def chat_completions(self, messages: List[Dict[str, str]], model_name: str, stream: bool = False, **kwargs) -> Iterator[Dict[str, Any]] | Dict[str, Any]:
        payload = {
            "messages": messages,
            "model": model_name, # This is the model name expected by the upstream API
            "stream": stream,
            **kwargs
        }
        
        response = self._make_request("chat/completions", payload, stream=stream)

        if not stream:
            return response.json()
        else:
            def stream_generator():
                try:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data: '):
                                data_content = decoded_line[len('data: '):].strip()
                                if data_content == '[DONE]':
                                    break
                                if data_content: # Ensure it's not an empty data line
                                    try:
                                        yield json.loads(data_content)
                                    except json.JSONDecodeError:
                                        print(f"Skipping non-JSON line from upstream: {data_content}")
                                        continue
                finally:
                    response.close() # Ensure the response is closed
            return stream_generator()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Iterator, Dict, Any
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import Request
# import requests # No longer directly used in this file after client abstraction
# from llm_client import LLMClient # Replaced with ModelScopeClient
from modelscope_client import ModelScopeClient # Import the new client
from dotenv import load_dotenv
import os # Add this import
import datetime # Add this import
import asyncio # Add this import
import json # Ensure json is imported

# 加载.env文件中的变量到系统环境变量中
load_dotenv()
app = FastAPI()

@app.get("/api/tags")
async def get_tags():

    return JSONResponse(
        content={
            "models":[{"name":"deepseek","model":"deepseek","modified_at":"2025-05-23T22:51:25.989913572+08:00","size":394998579,"digest":"b5dc5e784f2a3ee1582373093acf69a2f4e2ac1710b253a001712b86a61f88bb", "details":{"parent_model":"","format":"gguf","family":"qwen2","families":["qwen2"],"parameter_size":"620M","quantization_level":"Q4_0"}}]
        }
    )

@app.post("/api/pull")
async def pull_model():
    async def generate_events():
        events = [
            '{"status":"pulling manifest deepseek"}\n',
            '{"status":"verifying sha256 digest"}\n',
            '{"status":"pulling manifest deepseek/tokenizer"}\n',
            '{"status":"verifying sha256 digest for tokenizer"}\n',
            '{"status":"downloading 10% 40 MB"}\n',
            '{"status":"downloading 30% 120 MB"}\n',
            '{"status":"downloading 60% 240 MB"}\n',
            '{"status":"downloading 90% 355 MB"}\n',
            '{"status":"downloading 100% 395 MB"}\n',
            '{"status":"extracting files"}\n',
            '{"status":"verifying files"}\n',
            '{"status":"success"}\n'
        ]
        for event in events:
            yield event
            await asyncio.sleep(2)  # 模拟延迟

    return StreamingResponse(generate_events(), media_type="application/x-ndjson")

@app.post("/api/show")
async def show_model(request: Request):
    data = await request.json()
    print("Received data:", data)
    model = data.get("model", "")
    result = json.load(open("show.json"))
    if model == "deepseek":
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/api/chat")
async def chat_model(request: Request):
    data = await request.json()
    print("Received data for /api/chat:", data) 
    
    request_model_name = data.get("model", "") 
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    ollama_tools = data.get("tools", []) # Renamed from 'tools' to avoid confusion
    ollama_options = data.get("options", {}) # Get options dictionary

    # Preprocess messages to ensure tool_calls.function.arguments is a string
    for message in messages:
        if "tool_calls" in message and isinstance(message["tool_calls"], list):
            for tool_call in message["tool_calls"]:
                if isinstance(tool_call, dict) and "function" in tool_call and isinstance(tool_call["function"], dict):
                    if "arguments" in tool_call["function"] and isinstance(tool_call["function"]["arguments"], dict):
                        tool_call["function"]["arguments"] = json.dumps(tool_call["function"]["arguments"])
                    # Also ensure arguments is present, even if empty, as a string
                    elif "arguments" not in tool_call["function"]:
                         tool_call["function"]["arguments"] = "{}" # Or handle as per API expectation for empty args

    # Parameters to pass directly to the OpenAI client, mapped from Ollama's options
    # These will be passed as keyword arguments to the ModelScopeClient
    client_kwargs = {}
    if ollama_options:
        if "temperature" in ollama_options and ollama_options["temperature"] is not None:
            client_kwargs["temperature"] = ollama_options["temperature"]
        if "top_p" in ollama_options and ollama_options["top_p"] is not None:
            client_kwargs["top_p"] = ollama_options["top_p"]
        if "seed" in ollama_options and ollama_options["seed"] is not None: # OpenAI client supports 'seed'
            client_kwargs["seed"] = ollama_options["seed"]
        if "num_predict" in ollama_options and ollama_options["num_predict"] is not None: # Ollama's num_predict
            client_kwargs["max_tokens"] = ollama_options["num_predict"] # OpenAI's max_tokens
        if "stop" in ollama_options and ollama_options["stop"] is not None: # Ollama's stop
            client_kwargs["stop"] = ollama_options["stop"] # OpenAI's stop (can be str or list of str)
        # Add other known OpenAI parameters if they exist in ollama_options and are supported by the client
        # e.g., presence_penalty, frequency_penalty

    # Convert Ollama tools to OpenAI format if present
    openai_tools_list = []
    if ollama_tools:
        stream = False # Ollama tool use implies non-streaming for the request/response cycle with tool calls
        for tool_config in ollama_tools:
            if tool_config.get("type") == "function" and "function" in tool_config:
                # Assume it's already in OpenAI tool format if type is "function"
                openai_tools_list.append(tool_config)
            elif tool_config.get("type") == "local_tool" and "function" in tool_config: # Handle the custom type
                # Convert local_tool to OpenAI function format
                openai_tools_list.append({
                    "type": "function",
                    "function": tool_config["function"]
                })
            # else: Consider logging a warning for unhandled tool types

    if openai_tools_list:
        client_kwargs["tools"] = openai_tools_list
        # client_kwargs["tool_choice"] = "auto" # Let OpenAI client default this, usually "auto" if tools are present

    if request_model_name == "deepseek" or request_model_name.startswith("qwen"):
        api_key = os.getenv("MODELSCOPE_API_KEY")
        if not api_key:
            print("CRITICAL ERROR: MODELSCOPE_API_KEY environment variable is not set.")
            raise HTTPException(
                status_code=500, 
                detail="Proxy server error: ModelScope API key is not configured."
            )
        
        client = ModelScopeClient(api_key=api_key)
        
        # Define the upstream ModelScope model ID.
        # This could be mapped from request_model_name if you support multiple ModelScope models.
        upstream_model_id = 'Qwen/Qwen3-235B-A22B' # Default ModelScope model for "deepseek"
        if request_model_name.startswith("qwen"): # Example: if Raycast asks for "qwen-7b", map it
            # Potentially map request_model_name to a specific ModelScope model ID here
            # For now, we'll use the default Qwen model above for any qwen-prefixed request.
            pass

        extra_body_params = {
            "enable_thinking": False,
            # "thinking_budget": 4096 # Optional
        }
        

        # Tools are now part of client_kwargs if present
        # The old tool_params is not needed.

        if not stream:
            try:
                api_response_data = client.chat_completions(
                    model=upstream_model_id,
                    messages=messages, # Use processed_messages
                    stream=False,
                    extra_body=extra_body_params,
                    **client_kwargs 
                )
                # api_response_data is a dict from model_dump()
                content = ""
                tool_calls = None
                
                if api_response_data.get("choices"):
                    choice = api_response_data["choices"][0]
                    if choice.get("message"):
                        message = choice["message"]
                        if message.get("content"):
                            content = message["content"]
                        if message.get("tool_calls"):
                            tool_calls = message["tool_calls"]
                
                usage = api_response_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")

                response_message = {
                    "role": "assistant",
                    "content": content
                }
                
                # Add tool_calls to response if present
                if tool_calls:
                    response_message["tool_calls"] = tool_calls

                return JSONResponse(content={
                    "model": request_model_name,
                    "created_at": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    "message": response_message,
                    "done": True,
                    "total_duration": 5191566416, # Mocked or could be calculated
                    "load_duration": 2154458,    # Mocked
                    "prompt_eval_count": prompt_tokens,
                    "prompt_eval_duration": 383809000, # Mocked
                    "eval_count": completion_tokens,
                    "eval_duration": 4799921000 # Mocked
                })
            except HTTPException as e: 
                raise e 
            except Exception as e:
                print(f"Error processing non-streaming ModelScope chat: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing chat with ModelScope: {str(e)}")
        else:
            # Streaming for ModelScope
            def ollama_modelscope_stream_generator():
                response_model_name_for_ollama = data.get("model", "deepseek") 
                stream_created_at = datetime.datetime.now(datetime.UTC).isoformat() + "Z"
                
                # full_content_for_final_message = [] # To accumulate content if needed for final stats (not used here)

                try:
                    # The client.chat_completions itself returns a generator for streaming
                    for chunk in client.chat_completions(
                                            model=upstream_model_id,
                                            messages=messages, # Pass the preprocessed messages
                                            stream=True, 
                                            extra_body=extra_body_params,
                                            **client_kwargs): # Pass mapped options and tools here
                        
                        # chunk is an OpenAI ChatCompletionChunk object
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            
                            # Process reasoning_content if present (ModelScope specific)
                            reasoning_text = None
                            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                                reasoning_text = delta.reasoning_content
                                ollama_chunk_reasoning = {
                                    "model": response_model_name_for_ollama,
                                    "created_at": stream_created_at,
                                    "message": {"role": "assistant", "content": reasoning_text},
                                    "done": False
                                }
                                yield f"{json.dumps(ollama_chunk_reasoning)}\n"
                                # full_content_for_final_message.append(reasoning_text)

                            # Process standard content
                            if delta.content:
                                ollama_chunk_content = {
                                    "model": response_model_name_for_ollama,
                                    "created_at": stream_created_at,
                                    "message": {"role": "assistant", "content": delta.content},
                                    "done": False
                                }
                                yield f"{json.dumps(ollama_chunk_content)}\n"
                                # full_content_for_final_message.append(delta.content)
                            
                            # Handle tool calls in streaming
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                for tool_call_chunk in delta.tool_calls: # Iterate through tool_call_chunks
                                    # Reconstruct the tool_call object for Ollama format
                                    # ModelScope might stream tool calls piece by piece.
                                    # We need to aggregate them or handle them according to how Ollama expects them.
                                    # For simplicity, let's assume a full tool_call structure is in the chunk if present.
                                    # This part might need refinement based on actual ModelScope streaming behavior for tools.
                                    
                                    # Ensure arguments are stringified if they are dicts,
                                    # though for responses, they usually are strings already.
                                    # However, the error was for *input* messages.
                                    # For output streaming, we need to format it for Ollama.
                                    
                                    ollama_tool_call_chunk = {
                                        "model": response_model_name_for_ollama,
                                        "created_at": stream_created_at,
                                        "message": {
                                            "role": "assistant",
                                            "content": None, # Typically null when tool_calls are present
                                            "tool_calls": [ # Ollama expects a list of tool calls
                                                {
                                                    "index": tool_call_chunk.index, # Preserve index
                                                    "id": tool_call_chunk.id,
                                                    "type": "function", # Assuming 'function'
                                                    "function": {
                                                        "name": tool_call_chunk.function.name,
                                                        "arguments": tool_call_chunk.function.arguments or "" # Ensure arguments is a string
                                                    }
                                                }
                                            ]
                                        },
                                        "done": False
                                    }
                                    yield f"{json.dumps(ollama_tool_call_chunk)}\n"
                    
                    # Send final done message (Ollama format)
                    final_ollama_chunk = {
                        "model": response_model_name_for_ollama,
                        "created_at": stream_created_at,
                        # "message": { "role": "assistant", "content": ""}, # Often empty or omitted in final Ollama stream chunk
                        "done": True,
                        "done_reason": "stop",
                        "total_duration": 5191566416, # Mocked
                        "load_duration": 2154458,    # Mocked
                        "prompt_eval_count": None, # Not available in stream typically
                        "prompt_eval_duration": 383809000, # Mocked
                        "eval_count": None, # Not available in stream typically
                        "eval_duration": 4799921000  # Mocked
                    }
                    yield f"{json.dumps(final_ollama_chunk)}\n"
                
                except HTTPException as e: 
                    print(f"HTTP Error during ModelScope streaming: {e.detail}")
                    # Yield an Ollama-formatted error chunk
                    error_chunk = {
                        "model": response_model_name_for_ollama,
                        "created_at": stream_created_at,
                        "error": f"ModelScope streaming error: {e.detail}",
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk)}\n"
                except Exception as e:
                    print(f"Error during ModelScope streaming: {e}")
                    # Yield an Ollama-formatted error chunk
                    error_chunk = {
                        "model": response_model_name_for_ollama,
                        "created_at": stream_created_at,
                        "error": f"Unexpected error during ModelScope streaming: {str(e)}",
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk)}\n"

            return StreamingResponse(ollama_modelscope_stream_generator(), media_type="application/x-ndjson")
    else: 
        raise HTTPException(status_code=404, detail=f"Model '{request_model_name}' not found or not supported by this proxy.")

if __name__ == "__main__":
    import uvicorn # Keep uvicorn import here for running the app directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
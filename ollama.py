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

def process_mcp_tool_calls(tool_calls):
    """Process tool calls to ensure MCP compliance with required 'thought' field"""
    if not tool_calls:
        return tool_calls
    
    processed_tool_calls = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            # Ensure MCP required fields are present
            processed_call = tool_call.copy()
            
            # Ensure other MCP required fields
            if "id" not in processed_call:
                processed_call["id"] = f"call_{len(processed_tool_calls)}"
            
            if "type" not in processed_call:
                processed_call["type"] = "function"
            
            # Process function details
            if "function" in processed_call:
                func_details = processed_call["function"]
                if isinstance(func_details, dict):
                    # Handle arguments - ensure it's properly formatted
                    if "arguments" in func_details:
                        args = func_details["arguments"]
                        
                        # Convert dict to JSON string for OpenAI API if needed
                        if isinstance(args, dict):
                            # For MCP tools that require 'thought' parameter, add it if missing and arguments are empty
                            if not args and func_details.get("name", "").endswith("sequentialthinking"):
                                args["thought"] = "Starting analysis"
                                args["nextThoughtNeeded"] = True
                                args["thoughtNumber"] = 1
                                args["totalThoughts"] = 3
                            
                            func_details["arguments"] = json.dumps(args)
                        elif not isinstance(args, str):
                            func_details["arguments"] = str(args)
            
            processed_tool_calls.append(processed_call)
    
    return processed_tool_calls

def enhance_tool_calls_for_streaming(tool_calls):
    """Enhanced processing for streaming tool calls with better MCP compliance"""
    if not tool_calls:
        return tool_calls
    
    enhanced_calls = []
    for i, tool_call in enumerate(tool_calls):
        if isinstance(tool_call, dict):
            enhanced_call = tool_call.copy()
            
            # Ensure required fields
            if "id" not in enhanced_call:
                enhanced_call["id"] = f"call_{i}"
            
            if "type" not in enhanced_call:
                enhanced_call["type"] = "function"
            
            # Process function details
            if "function" in enhanced_call:
                func_details = enhanced_call["function"]
                if isinstance(func_details, dict):
                    # Handle arguments conversion and validation
                    if "arguments" in func_details:
                        args = func_details["arguments"]
                        
                        # Parse JSON string back to dict for Ollama format
                        if isinstance(args, str):
                            try:
                                parsed_args = json.loads(args) if args else {}
                            except json.JSONDecodeError:
                                parsed_args = {}
                        else:
                            parsed_args = args if isinstance(args, dict) else {}
                        
                        # For sequential thinking tool, ensure required parameters
                        if func_details.get("name", "").endswith("sequentialthinking"):
                            if not parsed_args or not parsed_args.get("thought"):
                                parsed_args.update({
                                    "thought": "Starting analysis",
                                    "nextThoughtNeeded": True,
                                    "thoughtNumber": 1,
                                    "totalThoughts": 3
                                })
                        
                        # For search tools, ensure query parameter
                        elif "search" in func_details.get("name", "").lower():
                            if not parsed_args.get("query"):
                                # Try to extract from recent user message or use default
                                parsed_args["query"] = "search query"
                        
                        func_details["arguments"] = parsed_args
            
            enhanced_calls.append(enhanced_call)
    
    return enhanced_calls

@app.post("/api/chat")
async def chat_model(request: Request):
    data = await request.json()
    print("Received data for /api/chat:", data) 
    
    request_model_name = data.get("model", "") 
    messages: List[Dict[str, Any]] = data.get("messages", []) # Ensure type hint
    stream = data.get("stream", False)
    ollama_tools = data.get("tools", []) # Renamed from 'tools' to avoid confusion
    ollama_options = data.get("options", {}) # Get options dictionary

    # Enhanced message preprocessing with better tool call handling
    for msg_idx, msg_content in enumerate(messages):
        if msg_content.get("role") == "assistant" and "tool_calls" in msg_content:
            if isinstance(msg_content.get("tool_calls"), list):
                # Process tool calls for MCP compliance
                msg_content["tool_calls"] = process_mcp_tool_calls(msg_content["tool_calls"])

        # Handle tool response messages
        elif msg_content.get("role") == "tool":
            # Tool response messages should be fine as is, but ensure content is a string
            if "content" in msg_content and not isinstance(msg_content["content"], str):
                msg_content["content"] = json.dumps(msg_content["content"]) if isinstance(msg_content["content"], dict) else str(msg_content["content"])
            
            # Ensure MCP required fields for tool messages
            if "tool_call_id" not in msg_content and "call_id" in msg_content:
                msg_content["tool_call_id"] = msg_content["call_id"]

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

    if openai_tools_list:
        client_kwargs["tools"] = openai_tools_list

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
        upstream_model_id = 'Qwen/Qwen3-235B-A22B' # Default ModelScope model for "deepseek"
        if request_model_name.startswith("qwen"): # Example: if Raycast asks for "qwen-7b", map it
            pass

        extra_body_params = {
            "enable_thinking": False,
        }

        if not stream:
            try:
                api_response_data = client.chat_completions(
                    model=upstream_model_id,
                    messages=messages,
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
                            # Enhanced tool call processing for better MCP compliance
                            tool_calls = enhance_tool_calls_for_streaming(tool_calls)
                
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

                try:
                    # The client.chat_completions itself returns a generator for streaming
                    for chunk in client.chat_completions(
                                            model=upstream_model_id,
                                            messages=messages,
                                            stream=True, 
                                            extra_body=extra_body_params,
                                            **client_kwargs):
                        
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

                            # Process standard content
                            if delta.content:
                                ollama_chunk_content = {
                                    "model": response_model_name_for_ollama,
                                    "created_at": stream_created_at,
                                    "message": {"role": "assistant", "content": delta.content},
                                    "done": False
                                }
                                yield f"{json.dumps(ollama_chunk_content)}\n"
                            
                            # Handle tool calls in streaming with enhanced MCP compliance
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                enhanced_tool_calls = []
                                
                                for i, tool_call_chunk in enumerate(delta.tool_calls):
                                    # Process arguments
                                    arguments = tool_call_chunk.function.arguments or ""
                                    try:
                                        parsed_args = json.loads(arguments) if arguments else {}
                                    except json.JSONDecodeError:
                                        parsed_args = {}
                                    
                                    # Enhanced argument handling for specific tools
                                    func_name = tool_call_chunk.function.name
                                    if func_name and func_name.endswith("sequentialthinking"):
                                        if not parsed_args or not parsed_args.get("thought"):
                                            parsed_args.update({
                                                "thought": "Starting analysis",
                                                "nextThoughtNeeded": True,
                                                "thoughtNumber": 1,
                                                "totalThoughts": 3
                                            })
                                    elif "search" in func_name.lower() and not parsed_args.get("query"):
                                        # Extract search query from conversation context
                                        user_messages = [msg for msg in messages if msg.get("role") == "user"]
                                        if user_messages:
                                            last_user_msg = user_messages[-1].get("content", "")
                                            # Simple extraction - look for text after @ mentions
                                            if "@" in last_user_msg:
                                                query_part = last_user_msg.split("@")[-1].strip()
                                                if query_part:
                                                    parsed_args["query"] = query_part
                                    
                                    # Create enhanced tool call
                                    enhanced_tool_call = {
                                        "index": tool_call_chunk.index if hasattr(tool_call_chunk, 'index') else i,
                                        "id": tool_call_chunk.id or f"call_{i}",
                                        "type": "function",
                                        "function": {
                                            "name": func_name,
                                            "arguments": parsed_args
                                        }
                                    }
                                    
                                    enhanced_tool_calls.append(enhanced_tool_call)
                                
                                if enhanced_tool_calls:
                                    ollama_tool_call_chunk = {
                                        "model": response_model_name_for_ollama,
                                        "created_at": stream_created_at,
                                        "message": {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": enhanced_tool_calls
                                        },
                                        "done": False
                                    }
                                    yield f"{json.dumps(ollama_tool_call_chunk)}\n"
                    
                    # Send final done message (Ollama format)
                    final_ollama_chunk = {
                        "model": response_model_name_for_ollama,
                        "created_at": stream_created_at,
                        "message": { "role": "assistant", "content": ""},
                        "done": True,
                        "done_reason": "stop",
                        "total_duration": 5191566416, # Mocked
                        "load_duration": 2154458,    # Mocked
                        "prompt_eval_count": None,
                        "prompt_eval_duration": 383809000, # Mocked
                        "eval_count": None,
                        "eval_duration": 4799921000  # Mocked
                    }
                    yield f"{json.dumps(final_ollama_chunk)}\n"
                
                except HTTPException as e: 
                    print(f"HTTP Error during ModelScope streaming: {e.detail}")
                    error_chunk = {
                        "model": response_model_name_for_ollama,
                        "created_at": stream_created_at,
                        "error": f"ModelScope streaming error: {e.detail}",
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk)}\n"
                except Exception as e:
                    print(f"Error during ModelScope streaming: {e}")
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
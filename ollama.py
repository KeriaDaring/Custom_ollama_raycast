from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import Request
import asyncio
import json
import requests
import os
import datetime
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
    print("Received data:", data)
    model = data.get("model", "")
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", None)
    temperature = data.get("temperature", 1)
    top_p = data.get("top_p", 0.9)
    stream = data.get("stream", False)
    # stream = False

    if model == "deepseek":
        api_key = os.getenv("NEBULA_API_KEY")
        if not api_key:
            print("CRITICAL ERROR: NEBULA_API_KEY environment variable is not set.")
            raise HTTPException(
                status_code=500, 
                detail="Proxy server error: Upstream API key is not configured."
            )

        url = "https://inference.nebulablock.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer " + api_key  # Use API key from environment
        }
        payload = {
            "messages": messages,
            "model": "deepseek-ai/DeepSeek-V3-0324",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        if not stream:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return JSONResponse(content={
                "model": "deepseek",
                "created_at": "2023-12-12T14:13:43.416799Z",
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done": True,
                "total_duration": 5191566416,
                "load_duration": 2154458,
                "prompt_eval_count": 26,
                "prompt_eval_duration": 383809000,
                "eval_count": 298,
                "eval_duration": 4799921000
            })
        else:
            def stream_response():
                # Use the model name from the original request for the response
                # The outer 'data' variable holds the initial request JSON to this /api/chat endpoint
                response_model_name = data.get("model", "deepseek") 
                
                # Generate created_at once for all stream messages
                stream_created_at = datetime.datetime.now(datetime.UTC).isoformat() + "Z"

                try:
                    with requests.post(url, headers=headers, json=payload, stream=True) as resp:
                        resp.raise_for_status()
                        
                        for line in resp.iter_lines():
                            if line:
                                print(line, flush=True)
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith('data: '):
                                    data_content = decoded_line[6:]  # Remove 'data: ' prefix
                                    if data_content.strip() == '[DONE]':
                                        break
                                    try:
                                        chunk_data = json.loads(data_content)
                                        choices = chunk_data.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                ollama_chunk = {
                                                    "model": response_model_name,
                                                    "created_at": stream_created_at,
                                                    "message": {
                                                        "role": "assistant",
                                                        "content": content
                                                    },
                                                    "done": False
                                                }
                                                yield f"{json.dumps(ollama_chunk)}\n"
                                    except json.JSONDecodeError:
                                        continue
                        
                        # Send final done message
                        final_chunk = {
                            "model": response_model_name,
                            "created_at": stream_created_at,
                            "message": {
                                "role": "assistant",
                                "content": ""
                            },
                            "done": True,
                            "total_duration": 5191566416,
                            "load_duration": 2154458,
                            "prompt_eval_count": 26,
                            "prompt_eval_duration": 383809000,
                            "eval_count": 298,
                            "eval_duration": 4799921000
                        }
                        yield f"{json.dumps(final_chunk)}\n"
                        
                except Exception as e:
                    print(f"Error in streaming: {e}")
                    error_chunk = {
                        "model": response_model_name,
                        "created_at": stream_created_at,
                        "message": {
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        },
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk)}\n"

            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
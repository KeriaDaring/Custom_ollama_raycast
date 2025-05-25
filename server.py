# filepath: /Users/keria/Documents/develop/fake_ollama/server.py
import asyncio
import datetime
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Ollama API", version="0.1.0")

# --- Mock Data ---
MOCK_MODEL_NAME = "mock-llama3:latest"
MOCK_MODEL_DIGEST = "sha256:mockdigest1234567890abcdef"
MOCK_MODEL_SIZE = 1234567890  # Bytes

# Store loaded models and their keep_alive settings
LOADED_MODELS: Dict[str, float] = {} # model_name -> expiration_time (timestamp)

# --- Pydantic Models (based on api.md) ---

class ModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "gguf"
    family: str = "llama"
    families: Optional[List[str]] = ["llama"]
    parameter_size: str = "8B"
    quantization_level: str = "Q4_0"

class ModelTagInfo(BaseModel):
    name: str
    model: str 
    modified_at: str
    size: int
    digest: str
    details: ModelDetails

class TagsResponse(BaseModel):
    models: List[ModelTagInfo]

class ShowRequest(BaseModel):
    name: str = Field(..., description="Name of the model to show")

class ShowResponse(BaseModel):
    license: Optional[str] = "Mock License"
    modelfile: Optional[str] = "# Mock Modelfile\\nFROM /path/to/weights\\nPARAMETER temperature 0.7"
    parameters: Optional[str] = "num_ctx 4096\\ntemperature 0.7"
    template: Optional[str] = "{{ .Prompt }}"
    details: ModelDetails
    system: Optional[str] = "You are a helpful AI assistant."
    messages: Optional[List[Dict[str,str]]] = None

class PullRequest(BaseModel):
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = True # Ensure stream is here and defaults to True

class GenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = ""
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    keep_alive: Optional[Union[str, int, float]] = "5m"
    context: Optional[List[int]] = None

class GenerateResponseChunk(BaseModel):
    model: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    response: Optional[str] = None
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    context: Optional[List[int]] = None
    done_reason: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = True
    keep_alive: Optional[Union[str, int, float]] = "5m"

class ChatResponseChunk(BaseModel):
    model: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    message: Optional[ChatMessage] = None
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None

class CreateRequest(BaseModel):
    name: str
    modelfile: str
    stream: Optional[bool] = False
    path: Optional[str] = None

class CopyRequest(BaseModel):
    source: str
    destination: str

class DeleteRequest(BaseModel):
    name: str

class PushRequest(BaseModel):
    name: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = True

class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None

class EmbeddingsResponse(BaseModel):
    embedding: List[float]

class RunningModelInfo(BaseModel):
    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    expires_at: Optional[str] = None
    size_vram: Optional[int] = None

class ListRunningModelsResponse(BaseModel):
    models: List[RunningModelInfo]

# --- Helper Functions ---
def parse_keep_alive(keep_alive: Union[str, int, float]) -> float:
    if isinstance(keep_alive, str):
        if keep_alive.lower() == "-1" or keep_alive.lower() == "indefinite":
            return float('inf') 
        if keep_alive.endswith("s"):
            return float(keep_alive[:-1])
        elif keep_alive.endswith("m"):
            return float(keep_alive[:-1]) * 60
        elif keep_alive.endswith("h"):
            return float(keep_alive[:-1]) * 3600
        try:
            return float(keep_alive)
        except ValueError:
            return 300.0
    elif isinstance(keep_alive, (int, float)):
        return float(keep_alive)
    return 300.0

async def manage_model_loading(model_name: str, keep_alive_str: Union[str, int, float]):
    global LOADED_MODELS
    load_duration_ns = 0
    current_time = time.time()
    LOADED_MODELS = {m: exp for m, exp in LOADED_MODELS.items() if exp > current_time or exp == float('inf')}

    if model_name not in LOADED_MODELS or LOADED_MODELS.get(model_name, 0) <= current_time :
        print(f"Mock loading model: {model_name}")
        await asyncio.sleep(0.5)
        load_duration_ns = int(0.5 * 1_000_000_000)
        print(f"Mock model {model_name} loaded.")
    
    keep_alive_seconds = parse_keep_alive(keep_alive_str)

    if keep_alive_seconds == 0:
        if model_name in LOADED_MODELS:
            del LOADED_MODELS[model_name]
        print(f"Mock model {model_name} unloaded immediately due to keep_alive=0.")
        return load_duration_ns, "unload"
    elif keep_alive_seconds == float('inf'):
        LOADED_MODELS[model_name] = float('inf')
        print(f"Mock model {model_name} loaded indefinitely.")
    else:
        expiration_time = current_time + keep_alive_seconds
        LOADED_MODELS[model_name] = expiration_time
        print(f"Mock model {model_name} loaded. Keep-alive set for {keep_alive_seconds}s. Expires at {datetime.datetime.fromtimestamp(expiration_time)}.")
    
    return load_duration_ns, "load"

# --- API Endpoints ---

@app.get("/api/version")
async def get_version():
    return {"version": "0.1.mock", "ollama_version": "0.1.40"}

@app.get("/api/tags", response_model=TagsResponse)
async def list_local_models():
    mock_model_details = ModelDetails(
        family="mock-family", 
        parameter_size="7B", 
        quantization_level="Q4_0"
    )
    return TagsResponse(models=[
        ModelTagInfo(
            name=MOCK_MODEL_NAME,
            model=MOCK_MODEL_NAME,
            modified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            size=MOCK_MODEL_SIZE,
            digest=MOCK_MODEL_DIGEST,
            details=mock_model_details
        ),
        ModelTagInfo(
            name="another-mock:latest",
            model="another-mock:latest",
            modified_at=(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat(),
            size=MOCK_MODEL_SIZE // 2,
            digest="sha256:anotherdigestfedcba0987654321",
            details=ModelDetails(family="mock-family-2", parameter_size="3B", quantization_level="Q5_K_M")
        )
    ])

@app.post("/api/show", response_model=ShowResponse)
async def show_model_information(request: ShowRequest):
    if request.name != MOCK_MODEL_NAME and not request.name.startswith("another-mock"):
        raise HTTPException(status_code=404, detail=f"model '{request.name}' not found")
    
    details = ModelDetails()
    if request.name.startswith("another-mock"):
        details = ModelDetails(family="mock-family-2", parameter_size="3B", quantization_level="Q5_K_M")

    return ShowResponse(
        details=details,
        modelfile=f"# Modelfile for {request.name}\nFROM /path/to/mock/weights\nPARAMETER temperature 0.8",
        parameters="num_ctx=2048\ntemperature=0.8",
        template="User: {{ .Prompt }}\nAssistant:",
        system=f"You are a mock AI for model {request.name}."
    )

async def pull_stream_generator(model_name: str, insecure: Optional[bool] = False):
    yield json.dumps({"status": "pulling manifest"}) + "\\n"
    await asyncio.sleep(0.2)
    
    total_size = MOCK_MODEL_SIZE 
    layers_to_simulate = 3
    layer_digests = []

    # Simulate layer downloads
    for i in range(layers_to_simulate):
        layer_digest = f"sha256:mocklayer{i+1}digest" + ("f" * (64 - 20 - len(str(i+1)))) # Unique digest
        layer_digests.append(layer_digest)
        layer_size = total_size // layers_to_simulate
        
        # Initial download status for the layer (completed key may not be included yet)
        yield json.dumps({"status": f"downloading {layer_digest}", "digest": layer_digest, "total": layer_size}) + "\\n"
        await asyncio.sleep(0.1) # Brief pause before showing progress

        completed_bytes = 0
        steps = 5
        for step in range(steps):
            await asyncio.sleep(0.15) 
            completed_bytes = int(((step + 1) / steps) * layer_size)
            # Ensure completed does not exceed total, and is total at the last step
            completed_bytes = min(completed_bytes, layer_size)
            if step == steps - 1:
                completed_bytes = layer_size
            yield json.dumps({"status": f"downloading {layer_digest}", "digest": layer_digest, "total": layer_size, "completed": completed_bytes}) + "\\n"
        
    # Simulate verification for each layer
    for layer_digest in layer_digests:
        yield json.dumps({"status": f"verifying sha256 {layer_digest}", "digest": layer_digest, "total": total_size // layers_to_simulate, "completed": total_size // layers_to_simulate}) + "\\n"
        await asyncio.sleep(0.1)

    yield json.dumps({"status": "writing manifest"}) + "\\n"
    await asyncio.sleep(0.1)
    
    yield json.dumps({"status": "removing any unused layers"}) + "\\n"
    await asyncio.sleep(0.1)
    
    yield json.dumps({"status": "success"}) + "\\n"

@app.post("/api/pull")
async def pull_model(request: PullRequest):
    print(f"POST /api/pull received. Data: {request.model_dump(exclude_unset=True)}")
    if request.insecure:
        print(f"Note: Pulling model '{request.model}' with insecure flag set to True.")

    # Allow any model name for pull simulation, not just MOCK_MODEL_NAME
    # if request.model != MOCK_MODEL_NAME and not request.model.startswith("another-mock"):
    #     print(f"Simulating pull for unrecognized model: {request.model}")
    
    # Default to streaming if stream is None or True
    if request.stream is None or request.stream is True:
        return StreamingResponse(pull_stream_generator(request.model, request.insecure), media_type="application/x-ndjson")
    else:
        # Simulate work for non-streaming pull
        print(f"Simulating non-streaming pull for model: {request.model}")
        await asyncio.sleep(1) # Simulate the total time for all steps
        return {"status": "success"}

async def generate_stream_generator(req: GenerateRequest):
    load_duration_ns, load_status = await manage_model_loading(req.model, req.keep_alive)
    if req.prompt == "" and req.keep_alive == 0 :
        yield json.dumps(GenerateResponseChunk(model=req.model, done=True, load_duration=load_duration_ns, done_reason="unload").model_dump(exclude_none=True)) + "\\n"
        return
    elif req.prompt == "":
         yield json.dumps(GenerateResponseChunk(model=req.model, done=True, load_duration=load_duration_ns, done_reason="load").model_dump(exclude_none=True)) + "\\n"
         return

    mock_response_words = ["This", "is", "a", "mock", "response", "for", req.prompt[:20] + "..."]
    if req.format == "json":
        # Corrected f-string usage within the list
        prompt_snippet = req.prompt[:20].replace('"', '\\\\"') # Escape quotes in prompt snippet
        mock_response_words = ['{"key":', '"mock_value",', '"input_prompt":', f'"{prompt_snippet}..."', '}']
    
    full_response_text = ""
    for i, word in enumerate(mock_response_words):
        is_done = (i == len(mock_response_words) - 1)
        chunk = GenerateResponseChunk(model=req.model, response=word + (" " if not is_done and req.format != "json" else ""), done=False)
        full_response_text += chunk.response
        yield json.dumps(chunk.model_dump(exclude_none=True)) + "\n"
        await asyncio.sleep(0.1)

    prompt_eval_count = len(req.prompt.split()) if req.prompt else 0
    eval_count = len(full_response_text.split())
    total_duration_ns = int(1.5 * 1_000_000_000)
    prompt_eval_duration_ns = int(0.2 * 1_000_000_000)
    eval_duration_ns = total_duration_ns - prompt_eval_duration_ns - load_duration_ns
    if eval_duration_ns < 0: eval_duration_ns = int(0.1 * 1_000_000_000)

    final_chunk = GenerateResponseChunk(
        model=req.model, response="", done=True, total_duration=total_duration_ns, load_duration=load_duration_ns,
        prompt_eval_count=prompt_eval_count, prompt_eval_duration=prompt_eval_duration_ns, eval_count=eval_count,
        eval_duration=eval_duration_ns, context=[1,2,3] if not req.raw else None, done_reason="stop"
    )
    yield json.dumps(final_chunk.model_dump(exclude_none=True)) + "\n"

@app.post("/api/generate")
async def generate_completion(req: GenerateRequest):
    if req.model != MOCK_MODEL_NAME and not req.model.startswith("another-mock"):
        print(f"Warning: Generating for a non-declared mock model: {req.model}")

    if req.stream:
        return StreamingResponse(generate_stream_generator(req), media_type="application/x-ndjson")
    else:
        load_duration_ns, _ = await manage_model_loading(req.model, req.keep_alive)
        if req.prompt == "" and req.keep_alive == 0:
             return GenerateResponseChunk(model=req.model, done=True, load_duration=load_duration_ns, done_reason="unload").model_dump(exclude_none=True)
        elif req.prompt == "":
            return GenerateResponseChunk(model=req.model, done=True, load_duration=load_duration_ns, done_reason="load").model_dump(exclude_none=True)

        mock_response = "This is a mock non-streaming response for " + (req.prompt[:30] + "..." if req.prompt else "an empty prompt")
        if req.format == "json":
            mock_response = json.dumps({"key": "mock_value", "input_prompt": req.prompt[:30] + "..."})
        
        prompt_eval_count = len(req.prompt.split()) if req.prompt else 0
        eval_count = len(mock_response.split())
        total_duration_ns = int(1.6 * 1_000_000_000)
        prompt_eval_duration_ns = int(0.25 * 1_000_000_000)
        eval_duration_ns = total_duration_ns - prompt_eval_duration_ns - load_duration_ns
        if eval_duration_ns < 0: eval_duration_ns = int(0.1 * 1_000_000_000)

        return GenerateResponseChunk(
            model=req.model, response=mock_response, done=True, total_duration=total_duration_ns, load_duration=load_duration_ns,
            prompt_eval_count=prompt_eval_count, prompt_eval_duration=prompt_eval_duration_ns, eval_count=eval_count,
            eval_duration=eval_duration_ns, context=[1,2,3] if not req.raw else None, done_reason="stop"
        ).model_dump(exclude_none=True)

async def chat_stream_generator(req: ChatRequest):
    load_duration_ns, _ = await manage_model_loading(req.model, req.keep_alive)
    last_user_message = ""
    if req.messages:
        for msg in reversed(req.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
    
    mock_response_words = ["Okay,", "I", "will", "mock", "a", "chat", "response", "regarding:", last_user_message[:20] + "..."]
    if req.format == "json":
        # Corrected f-string usage within the list
        message_snippet = last_user_message[:20].replace('"', '\\\\"') # Escape quotes in message snippet
        mock_response_words = ['{"chat_response":', '"mocked",', '"user_query":', f'"{message_snippet}..."', '}']

    full_response_content = ""
    for i, word in enumerate(mock_response_words):
        is_done = (i == len(mock_response_words) - 1)
        content_chunk = word + (" " if not is_done and req.format != "json" else "")
        full_response_content += content_chunk
        chunk = ChatResponseChunk(model=req.model, message=ChatMessage(role="assistant", content=content_chunk), done=False)
        yield json.dumps(chunk.model_dump(exclude_none=True)) + "\n"
        await asyncio.sleep(0.1)

    prompt_eval_count = sum(len(m.content.split()) for m in req.messages)
    eval_count = len(full_response_content.split())
    total_duration_ns = int(1.8 * 1_000_000_000) 
    prompt_eval_duration_ns = int(0.3 * 1_000_000_000)
    eval_duration_ns = total_duration_ns - prompt_eval_duration_ns - load_duration_ns
    if eval_duration_ns < 0: eval_duration_ns = int(0.1 * 1_000_000_000)

    final_chunk = ChatResponseChunk(
        model=req.model, message=None, done=True, total_duration=total_duration_ns, load_duration=load_duration_ns,
        prompt_eval_count=prompt_eval_count, prompt_eval_duration=prompt_eval_duration_ns, eval_count=eval_count,
        eval_duration=eval_duration_ns, done_reason="stop"
    )
    yield json.dumps(final_chunk.model_dump(exclude_none=True)) + "\n"

@app.post("/api/chat")
async def generate_chat_completion(req: ChatRequest):
    if req.model != MOCK_MODEL_NAME and not req.model.startswith("another-mock"):
        print(f"Warning: Chatting with a non-declared mock model: {req.model}")

    if req.stream:
        return StreamingResponse(chat_stream_generator(req), media_type="application/x-ndjson")
    else:
        load_duration_ns, _ = await manage_model_loading(req.model, req.keep_alive)
        last_user_message = ""
        if req.messages:
            for msg in reversed(req.messages):
                if msg.role == "user":
                    last_user_message = msg.content
                    break
        
        mock_content = "This is a mock non-streaming chat response regarding: " + last_user_message[:30] + "..."
        if req.format == "json":
            mock_content = json.dumps({"chat_response": "mocked", "user_query": last_user_message[:30] + "..."})

        prompt_eval_count = sum(len(m.content.split()) for m in req.messages)
        eval_count = len(mock_content.split())
        total_duration_ns = int(1.9 * 1_000_000_000)
        prompt_eval_duration_ns = int(0.35 * 1_000_000_000)
        eval_duration_ns = total_duration_ns - prompt_eval_duration_ns - load_duration_ns
        if eval_duration_ns < 0: eval_duration_ns = int(0.1 * 1_000_000_000)

        return ChatResponseChunk(
            model=req.model, message=ChatMessage(role="assistant", content=mock_content), done=True,
            total_duration=total_duration_ns, load_duration=load_duration_ns, prompt_eval_count=prompt_eval_count,
            prompt_eval_duration=prompt_eval_duration_ns, eval_count=eval_count, eval_duration=eval_duration_ns,
            done_reason="stop"
        ).model_dump(exclude_none=True)

async def create_model_stream_generator(model_name: str):
    yield json.dumps({"status": f"creating model '{model_name}'"}) + "\n"
    await asyncio.sleep(0.1)
    yield json.dumps({"status": "parsing modelfile"}) + "\n"
    await asyncio.sleep(0.2)
    yield json.dumps({"status": "reading model metadata"}) + "\n"
    await asyncio.sleep(0.1)
    yield json.dumps({"status": "writing model manifest"}) + "\n"
    await asyncio.sleep(0.2)
    yield json.dumps({"status": "success", "digest": MOCK_MODEL_DIGEST}) + "\n"

@app.post("/api/create")
async def create_model(request: CreateRequest):
    print(f"Mock creating model: {request.name}")
    if request.modelfile:
        print(f"Using modelfile content for {request.name}")
    elif request.path:
        print(f"Using modelfile from path: {request.path} for {request.name}")
    else:
        raise HTTPException(status_code=400, detail="Either modelfile content or path must be provided")

    if request.stream:
        return StreamingResponse(create_model_stream_generator(request.name), media_type="application/x-ndjson")
    else:
        await asyncio.sleep(0.5)
        return {"status": "success", "digest": MOCK_MODEL_DIGEST}

@app.post("/api/copy")
async def copy_model(request: CopyRequest):
    print(f"Mock copying model from {request.source} to {request.destination}")
    await asyncio.sleep(0.1)
    return JSONResponse(content={"message": f"model '{request.source}' copied to '{request.destination}'"}, status_code=200)

@app.delete("/api/delete")
async def delete_model_endpoint(request: DeleteRequest):
    print(f"Mock deleting model: {request.name}")
    if request.name in LOADED_MODELS:
        del LOADED_MODELS[request.name]
        print(f"Unloaded model {request.name} as part of deletion.")
    await asyncio.sleep(0.1)
    return JSONResponse(content={"message": f"model '{request.name}' deleted"}, status_code=200)

async def push_model_stream_generator(model_name: str):
    yield json.dumps({"status": f"pushing model '{model_name}'"}) + "\n"
    await asyncio.sleep(0.2)
    total_size = MOCK_MODEL_SIZE
    layers_to_simulate = 2
    for i in range(layers_to_simulate):
        layer_digest = f"sha256:pushlayer{i+1}digest" + ("e" * (64-20-len(str(i+1))))
        layer_size = total_size // layers_to_simulate
        pushed_bytes = 0
        yield json.dumps({"status": "pushing", "digest": layer_digest, "total": layer_size, "completed": pushed_bytes}) + "\n"
        await asyncio.sleep(0.1)
        steps = 3
        for step in range(steps):
            await asyncio.sleep(0.15) 
            pushed_bytes = int(((step + 1) / steps) * layer_size)
            if step == steps - 1:
                pushed_bytes = layer_size
            yield json.dumps({"status": "pushing", "digest": layer_digest, "total": layer_size, "completed": pushed_bytes}) + "\n"
    yield json.dumps({"status": "success"}) + "\n"

@app.post("/api/push")
async def push_model(request: PushRequest):
    print(f"Mock pushing model: {request.name}")
    if request.stream:
        return StreamingResponse(push_model_stream_generator(request.name), media_type="application/x-ndjson")
    else:
        await asyncio.sleep(1)
        return {"status": "success"}

@app.post("/api/embeddings", response_model=EmbeddingsResponse)
async def generate_embeddings(request: EmbeddingsRequest):
    print(f"Mock generating embeddings for model: {request.model} with prompt: \"{request.prompt[:30]}...\"")
    await manage_model_loading(request.model, "5m")
    await asyncio.sleep(0.2)
    mock_embedding = [0.1 + (i * 0.01) for i in range(128)]
    return EmbeddingsResponse(embedding=mock_embedding)

@app.get("/api/ps", response_model=ListRunningModelsResponse)
async def list_running_models():
    global LOADED_MODELS
    running_models_info: List[RunningModelInfo] = []
    current_time = time.time()
    active_loaded_models = {}
    for model_name, expires_at_ts in LOADED_MODELS.items():
        if expires_at_ts > current_time or expires_at_ts == float('inf'):
            active_loaded_models[model_name] = expires_at_ts
            details = ModelDetails()
            size = MOCK_MODEL_SIZE
            digest = MOCK_MODEL_DIGEST
            if model_name.startswith("another-mock"):
                details = ModelDetails(family="mock-family-2", parameter_size="3B", quantization_level="Q5_K_M")
                size = MOCK_MODEL_SIZE // 2
                digest = "sha256:anotherdigestfedcba0987654321"
            elif model_name != MOCK_MODEL_NAME:
                details = ModelDetails(family="dynamic-mock", parameter_size="?", quantization_level="?")
            expires_at_str = datetime.datetime.fromtimestamp(expires_at_ts).isoformat() if expires_at_ts != float('inf') else "infinity"
            running_models_info.append(
                RunningModelInfo(
                    name=model_name, model=model_name, size=size, digest=digest, details=details,
                    expires_at=expires_at_str, size_vram=size // 2
                )
            )
    LOADED_MODELS = active_loaded_models
    return ListRunningModelsResponse(models=running_models_info)

if __name__ == "__main__":
    import uvicorn
    print("Starting mock Ollama server on http://localhost:11434")
    uvicorn.run(app, host="0.0.0.0", port=11435)

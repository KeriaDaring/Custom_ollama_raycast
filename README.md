# Raycast Custom Ollama Server

This project provides a mock Ollama server designed to integrate with Raycast for Large Language Model (LLM) services. It primarily focuses on proxying chat requests to an upstream LLM service and supports streaming responses in the Ollama format.

## Description

The main goal of this server is to act as a bridge between Raycast (when configured to use a custom Ollama endpoint) and another LLM inference API. It implements key Ollama API endpoints like `/api/tags`, `/api/show`, `/api/pull`, and most importantly, `/api/chat` (from `ollama.py`).

The `/api/chat` endpoint is designed to forward requests to an external service (currently configured in `ollama.py` for `https://inference.nebulablock.com/v1/chat/completions` but can be modified) and then transforms the responses, including streaming chunks, into the standard Ollama format.

**Important Note on Raycast Streaming:**
Currently, there appears to be an issue with Raycast's handling of streaming responses from custom Ollama endpoints, even when the "Advanced" option for streaming is enabled in Raycast's model settings. This can sometimes lead to decoding errors or unexpected behavior in Raycast. This project attempts to adhere to the Ollama streaming format, and further investigation may be needed to fully resolve Raycast-specific compatibility issues.

## Features

*   Mock implementations of `/api/tags`, `/api/show`, `/api/pull` in `ollama.py`.
*   Proxying for `/api/chat` to an upstream LLM service in `ollama.py`.
*   Support for streaming chat responses in Ollama format.
*   Basic structure for easy modification and extension.
*   Includes `server.py` which is a more comprehensive, but currently less focused, mock Ollama server.

## How to Use

### Prerequisites

*   Python 3.8+
*   Pip (Python package installer)
*   Git

### Setup

1.  **Clone the repository :**
    ```bash
    git clone https://github.com/KeriaDaring/Custom_ollama_raycast.git
    ```

2.  **Navigate to the project directory (if not already there):**
    ```bash
    cd Custom_ollama_raycast
    ```

3.  **Install dependencies:**
    Make sure `requirements.txt` lists all necessary packages (e.g., `fastapi`, `uvicorn`, `requests`, `pydantic`).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key (Important for `/api/chat` in `ollama.py`):**
    The `/api/chat` endpoint in `ollama.py` proxies requests to an external service. It currently uses a hardcoded API key.
    Please replace the configuration in *_client.py, or you can write a client for your own.

### Running the Server

1.  **Start the FastAPI server (using `ollama.py` for the Raycast proxy):**
    ```bash
    python ollama.py
    ```
    The server will typically start on `http://0.0.0.0:8000`.

2.  **Configure Raycast:**
    *   Open Raycast settings.
    *   Go to the AI / Large Language Models section.
    *   Add a new Ollama model.
    *   Set the **Ollama API Host** to `http://localhost:8000` (or `http://127.0.0.1:8000`).
    *   For the **Model Name**, you can use "deepseek" (as this is what `/api/tags` in `ollama.py` currently returns by default).
    *   In the model's **Advanced** settings within Raycast, ensure **Streaming** is enabled if you want to test streaming responses. (Be mindful of the potential Raycast-specific issues mentioned above).

### Other Endpoints

You can also modify the mock responses for `/api/tags`, `/api/pull`, and `/api/show` in `ollama.py` to better suit your testing needs. For example, `show.json` is used by `/api/show` and its content can be updated.

## Contributing

Feel free to fork this repository, make changes, and submit pull requests. If you encounter issues or have suggestions, please open an issue on GitHub.

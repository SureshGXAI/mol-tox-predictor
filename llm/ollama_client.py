"""
Minimal client for a local Ollama server (https://ollama.com).

Replaces the Groq / Llama-4-Scout dependency with a fully local model, so the
whole system runs offline with no API keys. Default model is `llama3.1` but any
pulled model works (e.g. `mistral`, `qwen2.5`, `gemma2`).

Setup:
    curl -fsSL https://ollama.com/install.sh | sh   # or download the app
    ollama pull llama3.1
    ollama serve      # usually already running as a service
"""
import os
import json
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def is_available(host: str = None, timeout: float = 2.0) -> bool:
    host = host or OLLAMA_HOST
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_models(host: str = None) -> list:
    host = host or OLLAMA_HOST
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def chat(prompt: str, system: str = None, model: str = None,
         host: str = None, temperature: float = 0.3, timeout: float = 120) -> str:
    """Single-turn chat completion. Raises RuntimeError if the server is down."""
    host = host or OLLAMA_HOST
    model = model or OLLAMA_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature}}
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed ({host}, model={model}): {e}")
    return r.json().get("message", {}).get("content", "").strip()

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import importlib
import os

router = APIRouter()

# Definisikan model input sesuai standar OpenAI API
class CompletionRequest(BaseModel):
    model: str
    messages: list[dict]  # Daftar pesan dengan format {"role": "user", "content": "message"}
    temperature: float = 1.0
    max_tokens: int = 256
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    Endpoint untuk menangani permintaan model LLM secara dinamis.
    """
    model_name = request.model
    messages = request.messages
    temperature = request.temperature
    top_p = request.top_p
    frequency_penalty = request.frequency_penalty
    presence_penalty = request.presence_penalty

    # Ambil API key dari environment variable
    api_key_env_var = f"{model_name.upper()}_API_KEY"
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=f"Environment variable '{api_key_env_var}' is not set."
        )

    # Import modul model secara dinamis
    try:
        module = importlib.import_module(f"app.llm.{model_name}")
        if not hasattr(module, "GeminiClient"):
            raise HTTPException(status_code=500, detail=f"Model '{model_name}' is not properly implemented.")
        
        # Inisialisasi client dan proses permintaan
        client = module.GeminiClient(api_key=api_key)
        response_text = client.generate_content(
            prompt=[msg["content"] for msg in messages],
            system_instruction="",
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return {
            "model": model_name,
            "choices": [{"message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
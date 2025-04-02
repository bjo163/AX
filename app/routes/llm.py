from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import importlib
import os

router = APIRouter()

# Definisikan model input sesuai standar OpenAI API
class CompletionRequest(BaseModel):
    model: str = Field(..., example="gemini")  # Contoh model API yang digunakan
    messages: list[dict] = Field(..., example=[  # Contoh format pesan
        {"role": "user", "content": "Tell me a joke."},
        # {"role": "assistant", "content": "Why don’t skeletons fight each other? They don’t have the guts."}
    ])
    temperature: float = Field(1.0, example=0.7)  # Contoh nilai suhu
    max_tokens: int = Field(256, example=150)  # Contoh nilai max_tokens
    top_p: float = Field(1.0, example=0.95)  # Contoh nilai top_p
    frequency_penalty: float = Field(0.0, example=0.0)  # Contoh nilai frequency_penalty
    presence_penalty: float = Field(0.0, example=0.0)  # Contoh nilai presence_penalty

# Definisikan model respons
class CompletionResponse(BaseModel):
    model: str
    choices: list[dict]

@router.post("/v1/completions", response_model=CompletionResponse)
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

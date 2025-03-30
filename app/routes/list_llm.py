from fastapi import APIRouter
import os
import importlib

router = APIRouter()

@router.get("/v1/llms")
async def list_llms():
    """
    Endpoint untuk me-list semua LLM yang tersedia di folder 'llm' dengan detail tambahan.
    """
    llm_folder = os.path.join(os.path.dirname(__file__), "..", "llm")
    data = []

    # Iterasi semua file di folder 'llm'
    for filename in os.listdir(llm_folder):
        if filename.endswith(".py") and filename != "__init__.py":
            llm_name = filename[:-3]  # Hapus ekstensi '.py'

            # Import modul secara dinamis
            try:
                module = importlib.import_module(f"app.llm.{llm_name}")
                # Ambil properti tambahan dari modul, jika ada
                description = getattr(module, "DESCRIPTION", "No description available")
                version = getattr(module, "VERSION", "Unknown version")
                author = getattr(module, "AUTHOR", "Unknown author")
            except Exception as e:
                # Jika ada error saat import, tambahkan informasi error
                description = f"Error loading module: {str(e)}"
                version = "N/A"
                author = "N/A"

            # Tambahkan detail ke daftar LLM
            data.append({
                "name": llm_name,
                "description": description,
                "version": version,
                "author": author
            })

    return {"data": data}
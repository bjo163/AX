from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.db.configuration import create_supabase_client
from app.routes.home import router as home_router
import logging
from config.logging_config import LOGGING_CONFIG
from config.logging_config import logger
import time
import os
import importlib

app = FastAPI()

# Initialize Supabase client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Ganti dengan URL frontend Anda
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode (GET, POST, dll.)
    allow_headers=["*"],  # Izinkan semua header
)

try:
    app.state.supabase = create_supabase_client()
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to initialize Supabase client: {str(e)}")

# Fungsi untuk memuat file di folder /llm/ dan sinkronisasi ke Supabase
# Fungsi untuk memuat file di folder /llm/ dan sinkronisasi ke Supabase
def sync_llm_to_supabase():
    """
    Memuat semua file Python di folder /llm/ dan menyimpan informasi ke tabel 'llm' di Supabase.
    """
    folder_path = os.path.join(os.path.dirname(__file__), "llm")
    llm_data = []

    # Iterasi semua file Python di folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename != "__init__.py":
            file_path = os.path.join(folder_path, filename)

            # Muat file Python secara dinamis
            spec = importlib.util.spec_from_file_location(filename[:-3], file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Ambil informasi dari file (jika tersedia)
            version = getattr(module, "VERSION", "unknown")
            description = getattr(module, "DESCRIPTION", "No description available")
            author = getattr(module, "AUTHOR", "Unknown author")

            # Tambahkan ke daftar data
            llm_data.append({
                "name": filename[:-3],
                "version": version,
                "system_instruction": description,
                "temperature": 1.0,  # Default value
                "top_p": 1.0,        # Default value
                "frequency_penalty": 0.0,  # Default value
                "presence_penalty": 0.0,   # Default value
                "error_message": None,
            })

    # Sinkronisasi data ke Supabase
    try:
        supabase = app.state.supabase
        response = supabase.table("llm").upsert(llm_data, on_conflict=["name"]).execute()
        res = response.dict()
        # Periksa respons dari Supabase
        if res.get("error"):
            print("Error saat menyimpan ke Supabase:", 
                #   res.get("error")
                  )
        else:
            print("Berhasil menyimpan ke Supabase:", )
    except Exception as e:
        print("Kesalahan saat menghubungkan ke Supabase:", e)

# Panggil fungsi sinkronisasi saat aplikasi dimulai
sync_llm_to_supabase()

# Auto-import all routers in the 'routes' folder
routes_folder = os.path.join(os.path.dirname(__file__), "routes")
for filename in os.listdir(routes_folder):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f"app.routes.{filename[:-3]}"  # Remove '.py' from filename
        module = importlib.import_module(module_name)
        if hasattr(module, "router"):
            app.include_router(module.router, prefix="/api")  # Add prefix 'api'

# Print all available routes
print("\nAvailable Routes:")
for route in app.routes:
    print(f"Path: {route.path}, Methods: {route.methods}")
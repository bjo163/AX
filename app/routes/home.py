from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import os
import importlib
import inspect
import time
from datetime import datetime
from app.llm.gemini import GeminiClient  # Pastikan GeminiClient sudah diimplementasikan

router = APIRouter()

CACHE_FILE = "cached_home.html"  # File untuk menyimpan hasil HTML
CACHE_EXPIRATION = 5 * 60  # Waktu kedaluwarsa cache dalam detik (5 menit)

def is_cache_valid():
    """
    Periksa apakah cache masih valid berdasarkan waktu kedaluwarsa.
    """
    if not os.path.exists(CACHE_FILE):
        return False
    last_modified = os.path.getmtime(CACHE_FILE)
    current_time = time.time()
    return (current_time - last_modified) < CACHE_EXPIRATION

def save_to_cache(content):
    """
    Simpan konten HTML ke file cache.
    """
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(content)

def load_from_cache():
    """
    Muat konten HTML dari file cache.
    """
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return f.read()

@router.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Halaman beranda API yang dinamis dengan analisis seluruh file Python di proyek.
    """
    # Periksa apakah cache masih valid
    if is_cache_valid():
        return HTMLResponse(content=load_from_cache())

    project_folder = os.path.join(os.path.dirname(__file__), "..", "..")
    analyzed_files = []

    # Proses 1: Analisis File Python
    for root, dirs, files in os.walk(project_folder):
        # Abaikan folder yang diawali dengan '.' atau '__'
        dirs[:] = [d for d in dirs if not (d.startswith(".") or d.startswith("__") or d == "node_modules")]

        for filename in files:
            if filename.endswith(".py") and filename != "__init__.py":
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, project_folder)
                module_name = relative_path.replace(os.sep, ".")[:-3]  # Convert to module name

                file_info = {"filename": relative_path, "properties": []}

                try:
                    # Import modul secara dinamis
                    module = importlib.import_module(module_name)

                    # Ambil semua atribut di modul
                    for name, obj in inspect.getmembers(module):
                        if not name.startswith("__"):  # Abaikan atribut bawaan
                            file_info["properties"].append({
                                "name": name,
                                "type": str(type(obj)),
                                "doc": inspect.getdoc(obj) or "No documentation available"
                            })
                except Exception as e:
                    file_info["error"] = f"Error loading module: {str(e)}"

                analyzed_files.append(file_info)

    # Ambil API key dari environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Environment variable 'GEMINI_API_KEY' is not set or invalid."
        )

    gemini_client = GeminiClient(api_key=api_key)

    # Proses 2: Ringkasan Analisis
    analysis_prompt = (
        "You are an expert technical writer. Analyze the following Python files from a project and create "
        "a concise, professional summary in Markdown format. Include key features, file descriptions, and "
        "any notable details:\n\n"
        f"{analyzed_files}"
    )
    gemini_analysis_response = gemini_client.generate_content(prompt=analysis_prompt)

    # Proses 3: Struktur Halaman HTML
    html_structure_prompt = (
        "You are a professional web developer. Based on the following Markdown summary, create a clean and "
        "well-structured HTML page. Include sections for an overview, analyzed files, and key features:\n\n"
        f"{gemini_analysis_response}"
    )
    gemini_html_structure_response = gemini_client.generate_content(prompt=html_structure_prompt)

    # Proses 4: Desain Halaman HTML
    html_design_prompt = (
        "You are a UI/UX designer. Enhance the following HTML structure by adding modern design elements, "
        "such as colors, typography, and layout improvements. Ensure the page is visually appealing and "
        "easy to navigate:\n\n"
        f"{gemini_html_structure_response}"
    )
    gemini_html_response = gemini_client.generate_content(prompt=html_design_prompt)

    # Proses 5: Knowledge Base
    knowledge_base_prompt = (
        "You are an expert in creating knowledge bases. Add a knowledge base section to the following HTML "
        "page. Include a table of contents, navigation links, and additional helpful content for users:\n\n"
        f"{gemini_html_response}"
    )
    gemini_knowledge_base_response = gemini_client.generate_content(prompt=knowledge_base_prompt)

    # Proses 6: Tanggal Rilis dan Optimasi
    release_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    optimization_prompt = (
        "You are a futurist web developer. Add a release date ('Generated on: {release_date}') at the top "
        "of the following HTML page. Ensure the page is optimized for readability and includes subtle "
        "animations for a modern feel:\n\n"
        f"{gemini_knowledge_base_response}"
    )
    gemini_optimized_response = gemini_client.generate_content(prompt=optimization_prompt)

    # Tambahkan tanggal rilis ke HTML
    final_html = f"""
    <div style="text-align: right; font-size: 12px; color: gray;">
        Generated on: {release_date}
    </div>
    {gemini_optimized_response}
    """

    # Simpan hasil ke cache
    save_to_cache(final_html)

    # Kembalikan hasil HTML langsung
    return HTMLResponse(content=final_html)
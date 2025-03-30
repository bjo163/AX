from supabase import Client, create_client
from config.database_config import APP_DB_KEY, APP_DB_URL
import os
import socket
import uuid

def create_supabase_client():
    """Inisialisasi Supabase Client dan melakukan upsert data identifikasi perangkat."""
    try:
        supabase: Client = create_client(APP_DB_URL, APP_DB_KEY)
    except (ValueError, ConnectionError) as e:
        print("Gagal menghubungkan ke Supabase:", e)
        return None  # Jika gagal koneksi, return None

    # Mengumpulkan informasi sistem
    hostname = socket.gethostname()
    instance_id = str(uuid.uuid4())
    os_name = os.name
    os_platform = os.sys.platform

    # Mendapatkan IP Address lokal
    try:
        ip_address = socket.gethostbyname(hostname)
    except Exception:
        ip_address = "Unknown"

    # Susun resource yang akan di-upsert
    resources = [
        {"name": "hostname", "resource_type": "OS", "value": hostname},
        {"name": "instance_id", "resource_type": "OS", "value": instance_id},
        {"name": "os_name", "resource_type": "OS", "value": os_name},
        {"name": "os_platform", "resource_type": "OS", "value": os_platform},
        {"name": "ip_address", "resource_type": "OS", "value": ip_address},
    ]
    
    # Lakukan upsert ke tabel 'resources'
    try:
        response = supabase.from_("resources").upsert(resources, on_conflict=["name"]).execute()
        res = response.dict()
        if res.get("error"):
            print("Error saat menyimpan ke Supabase:", 
                #   res.get("error")
                  )
        else:
            print("Berhasil menyimpan ke Supabase:", 
                #   res.get("data")
                  )
    except (ValueError, ConnectionError, TimeoutError) as e:
        print("Kesalahan saat upsert ke Supabase:", e)
    
    return supabase

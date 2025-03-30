from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from app.main import app

router = APIRouter()

# Pydantic model untuk request dan response
class AgentBase(BaseModel):
    name: str
    description: Optional[str] = None
    goal: Optional[str] = None
    llm_id: int
    configuration: Optional[dict] = None

class AgentResponse(AgentBase):
    id: int
    llm_name: str
    create_date: str
    write_date: str

    class Config:
        orm_mode = True

# Dependency untuk Supabase client
def get_supabase():
    return app.state.supabase

# Route untuk mendapatkan semua agen
# Route untuk mendapatkan semua agen
# Route untuk mendapatkan semua agen
@router.get("/v1/agents", response_model=List[AgentResponse])
async def get_agents(supabase=Depends(get_supabase)):
    try:
        # Ambil data dari tabel agent
        agent_response = supabase.table("agent").select("*").execute()
        print("Agent Response:", agent_response)  # Tambahkan logging
        if not agent_response.data:
            return []  # Kembalikan list kosong jika tidak ada data

        agents = agent_response.data

        # Ambil data dari tabel llm
        llm_response = supabase.table("llm").select("id, name").execute()
        print("LLM Response:", llm_response)  # Tambahkan logging
        if not llm_response.data:
            raise HTTPException(status_code=500, detail="Gagal mengambil data LLM")

        llms = {llm["id"]: llm["name"] for llm in llm_response.data}

        # Gabungkan data agent dengan llm_name
        for agent in agents:
            agent["llm_name"] = llms.get(agent["llm_id"], "Unknown LLM")

        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat mengambil data agen: {str(e)}")

# Route untuk mendapatkan agen berdasarkan ID
@router.get("/v1/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, supabase=Depends(get_supabase)):
    try:
        # Ambil data agen berdasarkan ID
        agent_response = supabase.table("agent").select("*").eq("id", agent_id).single().execute()
        if not agent_response.data:
            raise HTTPException(status_code=404, detail="Agen tidak ditemukan")

        agent = agent_response.data

        # Ambil data llm untuk mendapatkan llm_name
        llm_response = supabase.table("llm").select("name").eq("id", agent["llm_id"]).single().execute()
        if llm_response.data:
            agent["llm_name"] = llm_response.data["name"]
        else:
            agent["llm_name"] = "Unknown LLM"

        return agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat mengambil data agen: {str(e)}")

# Route untuk membuat agen baru
@router.post("/v1/agents", response_model=AgentResponse)
async def create_agent(agent: AgentBase, supabase=Depends(get_supabase)):
    try:
        response = supabase.table("agent").insert(agent.dict()).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Gagal membuat agen")

        created_agent = response.data[0]

        # Ambil llm_name untuk agen yang baru dibuat
        llm_response = supabase.table("llm").select("name").eq("id", created_agent["llm_id"]).single().execute()
        if llm_response.data:
            created_agent["llm_name"] = llm_response.data["name"]
        else:
            created_agent["llm_name"] = "Unknown LLM"

        return created_agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat membuat agen: {str(e)}")

# Route untuk memperbarui agen
@router.put("/v1/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: int, agent: AgentBase, supabase=Depends(get_supabase)):
    try:
        response = supabase.table("agent").update(agent.dict()).eq("id", agent_id).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Gagal memperbarui agen")

        updated_agent = response.data[0]

        # Ambil llm_name untuk agen yang diperbarui
        llm_response = supabase.table("llm").select("name").eq("id", updated_agent["llm_id"]).single().execute()
        if llm_response.data:
            updated_agent["llm_name"] = llm_response.data["name"]
        else:
            updated_agent["llm_name"] = "Unknown LLM"

        return updated_agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat memperbarui agen: {str(e)}")

# Route untuk menghapus agen
@router.delete("/v1/agents/{agent_id}")
async def delete_agent(agent_id: int, supabase=Depends(get_supabase)):
    try:
        response = supabase.table("agent").delete().eq("id", agent_id).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Gagal menghapus agen")
        return {"message": "Agen berhasil dihapus"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat menghapus agen: {str(e)}")
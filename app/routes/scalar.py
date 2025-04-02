from scalar_fastapi import get_scalar_api_reference
from fastapi import APIRouter, HTTPException
from app.main import app
router = APIRouter()
@router.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        default_open_all_tags=True
    )

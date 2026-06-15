"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import settings
from app.schemas.common import HealthResponse

app = FastAPI(
    title=settings.APP_NAME,
    description="BANKNIFTY pre-market gap direction and gap size prediction system",
    version="0.1.0",
)

app.include_router(api_router, prefix=f"/api/{settings.API_VERSION}")


@app.get("/")
async def project_info() -> dict[str, str]:
    """Return basic project information."""
    return {
        "app": settings.APP_NAME,
        "description": (
            "BANKNIFTY pre-market gap direction and gap size prediction system"
        ),
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return application health."""
    return HealthResponse(status="healthy")

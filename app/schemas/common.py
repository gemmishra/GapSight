"""Common API response schemas."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard API error response."""

    detail: str


class HealthResponse(BaseModel):
    """Application health response."""

    status: str

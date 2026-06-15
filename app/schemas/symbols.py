"""Schemas describing tradable symbols."""

from pydantic import BaseModel


class SupportedSymbol(BaseModel):
    """A symbol supported by the GapSight platform."""

    symbol: str
    name: str
    instrument_type: str
    exchange: str
    model_available: bool
    is_active: bool

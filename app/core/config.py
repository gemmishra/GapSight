"""Centralized configuration for GapSight."""

import os

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    BaseSettings = None
    SettingsConfigDict = None


class Settings(BaseSettings if BaseSettings is not None else object):
    """Application settings loaded from environment variables."""

    APP_NAME: str = "GapSight"
    API_VERSION: str = "v1"
    DEFAULT_SYMBOL: str = "BANKNIFTY"
    MODEL_DIR: str = "app/models"
    RAW_DATA_DIR: str = "app/data/raw"
    PROCESSED_DATA_DIR: str = "app/data/processed"
    DISCORD_WEBHOOK_URL: str = ""
    ENABLE_DISCORD_NOTIFICATIONS: bool = False
    OPENCLAW_API_TOKEN: str = ""
    ENABLE_OPENCLAW_AUTH: bool = False

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def __init__(self) -> None:
        if BaseSettings is not None:
            super().__init__()
            return

        for field_name in (
            "APP_NAME",
            "API_VERSION",
            "DEFAULT_SYMBOL",
            "MODEL_DIR",
            "RAW_DATA_DIR",
            "PROCESSED_DATA_DIR",
            "DISCORD_WEBHOOK_URL",
            "ENABLE_DISCORD_NOTIFICATIONS",
            "OPENCLAW_API_TOKEN",
            "ENABLE_OPENCLAW_AUTH",
        ):
            value = os.getenv(field_name, getattr(self, field_name))
            if field_name in {
                "ENABLE_DISCORD_NOTIFICATIONS",
                "ENABLE_OPENCLAW_AUTH",
            }:
                value = str(value).lower() in {"1", "true", "yes", "on"}
            setattr(self, field_name, value)


settings = Settings()

APP_NAME = settings.APP_NAME
API_VERSION = settings.API_VERSION
DEFAULT_SYMBOL = settings.DEFAULT_SYMBOL
MODEL_DIR = settings.MODEL_DIR
RAW_DATA_DIR = settings.RAW_DATA_DIR
PROCESSED_DATA_DIR = settings.PROCESSED_DATA_DIR
DISCORD_WEBHOOK_URL = settings.DISCORD_WEBHOOK_URL
ENABLE_DISCORD_NOTIFICATIONS = settings.ENABLE_DISCORD_NOTIFICATIONS
OPENCLAW_API_TOKEN = settings.OPENCLAW_API_TOKEN
ENABLE_OPENCLAW_AUTH = settings.ENABLE_OPENCLAW_AUTH

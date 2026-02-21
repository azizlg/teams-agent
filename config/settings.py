"""
settings.py — All environment variables and configuration.

Uses pydantic-settings BaseSettings to load, validate, and group all
configuration values from environment variables and .env files.
Provides a single ``settings`` instance imported across the whole project.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnvironmentType(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ---------------------------------------------------------------------------
# Grouped sub-models
# ---------------------------------------------------------------------------

class AzureBotSettings(BaseSettings):
    """Microsoft Azure Bot credentials (still needed for Teams integration)."""

    model_config = SettingsConfigDict(env_prefix="MICROSOFT_")

    app_id: str = Field(default="", description="Azure Bot App ID")
    app_password: str = Field(default="", description="Azure Bot App Password")
    app_tenant_id: str = Field(default="", description="Azure AD Tenant ID")


class MinioSettings(BaseSettings):
    """MinIO (S3-compatible) object storage configuration."""

    model_config = SettingsConfigDict(env_prefix="MINIO_")

    endpoint: str = Field(default="localhost:9000", description="MinIO server endpoint")
    access_key: str = Field(default="minioadmin", description="MinIO access key")
    secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    bucket_name: str = Field(default="meeting-audio-chunks", description="Bucket for audio chunks")
    secure: bool = Field(default=False, description="Use HTTPS for MinIO")


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection configuration."""

    url: str = Field(
        default="postgresql+asyncpg://meeting_agent:changeme@localhost:5432/meeting_agent",
        alias="DATABASE_URL",
        description="Async SQLAlchemy connection URL",
    )
    pool_size: int = Field(default=10, description="Connection pool size")
    pool_max_overflow: int = Field(default=20, description="Max pool overflow")

    model_config = SettingsConfigDict(populate_by_name=True)


class RedisSettings(BaseSettings):
    """Redis connection and Streams configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    stream_name: str = Field(default="audio-chunks", description="Stream key")
    consumer_group: str = Field(default="whisper-workers", description="Consumer group name")
    max_retries: int = Field(default=3, description="Max retries before dead-letter")


class GroqSettings(BaseSettings):
    """Groq LLM API configuration (free tier available)."""

    api_key: str = Field(default="", alias="GROQ_API_KEY", description="Groq API key")
    model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="GROQ_MODEL",
        description="Groq model name",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class WhisperSettings(BaseSettings):
    """faster-whisper model configuration."""

    model_config = SettingsConfigDict(env_prefix="WHISPER_")

    model_size: str = Field(default="large-v3", description="Whisper model size")
    device: str = Field(default="auto", description="Device: auto | cpu | cuda")
    compute_type: str = Field(default="float16", description="Compute type")
    realtime_chunk_seconds: float = Field(
        default=10.0,
        description="Audio chunk duration for near-real-time transcription (seconds)",
    )


class AudioSettings(BaseSettings):
    """Audio stream and chunk parameters."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_")

    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_duration_seconds: int = Field(default=300, description="Chunk duration (seconds)")
    silence_threshold_db: float = Field(default=-40.0, description="Silence threshold in dB")
    silence_min_duration_ms: int = Field(default=500, description="Min silence duration (ms)")

    @property
    def bytes_per_second(self) -> int:
        """Bytes per second of 16-bit PCM audio."""
        return self.sample_rate * self.channels * 2  # 16-bit = 2 bytes

    @property
    def chunk_size_bytes(self) -> int:
        """Maximum chunk size in bytes."""
        return self.bytes_per_second * self.chunk_duration_seconds


class PyannoteSettings(BaseSettings):
    """pyannote speaker diarization configuration."""

    auth_token: str = Field(default="", alias="PYANNOTE_AUTH_TOKEN", description="HuggingFace auth token")

    model_config = SettingsConfigDict(populate_by_name=True)


class NgrokSettings(BaseSettings):
    """ngrok tunnel configuration for local development."""

    authtoken: str = Field(default="", alias="NGROK_AUTHTOKEN", description="ngrok auth token")

    model_config = SettingsConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Root application settings.

    Groups all sub-settings and common top-level values.
    Loads from ``.env`` file automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Top-level
    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT,
        description="Runtime environment",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="Log output format",
    )

    # Grouped settings
    bot: AzureBotSettings = Field(default_factory=AzureBotSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    pyannote: PyannoteSettings = Field(default_factory=PyannoteSettings)
    ngrok: NgrokSettings = Field(default_factory=NgrokSettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == EnvironmentType.PRODUCTION

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, v: str) -> str:
        return v.upper() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (singleton)."""
    return Settings()


# Convenience shortcut — ``from config.settings import settings``
settings: Settings = get_settings()

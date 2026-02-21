"""
main.py — FastAPI entry point.

Defines the FastAPI application with lifespan management, meeting control
endpoints, transcript retrieval, and the Teams Bot Framework webhook receiver.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MeetingJoinRequest(BaseModel):
    """Request body for joining a Teams meeting."""

    meeting_url: str = Field(..., description="Teams meeting join URL")
    meeting_id: str | None = Field(None, description="Optional meeting identifier")
    enable_transcription: bool = Field(True, description="Enable real-time transcription")
    enable_whisper: bool = Field(True, description="Enable Whisper reprocessing")


class MeetingStatus(str, Enum):
    """Meeting lifecycle status."""

    PENDING = "pending"
    JOINING = "joining"
    ACTIVE = "active"
    LEAVING = "leaving"
    ENDED = "ended"
    ERROR = "error"


class MeetingStatusResponse(BaseModel):
    """Response for meeting status queries."""

    meeting_id: str
    status: MeetingStatus
    participants: int = 0
    transcription_active: bool = False
    chunks_processed: int = 0
    duration_seconds: float = 0.0


class TranscriptSegmentResponse(BaseModel):
    """A single transcript segment in API responses."""

    text: str
    language: str
    timestamp: float
    confidence: float
    speaker_id: str | None = None
    source: str = "whisper"  # transcription source


class TranscriptResponse(BaseModel):
    """Full transcript response."""

    meeting_id: str
    segments: list[TranscriptSegmentResponse]
    total_segments: int
    languages_detected: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "0.1.0"
    environment: str = "development"


# ---------------------------------------------------------------------------
# Application state (lightweight in-memory store for active meetings)
# ---------------------------------------------------------------------------

class AppState:
    """Shared application state — initialised during lifespan."""

    def __init__(self) -> None:
        self.redis_queue = None
        self.blob_storage = None
        self.db_engine = None
        self.speech_client = None
        self.active_meetings: dict[str, dict[str, Any]] = {}

    async def startup(self) -> None:
        """Initialise all service clients."""
        logger.info("Initialising application services...")

        # Redis
        try:
            from storage.redis_queue import RedisQueue
            self.redis_queue = RedisQueue()
            await self.redis_queue.connect()
            logger.info("✓ Redis connected")
        except Exception:
            logger.exception("✗ Redis connection failed")

        # Database
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            self.db_engine = create_async_engine(
                settings.database.url,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.pool_max_overflow,
                echo=False,
            )
            logger.info("✓ Database engine created")
        except Exception:
            logger.exception("✗ Database engine creation failed")

        logger.info("Application startup complete")

    async def shutdown(self) -> None:
        """Gracefully close all connections and flush buffers."""
        logger.info("Shutting down application services...")

        # Flush active meeting buffers
        for meeting_id, meeting_data in self.active_meetings.items():
            handler = meeting_data.get("audio_handler")
            if handler:
                try:
                    await handler.stop()
                    logger.info("Flushed audio buffer for meeting %s", meeting_id)
                except Exception:
                    logger.exception("Error flushing meeting %s", meeting_id)

        # Close Redis
        if self.redis_queue:
            try:
                await self.redis_queue.close()
                logger.info("✓ Redis closed")
            except Exception:
                logger.exception("Error closing Redis")

        # Close database
        if self.db_engine:
            try:
                await self.db_engine.dispose()
                logger.info("✓ Database engine disposed")
            except Exception:
                logger.exception("Error disposing database engine")

        logger.info("Application shutdown complete")


# Singleton state
app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    # Configure logging
    log_level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    await app_state.startup()
    yield
    await app_state.shutdown()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Intelligent Meeting Agent",
    description="Autonomous multilingual meeting transcription and analysis agent for Microsoft Teams",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — required for Azure Bot Service preflight requests
from starlette.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint for Docker and load balancer probes."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=settings.environment.value,
    )


# ---------------------------------------------------------------------------
# Meeting endpoints
# ---------------------------------------------------------------------------

@app.post("/meetings/join", tags=["meetings"])
async def join_meeting(request: MeetingJoinRequest) -> dict[str, str]:
    """
    Trigger the bot to join a specific Teams meeting.

    Initialises the AudioStreamHandler, AzureSpeechClient, and ChunkManager,
    then instructs the Teams bot to join the meeting.
    """
    import uuid

    meeting_id = request.meeting_id or uuid.uuid4().hex[:12]

    if meeting_id in app_state.active_meetings:
        raise HTTPException(
            status_code=409,
            detail=f"Already in meeting {meeting_id}",
        )

    logger.info("Joining meeting %s — url=%s", meeting_id, request.meeting_url)

    # Initialise pipeline components
    try:
        from bot.audio_stream import AudioStreamHandler
        from transcription.whisper_realtime import WhisperRealtimeTranscriber
        from transcription.chunk_manager import ChunkManager

        # Create components
        speech_client = WhisperRealtimeTranscriber() if request.enable_transcription else None
        chunk_manager = ChunkManager(
            meeting_id=meeting_id,
            blob_uploader=app_state.blob_storage,
            queue_publisher=app_state.redis_queue,
        ) if request.enable_whisper else None

        audio_handler = AudioStreamHandler(
            speech_client=speech_client,
            chunk_manager=chunk_manager,
        )

        # Start the pipeline
        if speech_client:
            await speech_client.start()
        await audio_handler.start()

        # Store in active meetings
        app_state.active_meetings[meeting_id] = {
            "status": MeetingStatus.ACTIVE,
            "audio_handler": audio_handler,
            "speech_client": speech_client,
            "chunk_manager": chunk_manager,
            "meeting_url": request.meeting_url,
            "transcript_segments": [],
        }

        logger.info("Meeting %s pipeline initialised and active", meeting_id)

    except Exception as e:
        logger.exception("Failed to join meeting %s", meeting_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialise meeting pipeline: {e}",
        )

    return {
        "meeting_id": meeting_id,
        "status": "joined",
        "message": f"Bot is joining meeting {meeting_id}",
    }


@app.get(
    "/meetings/{meeting_id}/status",
    response_model=MeetingStatusResponse,
    tags=["meetings"],
)
async def get_meeting_status(meeting_id: str) -> MeetingStatusResponse:
    """Get the live status of an active meeting."""
    meeting = app_state.active_meetings.get(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")

    audio_handler = meeting.get("audio_handler")
    return MeetingStatusResponse(
        meeting_id=meeting_id,
        status=meeting.get("status", MeetingStatus.ACTIVE),
        transcription_active=meeting.get("speech_client") is not None,
        chunks_processed=(
            meeting.get("chunk_manager").sequence_number
            if meeting.get("chunk_manager")
            else 0
        ),
        duration_seconds=(
            audio_handler.stats.elapsed_seconds if audio_handler else 0.0
        ),
    )


@app.get(
    "/meetings/{meeting_id}/transcript",
    response_model=TranscriptResponse,
    tags=["meetings"],
)
async def get_meeting_transcript(meeting_id: str) -> TranscriptResponse:
    """Get the current transcript for an active or completed meeting."""
    meeting = app_state.active_meetings.get(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")

    segments = meeting.get("transcript_segments", [])
    languages = list({s.get("language", "unknown") for s in segments})

    return TranscriptResponse(
        meeting_id=meeting_id,
        segments=[
            TranscriptSegmentResponse(
                text=s.get("text", ""),
                language=s.get("language", "unknown"),
                timestamp=s.get("timestamp", 0.0),
                confidence=s.get("confidence", 0.0),
                speaker_id=s.get("speaker_id"),
                source=s.get("source", "whisper"),
            )
            for s in segments
        ],
        total_segments=len(segments),
        languages_detected=languages,
    )


# ---------------------------------------------------------------------------
# Bot Framework SDK adapter (handles OAuth + reply routing automatically)
# ---------------------------------------------------------------------------

import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()  # ensure .env is loaded

from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import Activity

_BOT_APP_ID = _os.getenv("MICROSOFT_APP_ID", "")
_BOT_APP_PASSWORD = _os.getenv("MICROSOFT_APP_PASSWORD", "")
_BOT_TENANT_ID = _os.getenv("MICROSOFT_APP_TENANT_ID", "")
logger.info("Bot credentials loaded: app_id=%s tenant=%s", _BOT_APP_ID[:8] + "..." if _BOT_APP_ID else "EMPTY", _BOT_TENANT_ID[:8] + "..." if _BOT_TENANT_ID else "EMPTY")

_adapter_settings = BotFrameworkAdapterSettings(
    app_id=_BOT_APP_ID,
    app_password=_BOT_APP_PASSWORD,
    channel_auth_tenant=_BOT_TENANT_ID,
)
_adapter = BotFrameworkAdapter(_adapter_settings)

# Error handler — log errors so they are visible
async def _on_adapter_error(context: TurnContext, error: Exception):
    logger.exception("Bot adapter error: %s", error)
    try:
        await context.send_activity("Sorry, an internal error occurred.")
    except Exception:
        pass

_adapter.on_turn_error = _on_adapter_error

# Instantiate the bot (uses TeamsBot from bot/teams_bot.py)
from bot.teams_bot import TeamsBot
_bot = TeamsBot()


# ---------------------------------------------------------------------------
# Teams Bot Framework messaging endpoint
# ---------------------------------------------------------------------------

@app.post("/api/messages", tags=["webhook"])
async def bot_messages(request: Request) -> Response:
    """
    Receive Bot Framework activities from Azure Bot Service.

    This is the endpoint configured as 'Messaging endpoint' in
    the Azure Bot registration.
    """
    try:
        body = await request.json()
        activity = Activity().deserialize(body)
        auth_header = request.headers.get("Authorization", "")
        logger.info("Bot activity received: type=%s channel=%s", activity.type, activity.channel_id)

        response = await _adapter.process_activity(
            activity, auth_header, _bot.on_turn
        )
        if response:
            return Response(
                content=response.body,
                status_code=response.status,
                headers=response.headers,
            )
        return Response(status_code=201)

    except Exception as e:
        logger.exception("Bot activity error: %s", e)
        return Response(status_code=200)  # Always return 200 to Azure


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

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
        self.graph_client = None  # GraphCallsClient for joining meetings
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

        # Graph Communications client (for joining meeting calls)
        try:
            from bot.graph_calls import GraphCallsClient
            self.graph_client = GraphCallsClient()
            logger.info("✓ Graph Communications client created")
        except Exception:
            logger.exception("✗ Graph Communications client init failed")

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

        # Close Graph client
        if self.graph_client:
            try:
                await self.graph_client.close()
                logger.info("✓ Graph client closed")
            except Exception:
                logger.exception("Error closing Graph client")

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

    # Set tunnel URL for Graph callbacks (env var first, then auto-detect)
    if app_state.graph_client:
        import os as _os
        _tunnel_url = _os.environ.get("TUNNEL_URL", "").strip().rstrip("/")
        if _tunnel_url:
            app_state.graph_client.set_callback_url(_tunnel_url)
            logger.info("Using configured TUNNEL_URL for Graph callbacks: %s", _tunnel_url)
        else:
            try:
                import httpx as _httpx
                async with _httpx.AsyncClient() as _c:
                    _r = await _c.get("http://127.0.0.1:20242/quicktunnel", timeout=3)
                    if _r.status_code == 200:
                        _host = _r.json().get("hostname", "")
                        if _host:
                            app_state.graph_client.set_callback_url(f"https://{_host}")
                            logger.info("Auto-detected tunnel for Graph callbacks: https://%s", _host)
            except Exception:
                logger.info("No cloudflared tunnel detected — set Graph callback URL manually if needed")

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

@app.post("/meetings/demo", tags=["meetings"])
async def create_demo_meeting(meeting_id: str = "demo001") -> dict:
    """
    Create a mock meeting with pre-loaded transcript data for testing.
    Bypasses the audio pipeline — no model loading required.
    """
    import time

    if meeting_id in app_state.active_meetings:
        del app_state.active_meetings[meeting_id]  # reset if exists

    now = time.time()
    app_state.active_meetings[meeting_id] = {
        "status": MeetingStatus.ACTIVE,
        "audio_handler": None,
        "speech_client": None,
        "chunk_manager": None,
        "meeting_url": "https://teams.microsoft.com/demo",
        "transcript_segments": [
            {"text": "Bonjour tout le monde, commençons la réunion.", "language": "fr", "timestamp": now - 300, "confidence": 0.97, "speaker_id": "Speaker_1", "source": "whisper"},
            {"text": "Good morning everyone, let's start the meeting.", "language": "en", "timestamp": now - 290, "confidence": 0.98, "speaker_id": "Speaker_2", "source": "whisper"},
            {"text": "مرحباً، هل يمكننا مراجعة تقرير المبيعات؟", "language": "ar", "timestamp": now - 280, "confidence": 0.95, "speaker_id": "Speaker_3", "source": "whisper"},
            {"text": "The Q4 results show a 15% increase in revenue compared to last quarter.", "language": "en", "timestamp": now - 270, "confidence": 0.99, "speaker_id": "Speaker_2", "source": "whisper"},
            {"text": "Excellent, nous devons maintenant discuter du budget pour le prochain trimestre.", "language": "fr", "timestamp": now - 260, "confidence": 0.96, "speaker_id": "Speaker_1", "source": "whisper"},
            {"text": "I agree, the AI transcription feature has been very helpful for our team.", "language": "en", "timestamp": now - 250, "confidence": 0.98, "speaker_id": "Speaker_2", "source": "whisper"},
            {"text": "شكراً جزيلاً، سنتابع هذه النقاط في الاجتماع القادم.", "language": "ar", "timestamp": now - 240, "confidence": 0.94, "speaker_id": "Speaker_3", "source": "whisper"},
            {"text": "Let's schedule a follow-up meeting for next week.", "language": "en", "timestamp": now - 230, "confidence": 0.99, "speaker_id": "Speaker_2", "source": "whisper"},
        ],
    }
    return {
        "meeting_id": meeting_id,
        "status": "demo_active",
        "message": f"Demo meeting '{meeting_id}' created with {len(app_state.active_meetings[meeting_id]['transcript_segments'])} mock transcript segments (EN/FR/AR)",
    }


@app.post("/meetings/join", tags=["meetings"])
async def join_meeting(request: MeetingJoinRequest) -> dict[str, str]:
    """
    Trigger the bot to join a specific Teams meeting.

    1. Initialises the audio pipeline (AudioStreamHandler + Whisper + ChunkManager)
    2. Calls the Microsoft Graph Communications API to join the meeting call
    3. Audio will arrive via Graph media callbacks once the call is established
    """
    import uuid

    meeting_id = request.meeting_id or uuid.uuid4().hex[:12]

    if meeting_id in app_state.active_meetings:
        raise HTTPException(
            status_code=409,
            detail=f"Already in meeting {meeting_id}",
        )

    logger.info("Joining meeting %s — url=%s", meeting_id, request.meeting_url)

    # Step 1: Start the MeetingHandler pipeline
    try:
        session = await _meeting_handler.on_meeting_start(
            meeting_id=meeting_id,
            meeting_data={"meeting_url": request.meeting_url},
        )

        # Bridge to app_state so transcript/status endpoints work
        app_state.active_meetings[meeting_id] = {
            "status": MeetingStatus.ACTIVE,
            "audio_handler": session.audio_handler,
            "speech_client": session.speech_client,
            "chunk_manager": session.chunk_manager,
            "meeting_url": request.meeting_url,
            "transcript_segments": session.transcript_segments,  # shared ref
            "session": session,
        }

        logger.info("Meeting %s pipeline initialised", meeting_id)

    except Exception as e:
        logger.exception("Failed to initialise pipeline for %s", meeting_id)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline init failed: {e}",
        )

    # Step 2: Join via Graph Communications API
    call_session = None
    if app_state.graph_client and app_state.graph_client._callback_url:
        try:
            call_session = await app_state.graph_client.join_meeting(
                join_url=request.meeting_url,
                meeting_id=meeting_id,
            )
            app_state.active_meetings[meeting_id]["call_session"] = call_session
            logger.info(
                "Graph call created: call_id=%s for meeting %s",
                call_session.call_id,
                meeting_id,
            )
        except Exception as e:
            logger.warning(
                "Graph call join failed (pipeline still active): %s", e
            )
    else:
        logger.info(
            "Graph client not configured or no callback URL — "
            "pipeline running but bot won't join the call automatically. "
            "Meeting events from Teams will still trigger transcription."
        )

    return {
        "meeting_id": meeting_id,
        "status": "joined" if call_session else "pipeline_ready",
        "message": (
            f"Bot joined meeting call (call_id={call_session.call_id})"
            if call_session
            else f"Pipeline ready for meeting {meeting_id}. "
            f"Bot will transcribe when meeting events arrive via Teams."
        ),
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

    # Try MeetingHandler session first (live data), fall back to dict
    session = meeting.get("session")
    if session:
        segments = session.transcript_segments  # live reference
    else:
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


@app.post("/meetings/{meeting_id}/leave", tags=["meetings"])
async def leave_meeting(meeting_id: str) -> dict[str, str]:
    """Leave an active meeting — stop the pipeline and hang up the call."""
    meeting = app_state.active_meetings.get(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")

    # Stop pipeline via MeetingHandler
    await _meeting_handler.on_meeting_end(meeting_id=meeting_id, meeting_data={})

    # Hang up Graph call if active
    call_session = meeting.get("call_session")
    if call_session and app_state.graph_client:
        try:
            await app_state.graph_client.leave_meeting(call_session.call_id)
        except Exception:
            logger.exception("Error leaving Graph call %s", call_session.call_id)

    # Keep the meeting data for transcript access, mark as ended
    meeting["status"] = MeetingStatus.ENDED

    return {"meeting_id": meeting_id, "status": "left"}


# ---------------------------------------------------------------------------
# Graph Communications callback (receives call state + audio)
# ---------------------------------------------------------------------------

@app.post("/api/calls/callback", tags=["webhook"])
async def graph_calls_callback(request: Request) -> Response:
    """
    Handle callbacks from Microsoft Graph Communications API.

    Graph sends notifications here when:
    - Call state changes (establishing → established → terminated)
    - Audio data arrives (app-hosted media)
    - Participants join/leave the call
    """
    try:
        body = await request.json()
        logger.info("Graph calls callback received: %s", str(body)[:200])

        if app_state.graph_client:
            result = await app_state.graph_client.handle_callback(body)
            return Response(
                content='{"status":"ok"}',
                status_code=200,
                media_type="application/json",
            )
        else:
            logger.warning("Graph callback received but no graph_client configured")
            return Response(status_code=200)

    except Exception as e:
        logger.exception("Graph calls callback error: %s", e)
        return Response(status_code=200)


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

_is_dev = _os.getenv("ENVIRONMENT", "development") == "development"

# Always use REAL credentials for the adapter so that outgoing replies
# (send_activity) can authenticate to Azure Bot Service / Bot Connector.
# Incoming JWT validation is handled separately in the endpoint.
_adapter_settings = BotFrameworkAdapterSettings(
    app_id=_BOT_APP_ID,
    app_password=_BOT_APP_PASSWORD,
    # Must specify tenant for single-tenant app registrations so the
    # SDK fetches tokens from the correct Azure AD endpoint.
    channel_auth_tenant=_BOT_TENANT_ID or None,
    auth_configuration=None,
)
_adapter = BotFrameworkAdapter(_adapter_settings)

# Emulator adapter: NO credentials.  Single-tenant apps can't authenticate
# via botframework.com, and the emulator (configured without App ID/Password)
# does not require auth on either incoming activities or outgoing replies.
_emu_adapter_settings = BotFrameworkAdapterSettings(
    app_id="",
    app_password="",
    channel_auth_tenant=None,
    auth_configuration=None,
)
_emu_adapter = BotFrameworkAdapter(_emu_adapter_settings)

# Error handler — log errors so they are visible
async def _on_adapter_error(context: TurnContext, error: Exception):
    logger.exception("Bot adapter error: %s", error)
    try:
        import asyncio
        await asyncio.wait_for(
            context.send_activity("Sorry, an internal error occurred."),
            timeout=3.0,
        )
    except Exception:
        pass

_adapter.on_turn_error = _on_adapter_error
_emu_adapter.on_turn_error = _on_adapter_error

# Instantiate the MeetingHandler + TeamsBot with the handler wired in
from bot.meeting_handler import MeetingHandler
from bot.teams_bot import TeamsBot

_meeting_handler = MeetingHandler(
    blob_uploader=app_state.blob_storage,
    queue_publisher=app_state.redis_queue,
)

_bot = TeamsBot(
    meeting_handler=_meeting_handler,
    graph_client=app_state.graph_client,
    app_state=app_state,
)
logger.info("TeamsBot wired with MeetingHandler + GraphClient — bot will auto-join meetings")


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
        channel = (activity.channel_id or "").lower()
        logger.info(
            "Bot activity received: type=%s channel=%s from=%s serviceUrl=%s",
            activity.type, channel,
            getattr(activity.from_property, 'id', '?') if activity.from_property else '?',
            activity.service_url,
        )

        if channel == "emulator":
            # Emulator: use zero-credential adapter + blank auth so the SDK
            # skips JWT validation.  Outgoing replies go unauthenticated,
            # which the emulator (configured without App ID) accepts.
            logger.info("Using emulator adapter (no creds, no auth)")
            response = await _emu_adapter.process_activity(
                activity, "", _bot.on_turn
            )
        else:
            # Azure channels (webchat, msteams, etc.): use the real-
            # credential adapter and pass through the actual auth header
            # so the SDK can validate the incoming JWT and authenticate
            # outgoing replies.
            response = await _adapter.process_activity(
                activity, auth_header, _bot.on_turn
            )
        status = response.status if response else 201
        logger.info("Bot activity processed OK, returning HTTP %d", status)
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

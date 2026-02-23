"""
teams_bot.py — Bot Framework bot definition.

Handles incoming activities from Microsoft Teams via the Bot Framework SDK.
Routes meeting-related events to the MeetingHandler and manages the bot lifecycle.
Integrates with the audio pipeline for real-time meeting capture.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any
import uuid

from botbuilder.core import TurnContext
from botbuilder.core.teams import TeamsActivityHandler
from botbuilder.schema import (
    Activity,
    ActivityTypes,
    ChannelAccount,
)
from botbuilder.schema.teams import (
    MeetingStartEventDetails,
    MeetingEndEventDetails,
    MeetingParticipantsEventDetails,
)

from config.settings import settings

if TYPE_CHECKING:
    from bot.meeting_handler import MeetingHandler
    from bot.graph_calls import GraphCallsClient

logger = logging.getLogger(__name__)


class TeamsBot(TeamsActivityHandler):
    """
    Microsoft Teams bot that handles meeting-related activities.

    Routes meeting events (join, leave, participants changed) to the
    MeetingHandler, which manages the audio pipeline lifecycle.

    Attributes:
        meeting_handler: Injected handler for meeting lifecycle events.
    """

    def __init__(
        self,
        meeting_handler: MeetingHandler | None = None,
        graph_client: GraphCallsClient | None = None,
        app_state: Any | None = None,
    ) -> None:
        super().__init__()
        self._meeting_handler = meeting_handler
        self._graph_client_direct = graph_client  # may be None at init time
        self._app_state = app_state
        self._active_conversations: dict[str, dict[str, Any]] = {}

        logger.info("TeamsBot initialised (app_state=%s)", bool(app_state))

    @property
    def _graph_client(self):
        """Lazily resolve graph_client — it's set after startup()."""
        if self._graph_client_direct:
            return self._graph_client_direct
        if self._app_state:
            return getattr(self._app_state, "graph_client", None)
        return None

    # ------------------------------------------------------------------
    # Standard activity handlers
    # ------------------------------------------------------------------

    async def on_turn(self, turn_context: TurnContext) -> None:
        """
        Process every incoming activity before routing.

        Logs the activity type and delegates to the appropriate handler.
        """
        activity = turn_context.activity
        logger.info(
            "Activity received — type=%s, channel=%s, conversation=%s",
            activity.type,
            activity.channel_id,
            activity.conversation.id if activity.conversation else "none",
        )
        await super().on_turn(turn_context)

    # Regex to detect Teams meeting join URLs (both old and new formats)
    _JOIN_URL_RE = re.compile(
        r'(https://teams\.microsoft\.com/(?:l/meetup-join/|meet/)[^\s"<>]+)',
        re.IGNORECASE,
    )

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        """
        Handle direct messages to the bot.

        If the message contains a Teams meeting join URL, the bot joins
        that meeting via Graph API.  Otherwise responds with status/help.
        """
        text = turn_context.activity.text or ""
        text_lower = text.strip().lower()

        logger.info("Message received: %s", text[:200])

        # --- Check for a Teams meeting join URL -----------------------
        join_match = self._JOIN_URL_RE.search(text)
        if join_match:
            await self._handle_join_url(join_match.group(1), turn_context)
            return

        # --- Check for "join" keyword with a URL in channel_data ------
        if text_lower.startswith("join"):
            channel_data = getattr(turn_context.activity, "channel_data", None) or {}
            meeting_info = channel_data.get("meeting", {})
            join_url = meeting_info.get("joinUrl") or meeting_info.get("joinWebUrl")
            if join_url:
                await self._handle_join_url(join_url, turn_context)
                return

        # --- Standard commands ----------------------------------------
        if text_lower in ("status", "info"):
            await self._send_status(turn_context)
        elif text_lower in ("help", "?"):
            await self._send_help(turn_context)
        else:
            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text=(
                        "👋 I'm the Meeting Agent bot.\n\n"
                        "**To have me join a meeting**, paste the Teams meeting "
                        "join URL here (e.g. https://teams.microsoft.com/l/meetup-join/...).\n\n"
                        "**Commands:** `status`, `help`"
                    ),
                )
            )

    async def _handle_join_url(
        self, join_url: str, turn_context: TurnContext
    ) -> None:
        """Join a meeting when a user sends its join URL."""
        meeting_id = f"m-{uuid.uuid4().hex[:8]}"
        logger.info(
            "Join URL detected — meeting_id=%s, url=%s", meeting_id, join_url[:120]
        )

        if not self._graph_client:
            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text="⚠️ Graph client is not configured — cannot join meetings.",
                )
            )
            return

        if not self._graph_client._callback_url:
            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text="⚠️ No tunnel/callback URL — cannot join meetings.",
                )
            )
            return

        await turn_context.send_activity(
            Activity(
                type=ActivityTypes.message,
                text=f"🔄 Joining the meeting now…",
            )
        )

        try:
            call_session = await self._graph_client.join_meeting(
                join_url=join_url,
                meeting_id=meeting_id,
            )
            logger.info(
                "Joined meeting %s — call_id=%s", meeting_id, call_session.call_id
            )

            # Initialize meeting handler pipeline
            session = None
            if self._meeting_handler:
                session = await self._meeting_handler.on_meeting_start(
                    meeting_id=meeting_id,
                    meeting_data={"joinUrl": join_url},
                    turn_context=turn_context,
                )

            # Register in app_state
            if self._app_state:
                from api.main import MeetingStatus
                self._app_state.active_meetings[meeting_id] = {
                    "status": MeetingStatus.ACTIVE,
                    "meeting_url": join_url,
                    "session": session,
                    "call_session": call_session,
                    "audio_handler": getattr(session, "audio_handler", None),
                    "speech_client": getattr(session, "speech_client", None),
                    "chunk_manager": getattr(session, "chunk_manager", None),
                    "transcript_segments": getattr(session, "transcript_segments", []),
                }

            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text=(
                        f"✅ Successfully joined the meeting! (call_id: `{call_session.call_id}`)\n\n"
                        "🎙️ I'm now transcribing. I'll provide a summary when the meeting ends."
                    ),
                )
            )
        except Exception as e:
            logger.exception("Failed to join meeting via Graph: %s", e)
            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text=f"❌ Failed to join meeting: {e}",
                )
            )

    async def on_members_added_activity(
        self,
        members_added: list[ChannelAccount],
        turn_context: TurnContext,
    ) -> None:
        """Handle new members added to the conversation (including the bot)."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                logger.info("Member added: %s (%s)", member.name, member.id)
            else:
                logger.info("Bot added to conversation")
                await turn_context.send_activity(
                    Activity(
                        type=ActivityTypes.message,
                        text=(
                            "🤖 Meeting Agent is ready. "
                            "I will automatically transcribe meetings I'm invited to."
                        ),
                    )
                )

    async def on_members_removed_activity(
        self,
        members_removed: list[ChannelAccount],
        turn_context: TurnContext,
    ) -> None:
        """Handle members removed from the conversation."""
        for member in members_removed:
            logger.info("Member removed: %s (%s)", member.name, member.id)

    # ------------------------------------------------------------------
    # Teams-specific event handlers
    # ------------------------------------------------------------------

    async def on_event_activity(self, turn_context: TurnContext) -> None:
        """
        Handle Teams event activities.

        Logs the event, then delegates to TeamsActivityHandler which
        routes meeting-specific events to on_teams_meeting_start_event, etc.
        """
        activity = turn_context.activity
        event_name = activity.name or activity.type
        logger.info("Event activity: %s", event_name)

        # Let TeamsActivityHandler route meeting events to the correct handlers
        await super().on_event_activity(turn_context)

    async def on_teams_meeting_start_event(
        self,
        meeting: MeetingStartEventDetails,
        turn_context: TurnContext,
    ) -> None:
        """
        Called when a Teams meeting starts.

        1. Initialises the local audio pipeline via MeetingHandler
        2. Auto-joins the meeting call via Graph Communications API
        3. Registers the meeting in app_state for API tracking
        """
        meeting_dict = meeting.as_dict() if hasattr(meeting, "as_dict") else {}
        meeting_id = meeting_dict.get("id") or getattr(meeting, "id", None) or "unknown"
        logger.info("Teams meeting started: %s — data=%s", meeting_id, str(meeting_dict)[:300])

        # Extract the join URL from the Teams event data
        join_url = self._extract_join_url(meeting_dict, turn_context)
        logger.info("Extracted join URL for meeting %s: %s", meeting_id, join_url or "NONE")

        session = None
        if self._meeting_handler:
            session = await self._meeting_handler.on_meeting_start(
                meeting_id=meeting_id,
                meeting_data=meeting_dict,
                turn_context=turn_context,
            )

        # Auto-join the meeting call via Graph API
        call_session = None
        if self._graph_client and self._graph_client._callback_url and join_url:
            try:
                call_session = await self._graph_client.join_meeting(
                    join_url=join_url,
                    meeting_id=meeting_id,
                )
                logger.info(
                    "Auto-joined meeting %s via Graph — call_id=%s",
                    meeting_id, call_session.call_id,
                )
            except Exception as e:
                logger.exception(
                    "Failed to auto-join meeting %s via Graph: %s", meeting_id, e
                )
        elif not join_url:
            logger.warning(
                "Cannot auto-join meeting %s — no join URL found in event data",
                meeting_id,
            )

        # Register in app_state so API endpoints can track it
        if self._app_state and session:
            from api.main import MeetingStatus
            self._app_state.active_meetings[meeting_id] = {
                "status": MeetingStatus.ACTIVE,
                "audio_handler": session.audio_handler,
                "speech_client": session.speech_client,
                "chunk_manager": session.chunk_manager,
                "meeting_url": join_url or "",
                "transcript_segments": session.transcript_segments,
                "session": session,
                "call_session": call_session,
            }
            logger.info("Meeting %s registered in app_state for API tracking", meeting_id)

        # Notify the chat that the bot is transcribing
        try:
            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text=(
                        "🎙️ Meeting Agent has joined and is now transcribing. "
                        "I'll provide a summary when the meeting ends."
                    ),
                )
            )
        except Exception:
            logger.debug("Could not send join confirmation message")

    async def on_teams_meeting_end_event(
        self,
        meeting: MeetingEndEventDetails,
        turn_context: TurnContext,
    ) -> None:
        """
        Called when a Teams meeting ends.

        Flushes buffers, stops the pipeline, and leaves the Graph call.
        """
        meeting_dict = meeting.as_dict() if hasattr(meeting, "as_dict") else {}
        meeting_id = meeting_dict.get("id") or getattr(meeting, "id", None) or "unknown"
        logger.info("Teams meeting ended: %s", meeting_id)

        if self._meeting_handler:
            await self._meeting_handler.on_meeting_end(
                meeting_id=meeting_id,
                meeting_data=meeting_dict,
                turn_context=turn_context,
            )

        # Leave the Graph call if active
        if self._app_state:
            meeting_data = self._app_state.active_meetings.get(meeting_id, {})
            call_session = meeting_data.get("call_session")
            if call_session and self._graph_client:
                try:
                    await self._graph_client.leave_meeting(call_session.call_id)
                    logger.info("Left Graph call %s for meeting %s", call_session.call_id, meeting_id)
                except Exception:
                    logger.exception("Error leaving Graph call for meeting %s", meeting_id)

            # Mark meeting as ended (keep data for transcript access)
            if meeting_id in self._app_state.active_meetings:
                from api.main import MeetingStatus
                self._app_state.active_meetings[meeting_id]["status"] = MeetingStatus.ENDED

        # Notify the chat
        try:
            session = self._meeting_handler.get_session(meeting_id) if self._meeting_handler else None
            seg_count = len(session.transcript_segments) if session else 0
            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text=(
                        f"📝 Meeting ended — captured {seg_count} transcript segments. "
                        f"Use the API to retrieve the full transcript."
                    ),
                )
            )
        except Exception:
            logger.debug("Could not send meeting-end summary message")

    async def on_teams_meeting_participants_join_event(
        self,
        meeting: MeetingParticipantsEventDetails,
        turn_context: TurnContext,
    ) -> None:
        """Handle participants joining the meeting."""
        meeting_dict = meeting.as_dict() if hasattr(meeting, "as_dict") else {}
        members = meeting_dict.get("members", [])
        logger.info(
            "Participants joined: %s",
            [m.get("name", m.get("id", "?")) if isinstance(m, dict) else str(m) for m in members],
        )

        if self._meeting_handler:
            await self._meeting_handler.on_participants_join(
                participants=members,
                turn_context=turn_context,
            )

    async def on_teams_meeting_participants_leave_event(
        self,
        meeting: MeetingParticipantsEventDetails,
        turn_context: TurnContext,
    ) -> None:
        """Handle participants leaving the meeting."""
        meeting_dict = meeting.as_dict() if hasattr(meeting, "as_dict") else {}
        members = meeting_dict.get("members", [])
        logger.info(
            "Participants left: %s",
            [m.get("name", m.get("id", "?")) if isinstance(m, dict) else str(m) for m in members],
        )

        if self._meeting_handler:
            await self._meeting_handler.on_participants_leave(
                participants=members,
                turn_context=turn_context,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_status(self, turn_context: TurnContext) -> None:
        """Send current bot status to the user."""
        if self._meeting_handler:
            active = self._meeting_handler.active_meeting_count
            status_text = (
                f"📊 **Meeting Agent Status**\n\n"
                f"- Active meetings: {active}\n"
                f"- Environment: {settings.environment.value}\n"
            )
        else:
            status_text = "📊 Meeting Agent is running (no meeting handler configured)"

        await turn_context.send_activity(
            Activity(type=ActivityTypes.message, text=status_text)
        )

    async def _send_help(self, turn_context: TurnContext) -> None:
        """Send help text to the user."""
        help_text = (
            "🤖 **Meeting Agent Help**\n\n"
            "I join Teams meetings to provide:\n"
            "- Real-time multilingual transcription\n"
            "- Speaker identification\n"
            "- Post-meeting analytical reports\n\n"
            "**How to use:**\n"
            "1. Start or join a Teams meeting\n"
            "2. Copy the meeting join URL\n"
            "3. Paste it here — I'll join automatically!\n\n"
            "**Commands:**\n"
            "- Paste a meeting URL — Join that meeting\n"
            "- `status` — Show bot status\n"
            "- `help` — Show this message\n"
        )
        await turn_context.send_activity(
            Activity(type=ActivityTypes.message, text=help_text)
        )

    # ------------------------------------------------------------------
    # Join URL extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_join_url(
        meeting: dict[str, Any],
        turn_context: TurnContext,
    ) -> str | None:
        """
        Extract the Teams meeting join URL from the event data.

        The join URL can appear in several places depending on the
        event type and Teams SDK version:
        1. meeting.joinUrl
        2. meeting.joinWebUrl
        3. activity.channel_data.meeting.joinUrl
        4. activity.value.joinUrl
        5. Construct from activity.channel_data.meeting.id
        """
        # Direct fields on the meeting dict
        for key in ("joinUrl", "joinWebUrl", "JoinUrl", "JoinWebUrl"):
            url = meeting.get(key)
            if url:
                return url

        # Channel data
        activity = turn_context.activity
        channel_data = getattr(activity, "channel_data", None) or {}

        # meeting object inside channel_data
        cd_meeting = channel_data.get("meeting", {})
        for key in ("joinUrl", "joinWebUrl", "JoinUrl", "JoinWebUrl"):
            url = cd_meeting.get(key)
            if url:
                return url

        # activity.value
        value = getattr(activity, "value", None) or {}
        if isinstance(value, dict):
            for key in ("joinUrl", "joinWebUrl", "JoinUrl", "JoinWebUrl"):
                url = value.get(key)
                if url:
                    return url

        # Try constructing from meeting ID (thread ID)
        thread_id = cd_meeting.get("id")
        if thread_id:
            # Standard Teams join URL pattern
            tenant_id = channel_data.get("tenant", {}).get("id", "")
            if tenant_id:
                return (
                    f"https://teams.microsoft.com/l/meetup-join/"
                    f"{thread_id}/0?context=%7B%22Tid%22%3A%22{tenant_id}%22%7D"
                )

        return None


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("✓ TeamsBot module imports successfully")
    bot = TeamsBot()
    print(f"  Bot type: {type(bot).__name__}")
    print("  (Full testing requires Bot Framework test adapter)")

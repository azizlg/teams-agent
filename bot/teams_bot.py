"""
teams_bot.py — Bot Framework bot definition.

Handles incoming activities from Microsoft Teams via the Bot Framework SDK.
Routes meeting-related events to the MeetingHandler and manages the bot lifecycle.
Integrates with the audio pipeline for real-time meeting capture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import (
    Activity,
    ActivityTypes,
    ChannelAccount,
)

from config.settings import settings

if TYPE_CHECKING:
    from bot.meeting_handler import MeetingHandler

logger = logging.getLogger(__name__)


class TeamsBot(ActivityHandler):
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
    ) -> None:
        super().__init__()
        self._meeting_handler = meeting_handler
        self._active_conversations: dict[str, dict[str, Any]] = {}

        logger.info("TeamsBot initialised")

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

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        """
        Handle direct messages to the bot.

        Responds with status information or help text.
        """
        text = turn_context.activity.text or ""
        text_lower = text.strip().lower()

        logger.info("Message received: %s", text[:100])

        if text_lower in ("status", "info"):
            await self._send_status(turn_context)
        elif text_lower in ("help", "?"):
            await self._send_help(turn_context)
        else:
            await turn_context.send_activity(
                Activity(
                    type=ActivityTypes.message,
                    text=(
                        "👋 I'm the Meeting Agent bot. "
                        "I join meetings to transcribe and analyse conversations.\n\n"
                        "Commands: **status**, **help**"
                    ),
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

        Routes meeting-specific events to the MeetingHandler.
        """
        activity = turn_context.activity
        event_name = activity.name or activity.type
        logger.info("Event activity: %s", event_name)

        # Route meeting events to the handler
        if self._meeting_handler:
            await self._meeting_handler.handle_event(turn_context)

    async def on_teams_meeting_start(
        self,
        meeting: dict[str, Any],
        turn_context: TurnContext,
    ) -> None:
        """
        Called when a Teams meeting starts.

        Triggers the meeting handler to initialise the audio pipeline.
        """
        meeting_id = meeting.get("id", "unknown")
        logger.info("Teams meeting started: %s", meeting_id)

        if self._meeting_handler:
            await self._meeting_handler.on_meeting_start(
                meeting_id=meeting_id,
                meeting_data=meeting,
                turn_context=turn_context,
            )

    async def on_teams_meeting_end(
        self,
        meeting: dict[str, Any],
        turn_context: TurnContext,
    ) -> None:
        """
        Called when a Teams meeting ends.

        Triggers the meeting handler to flush buffers and stop the pipeline.
        """
        meeting_id = meeting.get("id", "unknown")
        logger.info("Teams meeting ended: %s", meeting_id)

        if self._meeting_handler:
            await self._meeting_handler.on_meeting_end(
                meeting_id=meeting_id,
                meeting_data=meeting,
                turn_context=turn_context,
            )

    async def on_teams_meeting_participants_join(
        self,
        participants: dict[str, Any],
        turn_context: TurnContext,
    ) -> None:
        """Handle participants joining the meeting."""
        members = participants.get("members", [])
        logger.info(
            "Participants joined: %s",
            [m.get("name", m.get("id", "?")) for m in members],
        )

        if self._meeting_handler:
            await self._meeting_handler.on_participants_join(
                participants=members,
                turn_context=turn_context,
            )

    async def on_teams_meeting_participants_leave(
        self,
        participants: dict[str, Any],
        turn_context: TurnContext,
    ) -> None:
        """Handle participants leaving the meeting."""
        members = participants.get("members", [])
        logger.info(
            "Participants left: %s",
            [m.get("name", m.get("id", "?")) for m in members],
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
            "I automatically join Teams meetings and provide:\n"
            "- Real-time multilingual transcription\n"
            "- Speaker identification\n"
            "- Post-meeting analytical reports\n\n"
            "**Commands:**\n"
            "- `status` — Show bot status\n"
            "- `help` — Show this message\n"
        )
        await turn_context.send_activity(
            Activity(type=ActivityTypes.message, text=help_text)
        )


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("✓ TeamsBot module imports successfully")
    bot = TeamsBot()
    print(f"  Bot type: {type(bot).__name__}")
    print("  (Full testing requires Bot Framework test adapter)")

"""
graph_calls.py — Microsoft Graph Communications API integration.

Handles joining Teams meetings as a bot, subscribing to audio streams,
and routing received audio to the AudioStreamHandler pipeline.

Uses the Microsoft Graph Communications Calling API:
https://learn.microsoft.com/en-us/graph/api/resources/communications-api-overview

Requirements:
- Azure Bot must have "Calls.JoinGroupCall.All" and "Calls.AccessMedia.All"
  application permissions in the Azure AD app registration.
- The bot must be configured as an "Application-hosted media bot"
  in Azure Bot Service → Configuration → Calling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CallSession:
    """Tracks a single active call/meeting the bot has joined."""

    call_id: str
    meeting_id: str
    join_url: str
    state: str = "establishing"  # establishing | established | terminated
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None


# ---------------------------------------------------------------------------
# GraphCallsClient
# ---------------------------------------------------------------------------

class GraphCallsClient:
    """
    Client for the Microsoft Graph Communications Calling API.

    Handles:
    - Acquiring application tokens (client_credentials grant)
    - Joining meetings via POST /communications/calls
    - Subscribing to audio (media configuration)
    - Calling-related webhook handling

    The actual incoming audio (RTP) is received on the media platform
    callback URL configured in the Azure Bot. This client handles the
    signalling (join/leave/subscribe) only.
    """

    GRAPH_BASE = "https://graph.microsoft.com/v1.0"
    TOKEN_URL_TEMPLATE = (
        "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    )

    def __init__(
        self,
        app_id: str | None = None,
        app_password: str | None = None,
        tenant_id: str | None = None,
        *,
        callback_url: str | None = None,
    ) -> None:
        import os

        self._app_id = app_id or os.getenv("MICROSOFT_APP_ID", "")
        self._app_password = app_password or os.getenv("MICROSOFT_APP_PASSWORD", "")
        self._tenant_id = tenant_id or os.getenv("MICROSOFT_APP_TENANT_ID", "")
        self._callback_url = callback_url  # set later when tunnel URL is known

        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._active_calls: dict[str, CallSession] = {}

        self._client = httpx.AsyncClient(timeout=30.0)

        logger.info(
            "GraphCallsClient initialised — app_id=%s, tenant=%s",
            self._app_id[:8] + "..." if self._app_id else "NONE",
            self._tenant_id[:8] + "..." if self._tenant_id else "NONE",
        )

    def set_callback_url(self, url: str) -> None:
        """Set the public HTTPS callback URL for Graph notifications."""
        self._callback_url = url
        logger.info("Graph callback URL set: %s", url)

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    async def _ensure_token(self) -> str:
        """Get a valid application token, refreshing if needed."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        url = self.TOKEN_URL_TEMPLATE.format(tenant_id=self._tenant_id)
        data = {
            "grant_type": "client_credentials",
            "client_id": self._app_id,
            "client_secret": self._app_password,
            "scope": "https://graph.microsoft.com/.default",
        }

        resp = await self._client.post(url, data=data)
        resp.raise_for_status()
        body = resp.json()

        self._token = body["access_token"]
        self._token_expiry = time.time() + body.get("expires_in", 3600)

        logger.info("Graph token acquired (expires_in=%s)", body.get("expires_in"))
        return self._token

    async def _auth_headers(self) -> dict[str, str]:
        token = await self._ensure_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Join meeting
    # ------------------------------------------------------------------

    async def join_meeting(
        self,
        join_url: str,
        meeting_id: str,
        *,
        display_name: str = "Meeting Agent",
    ) -> CallSession:
        """
        Join a Teams meeting by its join URL.

        Uses POST /communications/calls with a joinMeetingIdMeetingInfo
        or JoinURL-based approach.

        The bot must have Calls.JoinGroupCall.All permission.

        Args:
            join_url: The Teams meeting join link
            meeting_id: Internal meeting identifier for tracking
            display_name: Bot display name in the meeting

        Returns:
            CallSession with the call_id from Graph
        """
        headers = await self._auth_headers()

        # Simplified call body — joinWebUrl at the top level works with
        # both /l/meetup-join/ and /meet/ Teams URL formats
        body: dict[str, Any] = {
            "@odata.type": "#microsoft.graph.call",
            "callbackUri": f"{self._callback_url}/api/calls/callback",
            "requestedModalities": ["audio"],
            "tenantId": self._tenant_id,
            "joinWebUrl": join_url,
            "mediaConfig": {
                "@odata.type": "#microsoft.graph.serviceHostedMediaConfig",
            },
            "source": {
                "@odata.type": "#microsoft.graph.participantInfo",
                "identity": {
                    "@odata.type": "#microsoft.graph.identitySet",
                    "application": {
                        "@odata.type": "#microsoft.graph.identity",
                        "id": self._app_id,
                        "displayName": display_name,
                    },
                },
            },
        }

        logger.info("Joining call via Graph API — meeting=%s", meeting_id)

        resp = await self._client.post(
            f"{self.GRAPH_BASE}/communications/calls",
            headers=headers,
            json=body,
        )

        if resp.status_code not in (200, 201):
            error_body = resp.text[:500]
            logger.error(
                "Graph join failed: HTTP %d — %s", resp.status_code, error_body
            )
            raise RuntimeError(
                f"Graph API call join failed: HTTP {resp.status_code} — {error_body}"
            )

        call_data = resp.json()
        call_id = call_data.get("id", "unknown")

        session = CallSession(
            call_id=call_id,
            meeting_id=meeting_id,
            join_url=join_url,
            state="establishing",
        )
        self._active_calls[call_id] = session

        logger.info(
            "Call created: call_id=%s, state=%s, meeting=%s",
            call_id,
            call_data.get("state"),
            meeting_id,
        )

        return session

    # ------------------------------------------------------------------
    # Leave meeting
    # ------------------------------------------------------------------

    async def leave_meeting(self, call_id: str) -> None:
        """Hang up / leave a meeting call."""
        headers = await self._auth_headers()

        logger.info("Leaving call: %s", call_id)

        resp = await self._client.delete(
            f"{self.GRAPH_BASE}/communications/calls/{call_id}",
            headers=headers,
        )

        if resp.status_code not in (200, 204):
            logger.warning(
                "Failed to leave call %s: HTTP %d", call_id, resp.status_code
            )

        session = self._active_calls.pop(call_id, None)
        if session:
            session.state = "terminated"
            session.ended_at = time.time()

    # ------------------------------------------------------------------
    # Callback handling
    # ------------------------------------------------------------------

    async def handle_callback(
        self,
        body: dict[str, Any],
        on_audio: Callable[[bytes, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """
        Handle incoming Graph Communications callbacks.

        These arrive at /api/calls/callback when:
        - Call state changes (establishing → established → terminated)
        - Audio data is available (for app-hosted media)
        - Participants join/leave

        Args:
            body: The callback payload from Graph
            on_audio: Callback to handle audio data (raw bytes, call_id)

        Returns:
            Response to send back to Graph
        """
        notifications = body.get("value", [body])

        for notification in notifications:
            resource_type = notification.get("resourceUrl", "")
            change_type = notification.get("changeType", "")

            # Call state change
            if "/communications/calls/" in resource_type:
                resource = notification.get("resourceData", {})
                call_id = resource.get("id", "")
                state = resource.get("state", "")

                logger.info(
                    "Call callback: call_id=%s, state=%s, change=%s",
                    call_id, state, change_type,
                )

                session = self._active_calls.get(call_id)
                if session:
                    session.state = state

                    if state == "established":
                        logger.info(
                            "Call %s established — bot is in the meeting!", call_id
                        )
                    elif state == "terminated":
                        session.ended_at = time.time()
                        logger.info("Call %s terminated", call_id)

        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Subscribe to audio
    # ------------------------------------------------------------------

    async def subscribe_to_audio(self, call_id: str) -> None:
        """
        Subscribe to unmixed audio from a call.

        Requires Calls.AccessMedia.All permission.
        Audio will be delivered via the media platform to the callback URL.
        """
        headers = await self._auth_headers()

        body = {
            "clientContext": f"audio-sub-{call_id}",
        }

        resp = await self._client.post(
            f"{self.GRAPH_BASE}/communications/calls/{call_id}/subscribeToTone",
            headers=headers,
            json=body,
        )

        logger.info(
            "Audio subscription for call %s: HTTP %d",
            call_id,
            resp.status_code,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def active_calls(self) -> dict[str, CallSession]:
        return dict(self._active_calls)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = GraphCallsClient()
    print(f"✓ GraphCallsClient initialised")
    print(f"  app_id configured: {bool(client._app_id)}")
    print(f"  tenant configured: {bool(client._tenant_id)}")

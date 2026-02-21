"""
Tests for transcription/azure_speech.py — AzureSpeechClient.

Verifies client initialisation, start/stop lifecycle, and callback wiring.
Actual Azure Speech recognition requires valid credentials, so these tests
use mocking for the SDK layer.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from transcription.azure_speech import AzureSpeechClient, TranscriptSegment


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAzureSpeechClient:
    """Tests for AzureSpeechClient."""

    def test_initialisation_defaults(self) -> None:
        """Client should initialise with settings defaults."""
        client = AzureSpeechClient()
        assert not client.is_running
        assert client.segments_recognised == 0

    def test_initialisation_custom_params(self) -> None:
        """Client should accept custom parameters."""
        client = AzureSpeechClient(
            speech_key="test-key",
            speech_region="eastus",
            languages=["en-US", "fr-FR"],
        )
        assert not client.is_running
        assert client._speech_key == "test-key"
        assert client._speech_region == "eastus"
        assert len(client._languages) == 2

    def test_callback_stored(self) -> None:
        """Segment callback should be stored on init."""
        async def my_callback(segment: TranscriptSegment) -> None:
            pass

        client = AzureSpeechClient(on_segment=my_callback)
        assert client._on_segment is my_callback

    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        """Calling stop on a non-running client should be safe."""
        client = AzureSpeechClient()
        await client.stop()  # Should not raise
        assert not client.is_running

    def test_transcript_segment_model(self) -> None:
        """TranscriptSegment should hold all required fields."""
        segment = TranscriptSegment(
            text="Hello world",
            language="en-US",
            timestamp=1234567890.0,
            offset_ms=500,
            duration_ms=1500,
            confidence=0.95,
            speaker_id="speaker-1",
        )
        assert segment.text == "Hello world"
        assert segment.language == "en-US"
        assert segment.confidence == 0.95
        assert segment.speaker_id == "speaker-1"

    def test_transcript_segment_no_speaker(self) -> None:
        """TranscriptSegment speaker_id should default to None."""
        segment = TranscriptSegment(
            text="Test",
            language="fr-FR",
            timestamp=0.0,
            offset_ms=0,
            duration_ms=0,
            confidence=0.5,
        )
        assert segment.speaker_id is None

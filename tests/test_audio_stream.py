"""
Tests for bot/audio_stream.py — AudioStreamHandler.

Verifies frame routing, buffer append, flush trigger, and statistics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
import pytest_asyncio

from bot.audio_stream import AudioFrame, AudioStreamHandler, StreamStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeFrame:
    """Minimal frame for testing the AudioConsumer protocol."""
    data: bytes
    timestamp: float = 0.0
    sequence_number: int = 0


class MockConsumer:
    """Records every frame it receives."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.frames: list = []
        self._fail_after = fail_after

    async def receive_frame(self, frame) -> None:
        self.frames.append(frame)
        if self._fail_after is not None and len(self.frames) >= self._fail_after:
            raise RuntimeError("Simulated consumer error")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAudioStreamHandler:
    """Tests for AudioStreamHandler."""

    @pytest.mark.asyncio
    async def test_frames_routed_to_both_consumers(self) -> None:
        """Frames should be dispatched to speech and buffer consumers."""
        speech = MockConsumer()
        buffer = MockConsumer()
        handler = AudioStreamHandler(speech_client=speech, chunk_manager=buffer)

        await handler.start()
        for _ in range(5):
            await handler.receive_audio_frame(b"\x00" * 640)

        # Give tasks time to complete
        await asyncio.sleep(0.1)
        await handler.stop()

        assert len(speech.frames) == 5
        assert len(buffer.frames) == 5

    @pytest.mark.asyncio
    async def test_stats_updated_correctly(self) -> None:
        """Stream stats should track frames and bytes."""
        handler = AudioStreamHandler()
        await handler.start()

        frame_data = b"\x00" * 640
        for _ in range(10):
            await handler.receive_audio_frame(frame_data)

        assert handler.stats.frames_received == 10
        assert handler.stats.bytes_received == 6400

        await handler.stop()

    @pytest.mark.asyncio
    async def test_handler_ignores_frames_when_stopped(self) -> None:
        """Frames received while stopped should be ignored."""
        speech = MockConsumer()
        handler = AudioStreamHandler(speech_client=speech)

        # Don't call start()
        await handler.receive_audio_frame(b"\x00" * 640)
        await asyncio.sleep(0.05)

        assert len(speech.frames) == 0
        assert handler.stats.frames_received == 0

    @pytest.mark.asyncio
    async def test_consumer_error_does_not_crash_handler(self) -> None:
        """Errors in consumers should be logged, not propagated."""
        speech = MockConsumer(fail_after=3)
        buffer = MockConsumer()
        handler = AudioStreamHandler(speech_client=speech, chunk_manager=buffer)

        await handler.start()
        for _ in range(5):
            await handler.receive_audio_frame(b"\x00" * 640)

        await asyncio.sleep(0.1)
        await handler.stop()

        # Buffer should still have all frames
        assert len(buffer.frames) == 5
        assert handler.stats.errors >= 1

    @pytest.mark.asyncio
    async def test_no_consumers_still_works(self) -> None:
        """Handler should work with no consumers configured."""
        handler = AudioStreamHandler()
        await handler.start()

        for _ in range(3):
            await handler.receive_audio_frame(b"\x00" * 640)

        await handler.stop()
        assert handler.stats.frames_received == 3

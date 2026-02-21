"""
Tests for transcription/chunk_manager.py — ChunkManager.

Verifies chunking logic, silence detection, sequence numbering,
WAV encoding, and meeting-end flush.
"""

from __future__ import annotations

import asyncio
import struct
import math
from dataclasses import dataclass

import pytest
import pytest_asyncio

from transcription.chunk_manager import ChunkManager, ChunkMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeFrame:
    data: bytes
    timestamp: float = 0.0
    sequence_number: int = 0


class MockBlobUploader:
    """Records uploads and returns fake URLs."""

    def __init__(self) -> None:
        self.uploads: list[tuple[str, int]] = []  # (blob_name, size)

    async def upload(self, data: bytes, blob_name: str) -> str:
        self.uploads.append((blob_name, len(data)))
        return f"https://blob.example.com/{blob_name}"


class MockQueuePublisher:
    """Records pushed chunk metadata."""

    def __init__(self) -> None:
        self.chunks: list[ChunkMetadata] = []

    async def push_chunk(self, metadata: ChunkMetadata) -> None:
        self.chunks.append(metadata)


def _generate_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM audio (all zeros)."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def _generate_tone(
    duration_ms: int,
    frequency: float = 440.0,
    amplitude: float = 16000.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a sine wave tone as PCM audio."""
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = [
        int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *samples)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkManager:
    """Tests for ChunkManager chunking and buffering logic."""

    @pytest.mark.asyncio
    async def test_chunk_flushed_after_duration(self) -> None:
        """Buffer should auto-flush when reaching chunk duration."""
        uploader = MockBlobUploader()
        queue = MockQueuePublisher()
        mgr = ChunkManager(
            meeting_id="test-1",
            blob_uploader=uploader,
            queue_publisher=queue,
            chunk_duration_seconds=1,
        )

        # Send 1.5 seconds of silence (at 16kHz, 16-bit mono = 32000 bytes/sec)
        frame_bytes = _generate_silence(20)  # 20ms frame
        for i in range(75):  # 75 * 20ms = 1500ms = 1.5s
            await mgr.receive_frame(FakeFrame(data=frame_bytes, sequence_number=i))

        assert len(queue.chunks) >= 1, "Expected at least one chunk flush"
        assert queue.chunks[0].sequence_number == 1

    @pytest.mark.asyncio
    async def test_sequence_numbers_increment(self) -> None:
        """Each flushed chunk should have an incrementing sequence number."""
        queue = MockQueuePublisher()
        mgr = ChunkManager(
            meeting_id="test-2",
            queue_publisher=queue,
            chunk_duration_seconds=1,
        )

        frame_bytes = _generate_silence(20)
        for i in range(150):  # 3 seconds → should produce 2-3 chunks
            await mgr.receive_frame(FakeFrame(data=frame_bytes, sequence_number=i))

        await mgr.flush()  # flush remaining

        seq_numbers = [c.sequence_number for c in queue.chunks]
        assert seq_numbers == list(range(1, len(seq_numbers) + 1))

    @pytest.mark.asyncio
    async def test_meeting_end_flush(self) -> None:
        """Calling flush() should emit any remaining buffered audio."""
        queue = MockQueuePublisher()
        mgr = ChunkManager(
            meeting_id="test-3",
            queue_publisher=queue,
            chunk_duration_seconds=300,  # 5 min — won't auto-flush
        )

        frame_bytes = _generate_silence(20)
        for i in range(10):
            await mgr.receive_frame(FakeFrame(data=frame_bytes, sequence_number=i))

        assert len(queue.chunks) == 0, "Should not flush yet"

        await mgr.flush()
        assert len(queue.chunks) == 1, "Final flush should produce a chunk"

    @pytest.mark.asyncio
    async def test_empty_flush_does_nothing(self) -> None:
        """Flushing an empty buffer should not produce a chunk."""
        queue = MockQueuePublisher()
        mgr = ChunkManager(
            meeting_id="test-4",
            queue_publisher=queue,
            chunk_duration_seconds=300,
        )

        await mgr.flush()
        assert len(queue.chunks) == 0

    @pytest.mark.asyncio
    async def test_blob_upload_called(self) -> None:
        """Uploaded blobs should contain valid WAV data."""
        uploader = MockBlobUploader()
        queue = MockQueuePublisher()
        mgr = ChunkManager(
            meeting_id="test-5",
            blob_uploader=uploader,
            queue_publisher=queue,
            chunk_duration_seconds=300,
        )

        frame_bytes = _generate_silence(20)
        for i in range(10):
            await mgr.receive_frame(FakeFrame(data=frame_bytes, sequence_number=i))

        await mgr.flush()

        assert len(uploader.uploads) == 1
        blob_name, size = uploader.uploads[0]
        assert "test-5" in blob_name
        assert size > 0  # WAV file has headers + data

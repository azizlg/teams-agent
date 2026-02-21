"""
redis_queue.py — Redis Streams queue.

RedisQueue uses Redis Streams with consumer groups for reliable,
horizontally-scalable delivery of audio chunk metadata to Whisper workers.
Supports ACK-based processing, dead-letter handling (3 retries), and
consumer group auto-creation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

import redis.asyncio as aioredis

from config.settings import settings

logger = logging.getLogger(__name__)


# Type alias for the processing callback
# Returns True if processing succeeded (message will be ACK'd)
ChunkCallback = Callable[[dict], Awaitable[bool]]


class RedisQueue:
    """
    Redis Streams queue for audio chunk metadata.

    Features:
    - Push chunk metadata to a Redis Stream
    - Consumer group pattern for horizontal scaling
    - ACK only after successful processing
    - Dead letter handling: retry up to ``max_retries`` then log and move on
    - Automatic consumer group creation
    """

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        stream_name: str | None = None,
        consumer_group: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._redis_url = redis_url or settings.redis.url
        self._stream_name = stream_name or settings.redis.stream_name
        self._consumer_group = consumer_group or settings.redis.consumer_group
        self._max_retries = max_retries or settings.redis.max_retries

        self._client: aioredis.Redis | None = None
        self._consumer_name: str = f"worker-{id(self)}"
        self._running: bool = False

        logger.info(
            "RedisQueue initialised — stream=%s, group=%s, max_retries=%d",
            self._stream_name,
            self._consumer_group,
            self._max_retries,
        )

    # ---------- connection ----------

    async def connect(self) -> None:
        """Connect to Redis and ensure the consumer group exists."""
        self._client = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
        )

        # Verify connection
        await self._client.ping()
        logger.info("Connected to Redis at %s", self._redis_url)

        # Create consumer group (ignore error if it already exists)
        try:
            await self._client.xgroup_create(
                self._stream_name,
                self._consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                "Created consumer group '%s' on stream '%s'",
                self._consumer_group,
                self._stream_name,
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group '%s' already exists", self._consumer_group)
            else:
                raise

    async def close(self) -> None:
        """Close the Redis connection."""
        self._running = False
        if self._client:
            await self._client.aclose()
            logger.info("Redis connection closed")

    # ---------- producer ----------

    async def push_chunk(self, metadata) -> str:
        """
        Push chunk metadata to the stream.

        Args:
            metadata: A ChunkMetadata pydantic model or dict.

        Returns:
            The Redis message ID.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis — call connect() first")

        # Convert pydantic model to dict if needed
        if hasattr(metadata, "model_dump"):
            data = metadata.model_dump()
        elif isinstance(metadata, dict):
            data = metadata
        else:
            data = dict(metadata)

        # Redis Streams requires string values — serialise complex values
        flat: dict[str, str] = {}
        for key, value in data.items():
            flat[key] = json.dumps(value) if not isinstance(value, str) else value

        message_id = await self._client.xadd(self._stream_name, flat)

        logger.info(
            "Chunk pushed to stream — id=%s, chunk_id=%s",
            message_id,
            data.get("chunk_id", "?"),
        )
        return message_id

    # ---------- consumer ----------

    async def consume_chunks(
        self,
        callback: ChunkCallback,
        *,
        batch_size: int = 1,
        block_ms: int = 5000,
    ) -> None:
        """
        Continuously consume chunks from the stream and process them.

        First processes any pending (un-ACK'd) messages, then reads new ones.

        Args:
            callback: Async function that processes a chunk dict.
                      Must return True on success, False on failure.
            batch_size: Number of messages to read per batch.
            block_ms: Block time in ms when waiting for new messages.
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis — call connect() first")

        self._running = True
        logger.info(
            "Consumer loop started — consumer=%s, group=%s",
            self._consumer_name,
            self._consumer_group,
        )

        # Phase 1: Process any pending messages from previous crashes
        await self._process_pending(callback)

        # Phase 2: Read new messages
        while self._running:
            try:
                messages = await self._client.xreadgroup(
                    groupname=self._consumer_group,
                    consumername=self._consumer_name,
                    streams={self._stream_name: ">"},
                    count=batch_size,
                    block=block_ms,
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        await self._handle_message(msg_id, msg_data, callback)

            except asyncio.CancelledError:
                logger.info("Consumer loop cancelled")
                break
            except Exception:
                logger.exception("Error in consumer loop — retrying in 5s")
                await asyncio.sleep(5)

        logger.info("Consumer loop stopped")

    async def _process_pending(self, callback: ChunkCallback) -> None:
        """Process any pending (previously read but un-ACK'd) messages."""
        if not self._client:
            return

        logger.info("Checking for pending messages...")

        try:
            messages = await self._client.xreadgroup(
                groupname=self._consumer_group,
                consumername=self._consumer_name,
                streams={self._stream_name: "0"},
                count=100,
            )

            if not messages:
                logger.info("No pending messages")
                return

            pending_count = 0
            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    if msg_data:  # non-empty means it's still pending
                        await self._handle_message(msg_id, msg_data, callback)
                        pending_count += 1

            if pending_count > 0:
                logger.info("Processed %d pending messages", pending_count)

        except Exception:
            logger.exception("Error processing pending messages")

    async def _handle_message(
        self,
        msg_id: str,
        msg_data: dict,
        callback: ChunkCallback,
    ) -> None:
        """Handle a single message: process, ACK, or retry/dead-letter."""
        if not self._client:
            return

        # Deserialise JSON-encoded values
        parsed: dict = {}
        for key, value in msg_data.items():
            try:
                parsed[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed[key] = value

        chunk_id = parsed.get("chunk_id", msg_id)

        # Check retry count
        retry_count = await self._get_retry_count(msg_id)
        if retry_count >= self._max_retries:
            logger.error(
                "Chunk %s exceeded max retries (%d) — dead-lettering",
                chunk_id,
                self._max_retries,
            )
            # ACK to remove from pending, but log as dead letter
            await self._client.xack(self._stream_name, self._consumer_group, msg_id)
            await self._dead_letter(msg_id, parsed)
            return

        # Process the chunk
        try:
            success = await callback(parsed)
            if success:
                await self._client.xack(
                    self._stream_name,
                    self._consumer_group,
                    msg_id,
                )
                logger.debug("Message %s ACK'd", msg_id)
            else:
                logger.warning(
                    "Chunk %s processing returned False — will retry (attempt %d/%d)",
                    chunk_id,
                    retry_count + 1,
                    self._max_retries,
                )
        except Exception:
            logger.exception(
                "Chunk %s processing raised error — will retry (attempt %d/%d)",
                chunk_id,
                retry_count + 1,
                self._max_retries,
            )

    async def _get_retry_count(self, msg_id: str) -> int:
        """Get the delivery count (retries) for a message from XPENDING."""
        if not self._client:
            return 0

        try:
            result = await self._client.xpending_range(
                self._stream_name,
                self._consumer_group,
                min=msg_id,
                max=msg_id,
                count=1,
            )
            if result:
                # result[0] is a dict with 'times_delivered'
                return result[0].get("times_delivered", 1) - 1
        except Exception:
            logger.debug("Could not get retry count for %s", msg_id)

        return 0

    async def _dead_letter(self, msg_id: str, data: dict) -> None:
        """Store a failed message in a dead-letter stream for later inspection."""
        if not self._client:
            return

        dead_letter_stream = f"{self._stream_name}:dead-letter"
        try:
            dead_data = {
                "original_id": msg_id,
                "data": json.dumps(data),
                "failed_at": str(time.time()),
                "consumer": self._consumer_name,
            }
            await self._client.xadd(dead_letter_stream, dead_data)
            logger.info("Dead-lettered message %s → %s", msg_id, dead_letter_stream)
        except Exception:
            logger.exception("Failed to dead-letter message %s", msg_id)

    # ---------- accessors ----------

    @property
    def is_connected(self) -> bool:
        return self._client is not None


# ---------------------------------------------------------------------------
# Unit test stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test() -> None:
        # This test just verifies the class can be instantiated
        queue = RedisQueue()
        assert not queue.is_connected
        print("✓ RedisQueue initialisation test passed")
        print("  (Full test requires running Redis instance)")

    asyncio.run(_test())

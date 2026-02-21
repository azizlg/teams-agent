# Intelligent Multilingual Meeting Agent

Autonomous bot that joins Microsoft Teams meetings, captures audio in real-time,
transcribes speech live using Azure Speech Services, reprocesses audio through
Whisper for accuracy, identifies speakers, and generates analytical reports.

## Architecture

```
Teams Meeting ──► AudioStreamHandler ──┬──► AzureSpeechClient (real-time)
                                       └──► ChunkManager ──► Redis Queue
                                                                  │
                                           WhisperWorker ◄────────┘
                                               │
                                           PostgreSQL
```

**Dual transcript pipeline:**
- **Fast path** — Azure Speech SDK provides real-time, low-latency transcription
- **Accurate path** — 5-minute audio chunks are reprocessed by faster-whisper (large-v3)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Bot Framework | botbuilder-core + aiohttp |
| Real-time STT | Azure Cognitive Services Speech |
| Offline STT | faster-whisper (large-v3) |
| Speaker ID | pyannote.audio |
| LLM | Claude (via LangChain) |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 15 + pgvector |
| Queue | Redis 7 Streams |
| Blob Storage | Azure Blob Storage |
| Containerisation | Docker + Docker Compose |

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Azure subscription (Speech, Blob Storage, Bot Service)

### 1. Clone & Configure

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 2. Run with Docker Compose

```bash
docker compose up -d
```

This starts:
- **app** — FastAPI on port 8000
- **whisper-worker** — background transcription worker
- **postgres** — PostgreSQL 15 with pgvector
- **redis** — Redis 7 for the chunk queue

### 3. Run Locally (Development)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Start infrastructure
docker compose up -d postgres redis

# Run the API
uvicorn api.main:app --reload --port 8000

# Run the Whisper worker (separate terminal)
python -m transcription.whisper_worker
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/meetings/join` | Join a Teams meeting |
| `GET` | `/meetings/{id}/status` | Meeting status |
| `GET` | `/meetings/{id}/transcript` | Current transcript |
| `POST` | `/webhook/teams` | Bot Framework webhook |

## Project Structure

```
intelligent-meeting-agent/
├── bot/                    # Teams bot and audio capture
│   ├── teams_bot.py        # Bot Framework definition
│   ├── meeting_handler.py  # Meeting lifecycle events
│   └── audio_stream.py     # PCM audio stream handler
├── transcription/          # Speech-to-text pipeline
│   ├── azure_speech.py     # Real-time Azure Speech client
│   ├── whisper_worker.py   # Offline Whisper reprocessing
│   ├── chunk_manager.py    # Audio buffering & chunking
│   └── diarization.py      # Speaker identification
├── storage/                # Persistence layer
│   ├── blob_storage.py     # Azure Blob Storage
│   ├── database.py         # PostgreSQL + SQLAlchemy
│   └── redis_queue.py      # Redis Streams queue
├── api/                    # REST API
│   └── main.py             # FastAPI application
├── config/                 # Configuration
│   └── settings.py         # pydantic-settings
├── tests/                  # Unit tests
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/ -v
```

## Supported Languages

Auto-detected during transcription:
English, French, Arabic, Spanish, German, Portuguese, Italian, Chinese, Japanese, Korean

## License

Proprietary — Capgemini

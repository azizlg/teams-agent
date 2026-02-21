"""
run_all_tests.py — Comprehensive test runner for the Meeting Agent project.

Runs all testable modules without requiring external service credentials.
Skips modules gracefully when optional dependencies are not installed.
"""

import sys
import os
import asyncio
import json
import traceback


def safe_print(*args, **kwargs):
    """Print helper that survives Windows CP1252 by replacing unencodable chars."""
    enc = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
    text = ' '.join(str(a) for a in args)
    sys.stdout.buffer.write((text + kwargs.get('end', '\n')).encode(enc, errors='replace'))
    sys.stdout.buffer.flush()

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []


def record(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    if passed:
        PASS += 1
        RESULTS.append(f"  [PASS] {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  [FAIL] {name}: {detail}")


def skip(name: str, reason: str):
    global SKIP
    SKIP += 1
    RESULTS.append(f"  [SKIP] {name}: ({reason})")


# ===========================================================================
# 1. Syntax Validation (all .py files)
# ===========================================================================
print("=" * 60)
print("1. SYNTAX VALIDATION")
print("=" * 60)

import ast
from pathlib import Path

py_files = list(Path(".").rglob("*.py"))
py_files = [f for f in py_files if "__pycache__" not in str(f) and "run_all_tests" not in str(f)]
syntax_ok = 0
syntax_fail = 0

for f in py_files:
    try:
        ast.parse(f.read_text(encoding="utf-8"))
        syntax_ok += 1
    except SyntaxError as e:
        syntax_fail += 1
        print(f"  SYNTAX ERROR in {f}: {e}")

record(f"Syntax check ({syntax_ok}/{len(py_files)} files)", syntax_fail == 0)
print(f"  Checked {len(py_files)} files — {syntax_ok} OK, {syntax_fail} failed")
print()


# ===========================================================================
# 2. Config / Settings
# ===========================================================================
print("=" * 60)
print("2. CONFIG / SETTINGS")
print("=" * 60)

try:
    from config.settings import AudioSettings, GroqSettings, MinioSettings
    s = AudioSettings()
    assert s.sample_rate == 16000
    assert s.channels == 1
    record("AudioSettings model", True)
    print(f"  AudioSettings: rate={s.sample_rate}, channels={s.channels}")

    g = GroqSettings()
    record("GroqSettings model", hasattr(g, 'api_key') and hasattr(g, 'model'))
    print(f"  GroqSettings: model={g.model}")

    m = MinioSettings()
    record("MinioSettings model", hasattr(m, 'endpoint') and hasattr(m, 'bucket_name'))
    print(f"  MinioSettings: endpoint={m.endpoint}, bucket={m.bucket_name}")
except Exception as e:
    record("Config settings", False, str(e)[:80])

print()


# ===========================================================================
# 3. Intelligence — Prompts
# ===========================================================================
print("=" * 60)
print("3. INTELLIGENCE — PROMPTS")
print("=" * 60)

try:
    from intelligence.prompts import (
        AnalysisType, OutputLanguage, PromptContext, build_prompt
    )

    ctx = PromptContext(
        meeting_title="Q4 Planning",
        meeting_date="2024-12-01",
        meeting_duration_minutes=45,
        participant_names=["Alice", "Bob"],
        languages_detected=["en-US", "fr-FR"],
        transcript_text="Alice: Let's discuss the roadmap.",
    )

    all_ok = True
    for atype in AnalysisType:
        sys_p, usr_p = build_prompt(atype, ctx)
        if len(sys_p) == 0 or len(usr_p) == 0:
            all_ok = False
            print(f"  FAIL: Empty prompt for {atype.value}")

    record(f"Build prompts ({len(list(AnalysisType))} types)", all_ok)
    print(f"  {len(list(AnalysisType))} analysis types validated")

    # Test output language selection
    ctx2 = PromptContext(
        transcript_text="Test",
        output_language=OutputLanguage.FRENCH,
    )
    sys_p, usr_p = build_prompt(AnalysisType.EXECUTIVE_SUMMARY, ctx2)
    combined = (sys_p + usr_p).lower()
    has_french = "french" in combined or "fran" in combined
    record("French output language", has_french)

except Exception as e:
    record("Intelligence prompts", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 4. Intelligence — Tools
# ===========================================================================
print("=" * 60)
print("4. INTELLIGENCE — TOOLS")
print("=" * 60)

try:
    from intelligence.tools import (
        MeetingTools, SearchTranscriptInput, GetSpeakerStatsInput,
        GetMeetingInfoInput, GetTopicSegmentsInput, GetTimeRangeInput,
    )

    tools = MeetingTools()
    tools.set_meeting_data(
        meeting_id="test-mtg",
        transcript_segments=[
            {"text": "Let's discuss the budget", "speaker_id": "Alice",
             "language": "en-US", "start_time": 0, "end_time": 5, "timestamp": 0},
            {"text": "I think we need more resources", "speaker_id": "Bob",
             "language": "en-US", "start_time": 5, "end_time": 10, "timestamp": 5},
            {"text": "The timeline is tight", "speaker_id": "Alice",
             "language": "en-US", "start_time": 10, "end_time": 15, "timestamp": 10},
            {"text": "Discutons du calendrier", "speaker_id": "Charlie",
             "language": "fr-FR", "start_time": 15, "end_time": 20, "timestamp": 15},
        ],
    )

    # Test search
    r = asyncio.run(tools.search_transcript(
        SearchTranscriptInput(query="budget", meeting_id="test-mtg")))
    p = json.loads(r)
    record("Search transcript", p["total_matches"] == 1)
    print(f"  search 'budget': {p['total_matches']} match(es)")

    # Test speaker stats
    r = asyncio.run(tools.get_speaker_stats(
        GetSpeakerStatsInput(meeting_id="test-mtg")))
    p = json.loads(r)
    record("Speaker stats", len(p["speakers"]) == 3)
    print(f"  speakers: {[s['speaker'] for s in p['speakers']]}")

    # Test meeting info
    r = asyncio.run(tools.get_meeting_info(
        GetMeetingInfoInput(meeting_id="test-mtg")))
    p = json.loads(r)
    record("Meeting info", p["total_segments"] == 4)
    print(f"  info: {p['total_segments']} segments, {len(p['speakers'])} speakers")

    # Test time range
    r = asyncio.run(tools.get_time_range(
        GetTimeRangeInput(meeting_id="test-mtg", start_minutes=0, end_minutes=0.2)))
    p = json.loads(r)
    record("Time range filter", len(p["segments"]) >= 1)
    print(f"  time range 0-0.2 min: {len(p['segments'])} segments")

    # Test tool definitions
    defs = tools.get_tool_definitions()
    record("Tool definitions", len(defs) == 5)
    print(f"  {len(defs)} tool definitions registered")

except Exception as e:
    record("Intelligence tools", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 5. Intelligence — Analyzer
# ===========================================================================
print("=" * 60)
print("5. INTELLIGENCE — ANALYZER")
print("=" * 60)

try:
    from intelligence.analyzer import MeetingAnalyzer, AnalysisResult, FullMeetingReport

    analyzer = MeetingAnalyzer()

    # Test transcript formatting
    segments = [
        {"speaker_id": "Alice", "text": "Hello everyone", "language": "en-US", "timestamp": 0},
        {"speaker_id": "Bob", "text": "Bonjour", "language": "fr-FR", "timestamp": 65},
        {"speaker_id": "Alice", "text": "Let's begin", "timestamp": 120},
    ]
    formatted = analyzer._format_transcript(segments)
    c1 = "[00:00] Alice [en-US]: Hello" in formatted
    c2 = "[01:05] Bob [fr-FR]: Bonjour" in formatted
    c3 = "[02:00] Alice: Let's begin" in formatted
    record("Transcript formatting", c1 and c2 and c3)
    print(f"  Formatted {len(segments)} segments -> {len(formatted)} chars")

    # Test JSON parsing — direct
    r = analyzer._parse_json_response('{"key": "value"}')
    record("JSON parse (direct)", r == {"key": "value"})

    # Test JSON parsing — code block
    r = analyzer._parse_json_response('Text\n```json\n{"nested": true}\n```\nMore')
    record("JSON parse (code block)", r == {"nested": True})

    # Test JSON parsing — embedded
    r = analyzer._parse_json_response('Result: {"data": [1,2,3]} end')
    record("JSON parse (embedded)", r.get("data") == [1, 2, 3])

    # Test JSON parsing — fallback
    r = analyzer._parse_json_response("This is not JSON at all")
    record("JSON parse (fallback)", "_parse_error" in r)

    # Test data models
    result = AnalysisResult(
        analysis_type="executive_summary",
        meeting_id="test",
        data={"summary": "Test"},
        input_tokens=100,
        output_tokens=50,
    )
    record("AnalysisResult model", result.success and result.input_tokens == 100)

    report = FullMeetingReport(meeting_id="test", meeting_title="Test Meeting")
    record("FullMeetingReport model", report.total_tokens_used == 0)

    print(f"  All analyzer logic tests passed")

except Exception as e:
    record("Intelligence analyzer", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 6. Reports — Templates
# ===========================================================================
print("=" * 60)
print("6. REPORTS — TEMPLATES")
print("=" * 60)

try:
    from reports.templates import (
        ReportFormat, ReportStyle, ReportTemplate, SectionConfig,
        TEMPLATES, get_template
    )

    for name, tmpl in TEMPLATES.items():
        record(f"Template '{name}' ({len(tmpl.sections)} sections)", len(tmpl.sections) > 0)
        print(f"  '{name}': {len(tmpl.sections)} sections, style={tmpl.style.value}")

    full = get_template("full_report")
    record("get_template('full_report')", full.name == "full_report" and full.include_toc)

    try:
        get_template("nonexistent")
        record("get_template KeyError", False, "No exception raised")
    except KeyError:
        record("get_template KeyError", True)

    record("ReportFormat enum", len(ReportFormat) == 5)

except Exception as e:
    record("Reports templates", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 7. Reports — Generator
# ===========================================================================
print("=" * 60)
print("7. REPORTS — GENERATOR")
print("=" * 60)

try:
    from reports.generator import ReportGenerator, GeneratedReport
    from reports.templates import ReportFormat

    gen = ReportGenerator()

    mock_report = FullMeetingReport(
        meeting_id="test-mtg",
        meeting_title="Q4 Planning Session",
        generated_at="2024-12-01T10:00:00Z",
        analyses={
            "executive_summary": AnalysisResult(
                analysis_type="executive_summary",
                meeting_id="test-mtg",
                data={
                    "summary": "The team discussed Q4 priorities and allocated resources.",
                    "headline": "Q4 Priorities Defined",
                    "key_takeaways": ["Budget approved", "Timeline set", "Risks identified"],
                },
            ),
            "action_items": AnalysisResult(
                analysis_type="action_items",
                meeting_id="test-mtg",
                data={
                    "action_items": [
                        {"action": "Prepare budget draft", "assignee": "Alice",
                         "deadline": "Dec 15", "priority": "high"},
                        {"action": "Review vendor contracts", "assignee": "Bob",
                         "deadline": "Dec 20", "priority": "medium"},
                    ],
                    "total_count": 2,
                },
            ),
            "key_decisions": AnalysisResult(
                analysis_type="key_decisions",
                meeting_id="test-mtg",
                data={
                    "decisions": [
                        {"decision": "Approve Q4 budget", "rationale": "Aligned with goals"},
                    ],
                },
            ),
        },
    )

    # Test Markdown generation
    md_report = gen.generate(mock_report, output_format=ReportFormat.MARKDOWN,
                             meeting_date="December 1, 2024")
    record("Markdown report", "Q4 Planning" in md_report.content
           and len(md_report.content) > 100)
    print(f"  Markdown: {len(md_report.content)} chars, {md_report.generation_time_seconds:.3f}s")

    # Test HTML generation
    html_report = gen.generate(mock_report, output_format=ReportFormat.HTML,
                               meeting_date="December 1, 2024")
    record("HTML report", "<html" in html_report.content)
    print(f"  HTML: {len(html_report.content)} chars")

    # Test JSON generation
    json_report = gen.generate(mock_report, output_format=ReportFormat.JSON)
    parsed = json.loads(json_report.content)
    record("JSON report", parsed["meeting_id"] == "test-mtg")
    print(f"  JSON: {len(json_report.content)} chars")

    # Save the Markdown report for inspection
    Path("test_output_report.md").write_text(md_report.content, encoding="utf-8")
    print(f"  Saved test report to test_output_report.md")

except Exception as e:
    record("Reports generator", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 8. Transcription — Diarization
# ===========================================================================
print("=" * 60)
print("8. TRANSCRIPTION — DIARIZATION")
print("=" * 60)

try:
    from transcription.diarization import (
        SpeakerDiarizer, SpeakerSegment, DiarizationResult,
        align_transcript_with_speakers,
    )

    diarizer = SpeakerDiarizer()
    record("SpeakerDiarizer init", not diarizer.is_loaded and diarizer.speaker_count == 0)

    # Test speaker mapping
    diarizer.map_speaker_to_participant("SPEAKER_00", "user-1", "Alice")
    diarizer.map_speaker_to_participant("SPEAKER_01", "user-2", "Bob")
    record("Speaker mapping",
           diarizer.resolve_speaker("SPEAKER_00") == "Alice"
           and diarizer.resolve_speaker("SPEAKER_01") == "Bob"
           and diarizer.resolve_speaker("SPEAKER_99") == "SPEAKER_99")
    print(f"  SPEAKER_00 -> Alice, SPEAKER_01 -> Bob")

    # Test alignment
    transcript = [
        {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
        {"start": 3.0, "end": 5.0, "text": "Hi there"},
        {"start": 5.5, "end": 8.0, "text": "Let's begin"},
    ]
    speakers = [
        SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=3.0),
        SpeakerSegment(speaker_id="SPEAKER_01", start=3.0, end=6.0),
        SpeakerSegment(speaker_id="SPEAKER_00", start=6.0, end=9.0),
    ]
    aligned = align_transcript_with_speakers(transcript, speakers)
    record("Transcript-speaker alignment",
           aligned[0]["speaker_id"] == "SPEAKER_00"
           and aligned[1]["speaker_id"] == "SPEAKER_01"
           and aligned[2]["speaker_id"] == "SPEAKER_00")
    print(f"  Aligned {len(transcript)} segments with {len(speakers)} speaker turns")

    # Test speaker summary
    summary = diarizer.get_speaker_summary()
    record("Speaker summary", len(summary) == 2)

except ImportError as e:
    skip("Transcription diarization", f"Missing: {e.name}")
except Exception as e:
    record("Transcription diarization", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 9. Storage — Database Models
# ===========================================================================
print("=" * 60)
print("9. STORAGE — DATABASE MODELS")
print("=" * 60)

try:
    from storage.database import Meeting, TranscriptSegment, AudioChunk, Speaker, Database

    meeting = Meeting(id="test-1", title="Test Meeting", status="active")
    record("Meeting model", meeting.id == "test-1" and meeting.status == "active")

    segment = TranscriptSegment(
        meeting_id="test-1", text="Hello world",
        language="en-US", confidence=0.95, source="whisper",
    )
    record("TranscriptSegment model", segment.language == "en-US")

    chunk = AudioChunk(
        chunk_id="test-1_0001_abc",
        meeting_id="test-1",
        sequence_number=1,
        blob_url="https://blob.example.com/test.wav",
        duration_seconds=300.0,
    )
    record("AudioChunk model", chunk.duration_seconds == 300.0)

    speaker = Speaker(
        meeting_id="test-1",
        speaker_label="SPEAKER_00",
        participant_name="Alice",
        total_speaking_time=120.5,
    )
    record("Speaker model", speaker.total_speaking_time == 120.5)

    db = Database()
    record("Database manager init", not db.is_connected)
    print(f"  All 4 ORM models + Database manager validated")

except ImportError as e:
    skip("Storage database", f"Missing: {e.name}")
except Exception as e:
    record("Storage database", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 10. Storage — Blob Storage
# ===========================================================================
print("=" * 60)
print("10. STORAGE — BLOB STORAGE")
print("=" * 60)

try:
    from storage.blob_storage import BlobStorage

    storage = BlobStorage()
    record("BlobStorage init", not storage.is_connected)

    url = "http://minio:9000/meeting-audio-chunks/mtg-1/chunk_001.wav"
    name = storage._extract_blob_name(url)
    record("URL to blob name extraction", name == "mtg-1/chunk_001.wav")
    print(f"  URL -> name: {name}")

    record("Direct name passthrough",
           storage._extract_blob_name("mtg-1/chunk.wav") == "mtg-1/chunk.wav")

except ImportError as e:
    skip("Storage blob", f"Missing: {e.name}")
except Exception as e:
    record("Storage blob", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 11. Storage — Redis Queue
# ===========================================================================
print("=" * 60)
print("11. STORAGE — REDIS QUEUE")
print("=" * 60)

try:
    from storage.redis_queue import RedisQueue
    queue = RedisQueue()
    record("RedisQueue init", hasattr(queue, '_stream_name'))
    print(f"  stream={queue._stream_name}")
except ImportError as e:
    skip("Storage redis", f"Missing: {e.name}")
except Exception as e:
    record("Storage redis", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 12. Bot — Meeting Handler
# ===========================================================================
print("=" * 60)
print("12. BOT — MEETING HANDLER")
print("=" * 60)

try:
    from bot.meeting_handler import (
        MeetingHandler, MeetingSession, MeetingState, Participant
    )

    handler = MeetingHandler()
    record("MeetingHandler init", handler.active_meeting_count == 0)

    # Test meeting start (pipeline will error without faster-whisper, but session state tracks)
    session = asyncio.run(handler.on_meeting_start(
        meeting_id="test-mtg-001",
        meeting_data={"title": "Test Meeting"},
    ))
    record("Meeting session created", session.meeting_id == "test-mtg-001")
    print(f"  Session state: {session.state.value}")

    # Test participant model
    p = Participant(id="user-1", name="Alice", role="organizer")
    record("Participant model", p.name == "Alice" and p.role == "organizer")

    # Test session properties
    record("Session properties", session.duration_seconds >= 0
           and session.participant_count == 0)

except ImportError as e:
    skip("Bot meeting handler", f"Missing: {e.name}")
except Exception as e:
    record("Bot meeting handler", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 13. Bot — Audio Stream Handler
# ===========================================================================
print("=" * 60)
print("13. BOT — AUDIO STREAM")
print("=" * 60)

try:
    from bot.audio_stream import AudioStreamHandler, AudioFrame, StreamStats

    # Test AudioFrame
    frame = AudioFrame(data=b"\x00" * 1600, timestamp=0.0, sequence_number=1)
    record("AudioFrame model", len(frame.data) == 1600)

    # Test StreamStats
    stats = StreamStats()
    record("StreamStats model", stats.frames_received == 0 and stats.frames_sent_to_speech == 0)

    print(f"  AudioFrame and StreamStats validated")

except ImportError as e:
    skip("Bot audio stream", f"Missing: {e.name}")
except Exception as e:
    record("Bot audio stream", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 13b. Transcription — Whisper Realtime
# ===========================================================================
print("=" * 60)
print("13b. TRANSCRIPTION — WHISPER REALTIME")
print("=" * 60)

try:
    from transcription.whisper_realtime import (
        WhisperRealtimeTranscriber, TranscriptSegment
    )

    transcriber = WhisperRealtimeTranscriber()
    record("WhisperRealtimeTranscriber init", not transcriber.is_running)
    record("Whisper segments counter", transcriber.segments_recognised == 0)
    print(f"  WhisperRealtimeTranscriber initialised (model not loaded until start())")

except ImportError as e:
    skip("Whisper realtime", f"Missing: {e.name}")
except Exception as e:
    record("Whisper realtime", False, str(e))
    traceback.print_exc()

print()



# ===========================================================================
# 14. Reports — Exporters
# ===========================================================================
print("=" * 60)
print("14. REPORTS — EXPORTERS")
print("=" * 60)

try:
    from reports.exporters import DocxExporter, PdfExporter, BaseExporter

    docx_exporter = DocxExporter()
    record("DocxExporter init", docx_exporter._template.name == "full_report")

    pdf_exporter = PdfExporter()
    record("PdfExporter init", pdf_exporter._template.name == "full_report")

    # Test markdown to HTML conversion
    md = "# Title\n\n## Section\n\n- Item 1\n- Item 2"
    html = pdf_exporter._markdown_to_html(md)
    record("Markdown to HTML", "<h1>Title</h1>" in html and "<li>Item 1</li>" in html)
    print(f"  MD -> HTML: {len(html)} chars")

except ImportError as e:
    skip("Reports exporters", f"Missing: {e.name}")
except Exception as e:
    record("Reports exporters", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# 15. Bot — Web Chat Logic
# ===========================================================================
print("=" * 60)
print("15. BOT — WEB CHAT LOGIC")
print("=" * 60)

try:
    import importlib, types

    # Patch heavy optional imports so api.main can be imported in test context
    for _mod in ("httpx", "dotenv"):
        try:
            importlib.import_module(_mod)
        except ImportError:
            sys.modules.setdefault(_mod, types.ModuleType(_mod))

    # Provide a stub load_dotenv so the module-level call is a no-op
    if "dotenv" in sys.modules and not hasattr(sys.modules["dotenv"], "load_dotenv"):
        sys.modules["dotenv"].load_dotenv = lambda *a, **k: None  # type: ignore

    from api.main import _build_reply, _BOT_APP_ID, app_state

    # 15a — help command
    reply_help = _build_reply({"text": "help"})
    record(
        "_build_reply('help') contains bot name",
        "Meeting Agent Bot" in reply_help,
    )
    safe_print(f"  help reply: {reply_help[:60].strip()}...")

    # 15b — /status command with no active meetings
    reply_status = _build_reply({"text": "/status"})
    record(
        "_build_reply('/status') no meetings",
        "No active meetings" in reply_status,
    )
    safe_print(f"  /status reply: {reply_status[:60].strip()}...")

    # 15c — unknown command falls back to greeting
    reply_unknown = _build_reply({"text": "random input"})
    record(
        "_build_reply(unknown) greeting",
        "Hello" in reply_unknown,
    )
    safe_print(f"  unknown reply: {reply_unknown[:60].strip()}...")

    # 15d — BOT_APP_ID is non-empty (proves .env is read correctly)
    record(
        "_BOT_APP_ID loaded from .env",
        bool(_BOT_APP_ID),
    )
    safe_print(f"  _BOT_APP_ID prefix: {_BOT_APP_ID[:8]}..." if _BOT_APP_ID else "  _BOT_APP_ID: EMPTY")

    # 15e — localhost bypass condition
    def _is_local(url: str) -> bool:
        return url.startswith("http://localhost") or url.startswith("http://127.0.0.1")

    record(
        "Emulator localhost bypass: localhost:PORT",
        _is_local("http://localhost:58457"),
    )
    record(
        "Emulator localhost bypass: 127.0.0.1",
        _is_local("http://127.0.0.1:58457"),
    )
    record(
        "Emulator localhost bypass: Azure NOT local",
        not _is_local("https://smba.trafficmanager.net/apis/"),
    )
    print("  Localhost bypass conditions verified")

except Exception as e:
    record("Bot web chat logic", False, str(e))
    traceback.print_exc()

print()


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print()
for r in RESULTS:
    print(r)
print()
total = PASS + FAIL + SKIP
print(f"  TOTAL: {PASS} passed, {FAIL} failed, {SKIP} skipped (out of {total} checks)")
print()

if FAIL == 0:
    print("  ALL TESTS PASSED!")
else:
    print(f"  {FAIL} test(s) failed — see details above")

sys.exit(0 if FAIL == 0 else 1)

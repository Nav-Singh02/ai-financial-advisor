import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_log.jsonl"


def log_interaction(
    question: str,
    summary: str,
    risk_notes: str,
    draft_email: str,
    source_chunks=None,
) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "summary": summary,
        "risk_notes": risk_notes,
        "draft_email": draft_email,
        "num_chunks_retrieved": len(source_chunks) if source_chunks else 0,
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Audit record written to %s", _LOG_PATH)

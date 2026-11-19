"""Loads, saves, and deletes company Markdown documents for the KYC service.

Documents are stored at:
    data/know_your_company/docs/<doc_id>.md

The original filename is embedded as the first line of each file so it
survives as a human-readable citation label without a separate index file.

Format of each stored file:
    filename:<original_filename>
    <blank line>
    <markdown content>
"""

from pathlib import Path

from src.config import DATA_DIR
from src.utils.logging_utils import console_log

_KYC_DOCS_DIR = DATA_DIR / "know_your_company" / "docs"
_FILENAME_PREFIX = "filename:"
_MODE = "know_your_company"


def _docs_dir() -> Path:
    _KYC_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return _KYC_DOCS_DIR


def save_doc(doc_id: str, filename: str, markdown: str) -> Path:
    """Persist a converted-MD doc. Returns the saved file path."""
    path = _docs_dir() / f"{doc_id}.md"
    path.write_text(f"{_FILENAME_PREFIX}{filename}\n\n{markdown}", encoding="utf-8")
    return path


def delete_doc(doc_id: str) -> bool:
    """Delete a doc by doc_id. Returns True if removed, False if not found."""
    path = _docs_dir() / f"{doc_id}.md"
    if path.exists():
        path.unlink()
        return True
    return False


def list_docs() -> list[dict]:
    """List stored docs without loading content: [{doc_id, filename}]."""
    result = []
    for path in sorted(_docs_dir().glob("*.md")):
        doc_id = path.stem
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            filename = (
                first_line[len(_FILENAME_PREFIX):]
                if first_line.startswith(_FILENAME_PREFIX)
                else path.name
            )
        except Exception:
            filename = path.name
        result.append({"doc_id": doc_id, "filename": filename})
    return result


def load_docs() -> list[dict]:
    """Load all docs with content: [{doc_id, filename, content}]."""
    docs = []
    for path in sorted(_docs_dir().glob("*.md")):
        doc_id = path.stem
        try:
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            if lines and lines[0].startswith(_FILENAME_PREFIX):
                filename = lines[0][len(_FILENAME_PREFIX):]
                # skip the filename line + the blank separator line
                content = "\n".join(lines[2:]).strip()
            else:
                filename = path.name
                content = text
            docs.append({"doc_id": doc_id, "filename": filename, "content": content})
        except Exception as exc:
            console_log(_MODE, "DOC_LOADER", f"failed to read {path.name}: {exc}")
    return docs


def join_docs_for_context(docs: list[dict]) -> str:
    """Format loaded docs into the static_context block passed to the LLM.

    Each document is separated by a clear header so the agent can cite by
    filename.
    """
    if not docs:
        return ""
    parts = [f"--- [{doc['filename']}] ---\n{doc['content']}" for doc in docs]
    return "\n\n".join(parts)

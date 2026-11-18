"""CRUD endpoints for the service-level business glossary.

GET  /glossary/{service}        — returns current glossary entries
PUT  /glossary/{service}        — replaces entire glossary with the new list
POST /glossary/{service}/entry  — appends a single entry
DELETE /glossary/{service}/{term} — removes the entry whose term matches
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.utils.glossary_loader import load_glossary, save_glossary

router = APIRouter(prefix="/glossary", tags=["Glossary"])


class GlossaryEntry(BaseModel):
    term: str
    definition: str = ""
    notes: str = ""


class GlossaryList(BaseModel):
    entries: list[GlossaryEntry]


@router.get("/{service}")
async def get_glossary(service: str) -> dict:
    return {"service": service, "entries": load_glossary(service)}


@router.put("/{service}")
async def replace_glossary(service: str, body: GlossaryList) -> dict:
    entries = [e.model_dump() for e in body.entries]
    save_glossary(service, entries)
    return {"success": True, "service": service, "count": len(entries)}


@router.post("/{service}/entry")
async def add_entry(service: str, entry: GlossaryEntry) -> dict:
    entries = load_glossary(service)
    term_lower = entry.term.strip().lower()
    if not term_lower:
        raise HTTPException(status_code=400, detail="term is required")
    # Replace if term already exists (case-insensitive), else append
    replaced = False
    for i, e in enumerate(entries):
        if e.get("term", "").lower() == term_lower:
            entries[i] = entry.model_dump()
            replaced = True
            break
    if not replaced:
        entries.append(entry.model_dump())
    save_glossary(service, entries)
    return {"success": True, "replaced": replaced, "count": len(entries)}


@router.delete("/{service}/{term}")
async def delete_entry(service: str, term: str) -> dict:
    entries = load_glossary(service)
    term_lower = term.strip().lower()
    before = len(entries)
    entries = [e for e in entries if e.get("term", "").lower() != term_lower]
    save_glossary(service, entries)
    return {"success": True, "removed": before - len(entries)}

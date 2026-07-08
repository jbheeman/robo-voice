from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MEMORY_FILE = Path("memory_v2.json")


def normalize_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {"users": {}}

    users = memory.get("users", {})
    if not isinstance(users, dict):
        return normalized

    for name, data in users.items():
        if not isinstance(name, str):
            continue

        notes: List[str] = []

        if isinstance(data, dict):
            old_notes = data.get("notes", [])
            if isinstance(old_notes, list):
                notes = [
                    str(note).strip()
                    for note in old_notes
                    if str(note).strip()
                ]

        normalized["users"][name] = {"notes": notes}

    return normalized


def load_memory() -> Dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {"users": {}}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        raw_memory = json.load(f)

    return normalize_memory(raw_memory)


def save_memory(memory: Dict[str, Any]) -> None:
    memory = normalize_memory(memory)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def set_user(memory: Dict[str, Any], name: str) -> None:
    name = name.strip()

    if not name:
        return

    if name not in memory["users"]:
        memory["users"][name] = {"notes": []}


def get_user_memory(memory: Dict[str, Any], name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}

    return memory["users"].get(name, {"notes": []})


def add_note(memory: Dict[str, Any], name: str, note: str) -> bool:
    name = name.strip()
    note = note.strip()

    if not name or not note:
        return False

    set_user(memory, name)

    notes = memory["users"][name]["notes"]
    existing_notes = {old_note.strip().lower() for old_note in notes}

    if note.lower() in existing_notes:
        return False

    notes.append(note)
    return True
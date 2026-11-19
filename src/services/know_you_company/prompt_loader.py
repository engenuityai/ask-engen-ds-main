"""KYC service prompt loader.

Mirrors the interface of src/utils/prompt_loader.py but points exclusively
to this service's own agents.yml and tasks.yml files, keeping KYC prompts
completely isolated from analytics.
"""

from pathlib import Path

import yaml

_AGENTS_PATH = Path(__file__).resolve().parent / "agents" / "agents.yml"
_TASKS_PATH = Path(__file__).resolve().parent / "tasks" / "tasks.yml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_agent(name: str) -> dict:
    agents = _load_yaml(_AGENTS_PATH)
    if name not in agents:
        raise KeyError(f"KYC agent '{name}' not found in {_AGENTS_PATH}")
    return agents[name]


def load_task(name: str) -> dict:
    tasks = _load_yaml(_TASKS_PATH)
    if name not in tasks:
        raise KeyError(f"KYC task '{name}' not found in {_TASKS_PATH}")
    return tasks[name]


def _render(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def build_agent_system_prompt(agent_name: str) -> str:
    agent = load_agent(agent_name)
    return (
        f"Role: {agent['role'].strip()}\n\n"
        f"Goal: {agent['goal'].strip()}\n\n"
        f"Backstory: {agent['backstory'].strip()}"
    )


def render_task_static_context(name: str, **values: str) -> str:
    """Renders static_context + instructions + output — the cacheable portion.

    Returns "" when the task has no static_context section so callers can
    treat that as "nothing to cache, use render_task_prompt instead".
    """
    task = load_task(name)
    static = task.get("static_context", "")
    if not static.strip():
        return ""
    sections = [static, task.get("instructions", ""), task.get("output", "")]
    template = "\n\n".join(s.strip() for s in sections if s and s.strip())
    return _render(template, **values)


def render_task_volatile_prompt(name: str, **values: str) -> str:
    """Renders just the input section — the per-request volatile part."""
    task = load_task(name)
    template = task.get("input", "").strip()
    return _render(template, **values)

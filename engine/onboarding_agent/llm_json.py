"""
llm_json.py
===========

Robust JSON extraction from LLM responses.

Live models (Anthropic etc.) frequently wrap JSON in prose or markdown code
fences, so a bare ``json.loads`` silently fails. This helper strips fences,
ignores surrounding prose and extracts the first balanced JSON array/object, and
ALWAYS reports an explicit parse status/error so the caller can label results
honestly (never claim an LLM result when parsing failed).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

OK = "ok"
OK_EXTRACTED = "ok_extracted"
EMPTY = "empty"
PARSE_FAILED = "parse_failed"


def extract_json(raw: Any) -> Tuple[Optional[Any], str, str]:
    """Return (parsed_obj_or_None, parse_status, parse_error).

    parse_status in {ok, ok_extracted, empty, parse_failed}. Never raises.
    """
    if raw is None:
        return None, EMPTY, "no response"
    if not isinstance(raw, str):
        return raw, OK, ""
    text = raw.strip()
    if not text:
        return None, EMPTY, "empty response"

    # Strip a leading ```json / ``` fence if present.
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    # Direct parse first.
    try:
        return json.loads(text), OK, ""
    except json.JSONDecodeError:
        pass

    # Extract the first balanced [...] or {...} block.
    last_err = ""
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start = text.find(open_c)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate), OK_EXTRACTED, ""
                    except json.JSONDecodeError as exc:
                        last_err = str(exc)
                        break
    return None, PARSE_FAILED, last_err or "could not parse JSON from response"


def extract_json_list(raw: Any) -> Tuple[list, str, str]:
    """Extract a LIST of JSON objects from an LLM response.

    Handles a JSON array, a single object, or JSON-lines (one object per line).
    Returns (list, parse_status, parse_error) — list may be empty.
    """
    obj, status, err = extract_json(raw)
    if isinstance(obj, list):
        return [o for o in obj if isinstance(o, dict)], status, err
    if isinstance(obj, dict):
        return [obj], status, err
    # Fallback: scan for multiple standalone objects (JSON-lines or concatenated).
    if isinstance(raw, str):
        items: list = []
        for chunk in re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, re.S):
            try:
                d = json.loads(chunk)
                if isinstance(d, dict):
                    items.append(d)
            except json.JSONDecodeError:
                continue
        if items:
            return items, OK_EXTRACTED, ""
    return [], status, err


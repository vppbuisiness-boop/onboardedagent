"""Underdog Fantasy pick'em client (UNVERIFIED from this environment).

Underdog's public over_under_lines endpoint answers HTTP 426 ('A new version is
required') unless the request carries the same client headers the web app sends.
The app itself is behind a bot wall from the research container, so the exact
header values could not be captured here. Capture them once from your browser's
devtools (Network tab, any request to api.underdogfantasy.com) and put them in
EDGELINE_UNDERDOG_HEADERS as a JSON object, e.g.
  {"Client-Type": "web", "Client-Version": "<value>", "Client-Device-Id": "<uuid>"}
"""
from __future__ import annotations

import json
import os

import requests

from ..config import USER_AGENT

API = "https://api.underdogfantasy.com/beta/v6/over_under_lines"


def fetch_raw(headers: dict | None = None) -> dict:
    h = {"User-Agent": USER_AGENT, "Accept": "application/json", "Referer": "https://underdogfantasy.com/"}
    env = os.environ.get("EDGELINE_UNDERDOG_HEADERS")
    if env:
        h.update(json.loads(env))
    if headers:
        h.update(headers)
    r = requests.get(API, headers=h, timeout=60)
    if r.status_code == 426:
        raise RuntimeError("Underdog requires current client headers; set EDGELINE_UNDERDOG_HEADERS (see module docstring)")
    r.raise_for_status()
    return r.json()

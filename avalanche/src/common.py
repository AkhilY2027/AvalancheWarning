from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

from config import MAX_RETRIES, REQUEST_DELAY_SECONDS, REQUEST_TIMEOUT, USER_AGENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("avalanche")


def cached_get(url: str, *, params: dict[str, Any] | None = None, cache_file: Path) -> Any:
    """GET JSON with a per-request cache, descriptive UA, throttling and backoff."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        import json
        return json.loads(cache_file.read_text())
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            import json
            cache_file.write_text(json.dumps(data))
            return data
        except (requests.RequestException, ValueError) as exc:
            if attempt + 1 == MAX_RETRIES:
                raise
            pause = min(60, 2 ** attempt)
            LOG.warning("Request failed (%s); retrying in %ss", exc, pause)
            time.sleep(pause)
    raise RuntimeError("unreachable")


def cli_seasons(parser):
    parser.add_argument("--start-season", type=int)
    parser.add_argument("--end-season", type=int)
    parser.add_argument("--zones", nargs="+")


def settings(args):
    from config import START_SEASON, END_SEASON, ZONES
    return (args.start_season or START_SEASON, args.end_season or END_SEASON,
            args.zones or ZONES)

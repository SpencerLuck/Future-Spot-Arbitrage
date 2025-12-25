#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from typing import Optional

DEFAULT_INSECURE_SSL = True

try:
    import requests  # type: ignore
    _HAVE_REQUESTS = True
except Exception:
    _HAVE_REQUESTS = False


def http_get(
    path: str,
    params: Optional[dict],
    *,
    base: str,
    user_agent: str,
    insecure_ssl: bool = DEFAULT_INSECURE_SSL,
    timeout: int = 30,
):
    url = base + path
    params = params or {}

    if _HAVE_REQUESTS:
        verify: bool | str = True
        if insecure_ssl:
            verify = False
        elif os.environ.get("REQUESTS_CA_BUNDLE"):
            verify = os.environ["REQUESTS_CA_BUNDLE"]
        r = requests.get(
            url,
            params=params,
            timeout=timeout,
            verify=verify,
            headers={"User-Agent": user_agent},
        )
        r.raise_for_status()
        return r.json()

    import ssl
    import urllib.parse
    import urllib.request

    qs = urllib.parse.urlencode(params)
    full_url = url + ("?" + qs if qs else "")
    req = urllib.request.Request(full_url, headers={"User-Agent": user_agent})

    if insecure_ssl:
        ctx = ssl._create_unverified_context()
    else:
        cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
        if cafile:
            ctx = ssl.create_default_context(cafile=cafile)
        else:
            try:
                import certifi  # type: ignore
                ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ctx = ssl.create_default_context()

    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_with_retry(
    path: str,
    params: dict,
    *,
    base: str,
    user_agent: str,
    insecure_ssl: bool = DEFAULT_INSECURE_SSL,
    retries: int = 3,
    pause: float = 0.5,
):
    delay = pause
    for attempt in range(1, retries + 1):
        try:
            return http_get(
                path,
                params,
                base=base,
                user_agent=user_agent,
                insecure_ssl=insecure_ssl,
            )
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 5.0)

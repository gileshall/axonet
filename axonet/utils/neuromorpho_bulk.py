#!/usr/bin/env python3
"""
neuromorpho_bulk.py

Bulk download NeuroMorpho.Org metadata + SWC files using ONLY requests.

Key features (mirrors the R code behavior you shared):
- Query metadata via /api/neuron/select with pagination (page starts 0, size <= 500). :contentReference[oaicite:3]{index=3}
- Default standardized SWC URL pattern:
    http(s)://neuromorpho.org/dableFiles/<archiveLower>/CNG%20version/<neuron_name>.CNG.swc :contentReference[oaicite:4]{index=4}
- Optional robust mode (--find) scrapes each neuron_info.jsp page to discover the exact
  "Morphology File (Standardized)" link (like find=TRUE in your R code). :contentReference[oaicite:5]{index=5}
- SSL verification can be disabled (--insecure) to work around expired certs.

Examples:
  # Metadata only
  python neuromorpho_bulk.py --query 'species:mouse AND brain_region:"neocortex"' --out outdir --metadata-only --insecure

  # Download standardized SWCs using the default URL pattern
  python neuromorpho_bulk.py --query 'species:mouse' --out outdir --insecure

  # Download standardized SWCs using per-neuron scraping (more stable)
  python neuromorpho_bulk.py --query 'species:mouse' --out outdir --find --insecure
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
import requests

try:
    import urllib3
except Exception:
    urllib3 = None


DEFAULT_SITE = "https://neuromorpho.org"
DEFAULT_API_BASE = DEFAULT_SITE + "/api"
NEURON_SELECT = DEFAULT_API_BASE + "/neuron/select"


# --- Helpers ---

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def backoff_sleep(attempt: int, base: float = 0.6, cap: float = 10.0) -> None:
    time.sleep(min(cap, base * (2 ** attempt)))


def get_text(session: requests.Session, url: str, *, verify: bool, timeout: float, retries: int = 6) -> str:
    last = None
    for a in range(retries):
        try:
            r = session.get(url, verify=verify, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            backoff_sleep(a)
    raise RuntimeError(f"GET failed: {url} ({last})")


def get_json(session: requests.Session, url: str, *, params: Dict[str, Any], verify: bool, timeout: float, retries: int = 6) -> Dict[str, Any]:
    last = None
    for a in range(retries):
        try:
            r = session.get(url, params=params, verify=verify, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            backoff_sleep(a)
    raise RuntimeError(f"GET JSON failed: {url} params={params} ({last})")


def download_file(session: requests.Session, url: str, out_path: str, *, verify: bool, timeout: float, retries: int = 6) -> None:
    last = None
    for a in range(retries):
        try:
            r = session.get(url, verify=verify, timeout=timeout)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            return
        except Exception as e:
            last = e
            backoff_sleep(a)
    raise RuntimeError(f"Download failed: {url} ({last})")


def archive_lower(x: str) -> str:
    return (x or "").strip().lower()


def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:240] if len(s) > 240 else s


def extract_neuron_list(page_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    # The API often returns Spring-style HAL with _embedded containing the list.
    if isinstance(page_json.get("_embedded"), dict):
        for v in page_json["_embedded"].values():
            if isinstance(v, list):
                return v

    for k in ("neurons", "Neuron", "results", "data"):
        if isinstance(page_json.get(k), list):
            return page_json[k]

    return []


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- SWC URL logic (pattern + scraping fallback) ---

# Regex based on your R code approach: find an <a> that references dableFiles and contains
# "Morphology File (Standardized)" somewhere around it. :contentReference[oaicite:6]{index=6}
STD_LINK_REGEX = re.compile(
    r'href\s*=\s*("?)(dableFiles[^"\s>]+)\1[^<]*</a>|</a>[^<]*href\s*=\s*("?)(dableFiles[^"\s>]+)\3',
    re.IGNORECASE
)

def find_standardized_swc_url_by_scrape(session: requests.Session, site_base: str, neuron_name: str, *, verify: bool, timeout: float) -> Optional[str]:
    """
    Scrape neuron_info.jsp to locate the standardized morphology SWC link.
    This emulates your R code's `find=TRUE` path. :contentReference[oaicite:7]{index=7}
    """
    info_url = f"{site_base}/neuron_info.jsp?neuron_name={requests.utils.quote(neuron_name)}"
    html = get_text(session, info_url, verify=verify, timeout=timeout)

    # The page contains text like "Morphology File (Standardized)" and an href to dableFiles/...
    # We'll just grab dableFiles hrefs and prefer those that look like SWC.
    candidates = []

    # Quick filter: split around </a> like the R code does, then select those with the label
    parts = html.split("</a>")
    for part in parts:
        if "Morphology File (Standardized)" in part:
            m = re.search(r'href\s*=\s*("?)(dableFiles[^"\s>]+)\1', part, flags=re.IGNORECASE)
            if m:
                candidates.append(m.group(2))

    # If that didn't work (HTML changes), fall back to any dableFiles link ending with .swc on the page
    if not candidates:
        for m in re.finditer(r'href\s*=\s*("?)(dableFiles[^"\s>]+\.swc)\1', html, flags=re.IGNORECASE):
            candidates.append(m.group(2))

    if not candidates:
        return None

    # Convert relative to absolute
    rel = candidates[0]
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    return site_base.rstrip("/") + "/" + rel.lstrip("/")


def build_standardized_swc_url_pattern(site_base: str, archive: str, neuron_name: str) -> str:
    """
    Default standardized SWC path pattern from the R code:
      /dableFiles/<archiveLower>/CNG%20version/<neuron_name>.CNG.swc :contentReference[oaicite:8]{index=8}
    """
    arch = archive_lower(archive)
    # Keep the same encoding as R: "CNG%20version"
    return f"{site_base}/dableFiles/{arch}/CNG%20version/{neuron_name}.CNG.swc"


# --- Main paging loop ---

def iter_pages(session: requests.Session, query: str, *, page_size: int, start_page: int,
               max_pages: Optional[int], verify: bool, timeout: float) -> Iterable[Tuple[int, Dict[str, Any]]]:
    page = start_page
    yielded = 0
    while True:
        if max_pages is not None and yielded >= max_pages:
            return
        params = {"q": query, "page": page, "size": page_size}
        data = get_json(session, NEURON_SELECT, params=params, verify=verify, timeout=timeout)
        yield page, data

        neurons = extract_neuron_list(data)
        if not neurons:
            return

        # Many API pages expose totalPages; stop if present and we reached the end.
        tp = data.get("totalPages")
        if isinstance(tp, int) and page >= tp - 1:
            return

        page += 1
        yielded += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help='Search query, e.g. species:mouse AND brain_region:"neocortex"')
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--page-size", type=int, default=200, help="Max 500 :contentReference[oaicite:9]{index=9}")
    ap.add_argument("--start-page", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--find", action="store_true", help="Scrape neuron_info.jsp to discover standardized SWC link (slower, more stable) :contentReference[oaicite:10]{index=10}")
    ap.add_argument("--site", default=DEFAULT_SITE, help="Base site URL (default https://neuromorpho.org). You can set http://neuromorpho.org too.")
    ap.add_argument("--insecure", action="store_true", help="Disable SSL verification (verify=False)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not (1 <= args.page_size <= 500):
        print("ERROR: --page-size must be between 1 and 500.", file=sys.stderr)
        return 2

    verify = not args.insecure
    if args.insecure and urllib3 is not None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    out_dir = args.out
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "swc"))

    meta_path = os.path.join(out_dir, "metadata.jsonl")
    log_path = os.path.join(out_dir, "download_log.jsonl")

    sess = requests.Session()
    sess.headers.update({"User-Agent": "neuromorpho-bulk-requests/2.0 (requests-only)"})

    # Override API base if site is overridden
    global DEFAULT_API_BASE, NEURON_SELECT
    DEFAULT_API_BASE = args.site.rstrip("/") + "/api"
    NEURON_SELECT = DEFAULT_API_BASE + "/neuron/select"

    total_meta = 0
    total_swc = 0
    total_fail = 0

    for page_idx, page_json in iter_pages(
        sess, args.query,
        page_size=args.page_size,
        start_page=args.start_page,
        max_pages=args.max_pages,
        verify=verify,
        timeout=args.timeout,
    ):
        neurons = extract_neuron_list(page_json)
        if not neurons:
            print(f"[page {page_idx}] No results; stopping.")
            break

        write_jsonl(meta_path, neurons)
        total_meta += len(neurons)
        print(f"[page {page_idx}] neurons={len(neurons)}  total_meta={total_meta}")

        if args.metadata_only:
            continue

        for meta in neurons:
            neuron_id = str(meta.get("neuron_id", "")).strip()
            neuron_name = str(meta.get("neuron_name", "")).strip()
            archive = str(meta.get("archive", "")).strip()

            if not neuron_name or not archive:
                total_fail += 1
                write_jsonl(log_path, [{
                    "status": "skip",
                    "reason": "missing neuron_name or archive",
                    "neuron_id": neuron_id,
                    "neuron_name": neuron_name,
                    "archive": archive,
                }])
                continue

            # Determine SWC URL
            swc_url = None
            if args.find:
                swc_url = find_standardized_swc_url_by_scrape(
                    sess, args.site.rstrip("/"), neuron_name,
                    verify=verify, timeout=args.timeout
                )
            if not swc_url:
                swc_url = build_standardized_swc_url_pattern(args.site.rstrip("/"), archive, neuron_name)

            # Output path
            fname = safe_filename(f"{neuron_id}_{neuron_name}.CNG.swc" if neuron_id else f"{neuron_name}.CNG.swc")
            out_path = os.path.join(out_dir, "swc", fname)
            if os.path.exists(out_path) and not args.overwrite:
                write_jsonl(log_path, [{
                    "status": "exists",
                    "neuron_id": neuron_id,
                    "neuron_name": neuron_name,
                    "url": swc_url,
                    "path": out_path,
                }])
                continue

            try:
                download_file(sess, swc_url, out_path, verify=verify, timeout=args.timeout)
                total_swc += 1
                write_jsonl(log_path, [{
                    "status": "ok",
                    "neuron_id": neuron_id,
                    "neuron_name": neuron_name,
                    "url": swc_url,
                    "path": out_path,
                }])
            except Exception as e:
                total_fail += 1
                write_jsonl(log_path, [{
                    "status": "fail",
                    "neuron_id": neuron_id,
                    "neuron_name": neuron_name,
                    "url": swc_url,
                    "error": str(e),
                }])
                print(f"  ! FAIL id={neuron_id or '?'} name={neuron_name} url={swc_url} err={e}", file=sys.stderr)

    print("\nDone.")
    print(f"metadata.jsonl: {meta_path} (records written: {total_meta})")
    if not args.metadata_only:
        print(f"SWC downloaded: {total_swc} -> {os.path.join(out_dir, 'swc')}")
        print(f"failures: {total_fail}")
        print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

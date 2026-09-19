from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "papers_manifest.csv"
DOCS_DIR = ROOT / "docs"

# MDPI's public static-resource host is normally much friendlier to scripted PDF downloads
# than the article site's /pdf endpoint, which can return an anti-bot HTML page.
MDPI_ISSN_TO_SLUG = {
    "2077-1312": "jmse",          # Journal of Marine Science and Engineering
    "1424-8220": "sensors",
    "2076-3417": "applsci",
    "2079-9292": "electronics",
    "2072-4292": "remotesensing",
}


def safe_name(text: str, max_len: int = 110) -> str:
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len].rstrip("_.")


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mdpi_static_pdf_url(landing_url: str) -> str | None:
    """Convert an MDPI article URL into its mdpi-res static PDF URL.

    Example:
      https://www.mdpi.com/2077-1312/11/4/861
      -> https://mdpi-res.com/d_attachment/jmse/jmse-11-00861/article_deploy/jmse-11-00861.pdf
    """
    try:
        p = urllib.parse.urlparse(landing_url)
        parts = [x for x in p.path.split("/") if x]
        if len(parts) < 4:
            return None

        issn, volume, _issue, article = parts[:4]
        slug = MDPI_ISSN_TO_SLUG.get(issn)
        if not slug or not volume.isdigit() or not article.isdigit():
            return None

        base = f"{slug}-{int(volume):02d}-{int(article):05d}"
        return (
            f"https://mdpi-res.com/d_attachment/{slug}/{base}/"
            f"article_deploy/{base}.pdf"
        )
    except Exception:
        return None


def candidate_urls(pdf_url: str, landing_url: str) -> list[str]:
    """Return download candidates in best-first order."""
    urls: list[str] = []

    if "mdpi.com/" in landing_url.lower():
        static_url = mdpi_static_pdf_url(landing_url)
        if static_url:
            urls.append(static_url)

    urls.append(pdf_url)

    # Some MDPI setups accept the query-string variant even when /pdf alone is challenged.
    if "mdpi.com/" in pdf_url.lower() and "?" not in pdf_url:
        urls.append(pdf_url + "?download=1")

    # Deduplicate while preserving order.
    out: list[str] = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_pdf(url: str, referer: str | None = None, timeout: int = 60) -> tuple[bool, str, bytes | None]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    if referer:
        headers["Referer"] = referer

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            ctype = (resp.headers.get("Content-Type") or "").lower()
            final_url = resp.geturl()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}", None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, f"network error: {e}", None

    if not data.startswith(b"%PDF"):
        return (
            False,
            f"not a PDF (content-type={ctype or 'unknown'}, bytes={len(data)}, final={final_url})",
            None,
        )

    return True, f"{len(data) / 1024 / 1024:.2f} MB", data


def download_pdf(pdf_url: str, landing_url: str, dest: Path, timeout: int = 60) -> tuple[bool, str]:
    attempts = []
    urls = candidate_urls(pdf_url, landing_url)

    for idx, url in enumerate(urls, start=1):
        print(f"             TRY   {idx}/{len(urls)} {url}")
        ok, msg, data = fetch_pdf(url, referer=landing_url, timeout=timeout)
        if ok and data is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True, f"{msg}; source={url}"

        attempts.append(f"{url} -> {msg}")
        # Be polite and avoid hammering the same host when falling back.
        if idx < len(urls):
            time.sleep(0.8)

    return False, " | ".join(attempts)


def main():
    parser = argparse.ArgumentParser(
        description="Download open-access underwater acoustic papers for the RAG knowledge base."
    )
    parser.add_argument("--limit", type=int, default=20, help="How many papers to attempt (default: 20)")
    parser.add_argument("--start", type=int, default=1, help="Start from manifest row id (default: 1)")
    parser.add_argument("--sleep", type=float, default=1.5, help="Delay between papers in seconds")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Redownload existing PDFs")
    args = parser.parse_args()

    rows = [r for r in read_manifest() if int(r["id"]) >= args.start][: max(args.limit, 0)]
    if not rows:
        print("No papers selected.")
        return

    success = 0
    failed = []

    for i, row in enumerate(rows, start=1):
        paper_id = int(row["id"])
        category = row["category"]
        title = row["title"]
        year = row["year"]
        pdf_url = row["pdf_url"]
        landing_url = row["landing_url"]

        filename = f"{paper_id:02d}_{year}_{safe_name(title)}.pdf"
        dest = DOCS_DIR / category / filename

        if dest.exists() and not args.overwrite:
            print(f"[{i}/{len(rows)}] SKIP  {dest.relative_to(ROOT)}")
            success += 1
            continue

        print(f"[{i}/{len(rows)}] GET   {title}")
        ok, msg = download_pdf(pdf_url, landing_url, dest, timeout=args.timeout)
        if ok:
            print(f"             OK    {dest.relative_to(ROOT)} ({msg})")
            success += 1
        else:
            print(f"             FAIL  {msg}")
            print(f"             PAGE  {landing_url}")
            failed.append((paper_id, title, landing_url, msg))

        if i < len(rows):
            time.sleep(max(args.sleep, 0))

    print("\n=== Download summary ===")
    print(f"Success/Already present: {success}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nStill failed (open the landing page manually if needed):")
        for paper_id, title, page, msg in failed:
            print(f"- [{paper_id:02d}] {title}\n  {page}\n  reason: {msg}")


if __name__ == "__main__":
    main()

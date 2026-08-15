#!/usr/bin/env python3
"""Download FHWA NBI delimited files for one state and year range.

This is intentionally only a downloader. Historical NBI schemas vary; the
InfraRL preprocessing pipeline should be used to harmonize fields before
training/evaluation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def download(url: str, target: Path, retries: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "InfraAWM/0.1"})
            with urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(target)
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(2 ** attempt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="CA", help="two-letter state code")
    ap.add_argument("--start-year", type=int, default=1992)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--output-dir", default="data/fhwa_nbi")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    state = args.state.upper()
    out = Path(args.output_dir)
    failures = []
    for year in range(args.start_year, args.end_year + 1):
        filename = f"{state}{str(year)[2:]}.txt"
        url = f"https://www.fhwa.dot.gov/bridge/nbi/{year}/delimited/{filename}"
        target = out / f"{state}{year}.txt"
        if target.exists() and target.stat().st_size > 0 and not args.force:
            print(f"skip {target}")
            continue
        print(f"download {year}: {url}")
        try:
            download(url, target)
        except RuntimeError as exc:
            print(f"ERROR {exc}")
            failures.append(year)

    if failures:
        raise SystemExit(f"failed years: {failures}")
    print(f"done: {out}")


if __name__ == "__main__":
    main()

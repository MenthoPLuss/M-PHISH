"""Build a fresher hostname dataset: PhiUSIIL plus current public lists. Never imported by the API.

Sources (downloaded into <output>/raw unless already there; parsed as plain text only,
nothing in them is visited):
  phishing     Phishing.Database active phishing hostnames, MIT licence
               https://github.com/Phishing-Database/Phishing.Database
  legitimate   Chrome UX Report top origins (sites real Chrome users load and search
               engines may index), CC BY 4.0 by Google, monthly snapshot via
               https://github.com/zakird/crux-top-lists

Rules, all recorded in <output>/combined.sources.json:
  * PhiUSIIL rows are copied unchanged, so its hostnames, labels and conflict removals
    match earlier runs exactly; new sources only add hostnames PhiUSIIL does not contain.
  * A hostname listed by both new sources is dropped (ambiguous).
  * A phishing entry that is a bare registered domain whose domain also serves a CrUX
    website is dropped: usually a compromised legitimate site, which a hostname model
    cannot tell apart and which would teach it to distrust real sites.
  * A seeded random sample of each new source is kept, to bound training time.

Example (then train with the printed command):
    python -m backend.app.services.ml.hostname_feeds \
        --phiusiil data/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv --output data/feeds/2026-11-24
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sys
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

from ..observations import registrable_domain
from .train_url_cnn import hostname_of

USER_AGENT = "m-phish-hostname-dataset-builder/1.0"
MAX_DOWNLOAD_BYTES = 200_000_000
PHISHING_DATABASE = {
    "url": "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/"
    "phishing-domains-ACTIVE.txt",
    "md5_url": "https://raw.githubusercontent.com/Phishing-Database/checksums/refs/heads/master/"
    "phishing-domains-ACTIVE.txt.md5",
    "license": "MIT (Phishing.Database contributors)",
    "citation": "Phishing.Database, https://github.com/Phishing-Database/Phishing.Database",
}
CRUX = {
    "url": "https://raw.githubusercontent.com/zakird/crux-top-lists/main/data/global/{month}.csv.gz",
    "license": "CC BY 4.0 (Chrome UX Report by Google)",
    "citation": "Chrome UX Report (CrUX), Google, https://developer.chrome.com/docs/crux; "
    "snapshot from https://github.com/zakird/crux-top-lists",
}


_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")


def valid_hostname(host: str) -> bool:
    """A DNS name of letters, digits and hyphens with a non-numeric TLD (IPs excluded)."""
    labels = host.split(".")
    return (
        len(host) <= 253
        and len(labels) >= 2
        and all(_LABEL.match(label) for label in labels)
        and not labels[-1].isdigit()
    )


def latest_crux_month(today: date) -> str:
    """CrUX publishes each month around the second Tuesday of the next month."""
    year, month = today.year, today.month - (1 if today.day >= 15 else 2)
    while month < 1:
        year, month = year - 1, month + 12
    return f"{year}{month:02d}"


def download(url: str, target: Path) -> Path:
    if target.exists():
        print(f"Using existing {target}")
        return target
    print(f"Downloading {url} ...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 -- fixed https URLs
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def phishing_hostnames(path: Path) -> dict[str, str]:
    """Hostname -> raw line, for a list with one hostname or URL per line."""
    found: dict[str, str] = {}
    for line in path.read_text("utf-8", "replace").splitlines():
        raw = line.strip()
        if raw and not raw.startswith("#"):
            host = hostname_of(raw)
            if valid_hostname(host):
                found.setdefault(host, raw)
    return found


def crux_hostnames(path: Path) -> dict[str, str]:
    """Hostname -> origin, from a CrUX top list (columns origin,rank)."""
    found: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "origin" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no origin column; found {reader.fieldnames}")
        for row in reader:
            origin = (row.get("origin") or "").strip()
            host = hostname_of(origin)
            if valid_hostname(host):
                found.setdefault(host, origin)
    return found


def phiusiil_rows(path: Path, url_column: str = "URL", label_column: str = "label"):
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            yield row[url_column], row[label_column]


def build(
    phiusiil: Path,
    phishing_file: Path,
    crux_file: Path,
    output: Path,
    *,
    phishing_sample: int,
    legit_sample: int,
    seed: int,
    sources: dict,
    legit_label: str = "1",
    phishing_label: str = "0",
) -> dict:
    known = {hostname_of(url) for url, _ in phiusiil_rows(phiusiil)} - {""}
    phishing, legit = phishing_hostnames(phishing_file), crux_hostnames(crux_file)
    both = phishing.keys() & legit.keys()
    crux_domains = {registrable_domain(h) or h for h in legit}
    compromised = {
        h for h in phishing.keys() - both if (registrable_domain(h) or h) == h and h in crux_domains
    }
    phishing_pool = sorted(phishing.keys() - known - both - compromised)
    legit_pool = sorted(legit.keys() - known - both)
    rng = random.Random(seed)
    new_phishing = rng.sample(phishing_pool, min(phishing_sample, len(phishing_pool)))
    new_legit = rng.sample(legit_pool, min(legit_sample, len(legit_pool)))

    output.mkdir(parents=True, exist_ok=True)
    combined = output / "combined.csv"
    with combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["URL", "label", "source"])
        for url, label in phiusiil_rows(phiusiil):
            writer.writerow([url, label, "phiusiil"])
        for host in sorted(new_phishing):
            writer.writerow([phishing[host], phishing_label, "phishing_database"])
        for host in sorted(new_legit):
            writer.writerow([legit[host], legit_label, "crux"])
    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "label_mapping": {"legitimate": legit_label, "phishing": phishing_label},
        "sources": sources,
        "counts": {
            "phiusiil_hostnames": len(known),
            "phishing_database_hostnames": len(phishing),
            "crux_hostnames": len(legit),
            "listed_by_both_dropped": len(both),
            "phishing_bare_domain_serving_crux_site_dropped": len(compromised),
            "phishing_already_in_phiusiil": len(phishing.keys() & known),
            "crux_already_in_phiusiil": len(legit.keys() & known),
            "phishing_available": len(phishing_pool),
            "crux_available": len(legit_pool),
            "phishing_added": len(new_phishing),
            "crux_added": len(new_legit),
        },
        "combined_csv_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
    }
    combined.with_suffix(".sources.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--phiusiil", required=True, help="PhiUSIIL CSV (label 1 = legitimate)")
    parser.add_argument("--output", required=True, help="dataset folder, e.g. data/feeds/<date>")
    parser.add_argument("--crux-month", help="CrUX snapshot YYYYMM (default: latest published)")
    parser.add_argument("--phishing-sample", type=int, default=100_000)
    parser.add_argument("--legit-sample", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.phishing_sample < 1 or args.legit_sample < 1:
        parser.error("sample sizes must be positive")
    month = args.crux_month or latest_crux_month(date.today())
    if len(month) != 6 or not month.isdigit():
        parser.error("--crux-month must look like 202610")
    phiusiil, output = Path(args.phiusiil), Path(args.output)
    if not phiusiil.is_file():
        parser.error(f"{phiusiil} not found (train_url_cnn --download-phiusiil fetches it)")
    if (output / "combined.csv").exists():
        parser.error(f"{output / 'combined.csv'} already exists; use a new --output folder")
    raw = output / "raw"

    phishing_file = download(PHISHING_DATABASE["url"], raw / "phishing-domains-ACTIVE.txt")
    md5_file = download(PHISHING_DATABASE["md5_url"], raw / "phishing-domains-ACTIVE.txt.md5")
    published = md5_file.read_text().split()[0].lower() if md5_file.read_text().split() else ""
    actual = hashlib.md5(phishing_file.read_bytes(), usedforsecurity=False).hexdigest()
    if published != actual:
        print("Warning: phishing list MD5 differs from the published checksum (mirror lag?)")
    crux_file = download(CRUX["url"].format(month=month), raw / f"crux-{month}.csv.gz")

    def described(meta, path, **extra):
        return (
            meta
            | extra
            | {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )

    sources = {
        "phishing_database": described(
            {k: v for k, v in PHISHING_DATABASE.items() if k != "md5_url"},
            phishing_file,
            md5_matches_published=published == actual,
        ),
        "crux": described(CRUX | {"url": CRUX["url"].format(month=month)}, crux_file, month=month),
    }
    report = build(
        phiusiil,
        phishing_file,
        crux_file,
        output,
        phishing_sample=args.phishing_sample,
        legit_sample=args.legit_sample,
        seed=args.seed,
        sources=sources,
    )
    print(json.dumps(report["counts"], indent=2))
    print(
        "\nTrain with (keeps every PhiUSIIL hostname in its earlier split):\n"
        "  python -m backend.app.services.ml.train_url_cnn --ensemble 3 "
        f"--csv {output / 'combined.csv'} --reuse-splits runs/<earlier run>/splits.json "
        "--output runs/<new run>"
    )
    return report


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

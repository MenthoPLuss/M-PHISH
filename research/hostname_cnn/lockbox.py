"""The locked final test (Phase 11 only).

Part 1, built now: a bank of legitimate hostnames from CrUX 2026-10 that share no hostname,
registered domain or name stem with any hostname in the training dataset. It is reserved
*before* any new training data is drawn, so later datasets exclude it.

Parts 2 and 3 (temporal holdouts) are built from snapshots taken after 2026-11-24:
hostnames new to the Phishing.Database ACTIVE list, and sites new in the next CrUX month.

Scoring the lockbox requires ``M_PHISH_LOCKBOX_UNLOCK=phase-11``; nothing in the screening
or tuning code reads these files.

    python -m research.hostname_cnn.lockbox build-legit-bank
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from backend.app.services.ml import hostname_feeds as feeds
from backend.app.services.observations import _suffix_extractor, registrable_domain

from .common import FEED, ROOT, load_dataset, sha256_file

LOCKBOX = ROOT / "data" / "feeds" / "lockbox"
MANIFEST = LOCKBOX / "manifest.json"
BANK = LOCKBOX / "legit_bank_crux_202610.txt"
UNLOCK = "phase-11"


def keys(host: str) -> tuple[str, str]:
    """(registered domain, name stem) without caching; used for bulk passes over lists."""
    return (registrable_domain(host) or host), (_suffix_extractor()(host).domain or host)


def reserved() -> dict:
    """Hostnames, registered domains and stems that later datasets must never use."""
    if not MANIFEST.exists():
        return {"hosts": set(), "registrable": set(), "stems": set()}
    hosts = set(BANK.read_text(encoding="utf-8").split())
    pairs = [keys(h) for h in hosts]
    return {
        "hosts": hosts,
        "registrable": {r for r, _ in pairs},
        "stems": {s for _, s in pairs},
    }


def build_legit_bank(size: int, seed: int) -> dict:
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} exists; the lockbox is built once and never rebuilt")
    data = load_dataset()
    used_hosts = set(data.hosts)
    used = [keys(h) for h in data.hosts]
    phishing = feeds.phishing_hostnames(FEED / "raw" / "phishing-domains-ACTIVE.txt")
    blocked_registrable = {r for r, _ in used} | {keys(h)[0] for h in phishing}
    used_stems = {s for _, s in used}
    crux = sorted(feeds.crux_hostnames(FEED / "raw" / "crux-202610.csv.gz"))
    by_domain: dict[str, list[str]] = {}
    candidates = 0
    for h in crux:
        if h in used_hosts or h in phishing:
            continue
        registrable, stem = keys(h)
        if registrable in blocked_registrable or stem in used_stems:
            continue
        candidates += 1
        # Sample whole registered domains so none straddles the bank and later training data.
        by_domain.setdefault(registrable, []).append(h)
    domains = sorted(by_domain)
    random.Random(seed).shuffle(domains)
    bank: list[str] = []
    for domain in domains:
        if len(bank) >= size:
            break
        bank.extend(by_domain[domain])
    LOCKBOX.mkdir(parents=True, exist_ok=True)
    BANK.write_text("\n".join(sorted(bank)) + "\n", encoding="utf-8")
    manifest = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "legit_bank": {
            "file": BANK.name,
            "sha256": sha256_file(BANK),
            "hostnames": len(bank),
            "source": "CrUX 2026-10 (CC BY 4.0), never used in any training dataset",
            "candidates": candidates,
            "subdomain_hosts": sum(keys(h)[0] != h for h in bank),
            "hosting_platform_hosts": sum(_suffix_extractor()(h).is_private for h in bank),
        },
        "temporal_reference_snapshots": {
            "phishing_database_active_2026-11-24_sha256": sha256_file(
                FEED / "raw" / "phishing-domains-ACTIVE.txt"
            ),
            "crux_202610_sha256": sha256_file(FEED / "raw" / "crux-202610.csv.gz"),
            "plan": "Phishing holdout = hostnames on an ACTIVE-list snapshot taken on or after "
            "2026-12-08 that are absent from the 2026-11-24 list; legitimate holdout = CrUX "
            "202611 sites absent from 202610. Both exclude every training registered domain "
            "and stem.",
        },
        "unlock": f"set M_PHISH_LOCKBOX_UNLOCK={UNLOCK} (Phase 11 only)",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    return manifest


def open_lockbox() -> list[str]:
    """The only reader of the bank; refuses unless Phase 11 is declared."""
    if os.environ.get("M_PHISH_LOCKBOX_UNLOCK") != UNLOCK:
        raise PermissionError("The lockbox is sealed until Phase 11 (see lockbox.py)")
    manifest = json.loads(MANIFEST.read_text())
    if sha256_file(BANK) != manifest["legit_bank"]["sha256"]:
        raise ValueError("Lockbox bank checksum mismatch")
    return BANK.read_text(encoding="utf-8").split()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the sealed final-test material.")
    parser.add_argument("command", choices=["build-legit-bank"])
    parser.add_argument("--size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    manifest = build_legit_bank(args.size, args.seed)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

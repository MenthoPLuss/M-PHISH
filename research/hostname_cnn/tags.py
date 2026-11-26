"""H8: tag phishing labels that probably sit on legitimate domains (external evidence only).

A phishing hostname is tagged when the hostname itself is a CrUX 2026-10 site or a Tranco Q2464
top-1M domain, or its public suffix is institutional (gov, edu, mil, ac, go, gob, gouv, govt at
any level). Subdomains of popular shared-hosting or dynamic-DNS domains are NOT tagged: there
the attacker chose the label, so the name carries signal. Model scores are never used. Writes:

* ``data/feeds/research/h8-tags.json``: tagged hostnames per split and the rule that hit;
* ``data/feeds/research/h8-weights-<w>.json``: training weights (tagged rows get ``w``).

    python -m research.hostname_cnn.tags --weight 0.25
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile

from backend.app.services.ml import hostname_feeds as feeds
from backend.app.services.ml.train_url_cnn import hostname_of
from backend.app.services.observations import _suffix_extractor

from .common import FEED, ROOT, load_dataset

OUT = ROOT / "data" / "feeds" / "research"
INSTITUTIONAL = {"gov", "edu", "mil", "ac", "go", "gob", "gouv", "govt"}


def popular_hosts() -> dict[str, set[str]]:
    crux = set(feeds.crux_hostnames(FEED / "raw" / "crux-202610.csv.gz"))
    tranco: set[str] = set()
    with zipfile.ZipFile(FEED / "raw" / "tranco-Q2464.csv.zip") as archive:
        rows = csv.reader(io.TextIOWrapper(archive.open("top-1m.csv"), "utf-8"))
        for _, domain in rows:
            host = hostname_of(domain)
            if host:
                tranco.add(host)
    return {"crux": crux, "tranco": tranco}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weight", type=float, default=0.25)
    args = parser.parse_args(argv)
    data = load_dataset()
    popular = popular_hosts()
    extract = _suffix_extractor()
    tagged: dict[str, dict[str, str]] = {}
    for host, label, split in zip(data.hosts, data.y, data.split, strict=True):
        if label != 1:
            continue
        suffix_labels = set(extract(host).suffix.split("."))
        rule = (
            "institutional suffix"
            if suffix_labels & INSTITUTIONAL
            else "is a CrUX site"
            if host in popular["crux"]
            else "is a Tranco domain"
            if host in popular["tranco"]
            else None
        )
        if rule:
            tagged.setdefault(split, {})[host] = rule
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        split: {
            "tagged": len(hosts),
            "phishing_in_split": int(((data.split == split) & (data.y == 1)).sum()),
            "by_rule": {r: sum(v == r for v in hosts.values()) for r in set(hosts.values())},
        }
        for split, hosts in tagged.items()
    }
    (OUT / "h8-tags.json").write_text(json.dumps({"summary": summary, "tagged": tagged}, indent=1))
    weights = dict.fromkeys(tagged.get("train", {}), args.weight)
    (OUT / f"h8-weights-{args.weight:g}.json").write_text(json.dumps(weights))
    print(json.dumps(summary, indent=1))
    print("examples:", list(tagged.get("validation", {}).items())[:20])


if __name__ == "__main__":
    main()

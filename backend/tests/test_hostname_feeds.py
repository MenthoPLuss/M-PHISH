"""Fresh-data dataset builder and split reuse; no network access."""

import csv
import gzip
import json
from datetime import date

import numpy as np
import pytest

from backend.app.services.ml import hostname_feeds as feeds
from backend.app.services.ml import train_url_cnn as training
from backend.app.services.ml.offline import canonical_hash
from backend.app.services.ml.offline import split_indices as grouped_splits
from backend.app.services.observations import registrable_domain


def write_inputs(tmp_path):
    phiusiil = tmp_path / "phiusiil.csv"
    phiusiil.write_text(
        "URL,label,extra\nhttps://www.known.com/a,1,x\nhttp://phish.known-bad.com/x,0,y\n"
        "conflict.com,1,z\nconflict.com,0,z\n"
    )
    phishing = tmp_path / "phishing.txt"
    phishing.write_text(
        "# comment\nknown.com\nboth.example\nhacked-site.com\npaypal-login.weebly.com\n"
        "secure-verify.xyz\nnot a host!!\n\n"
    )
    crux = tmp_path / "crux.csv.gz"
    with gzip.open(crux, "wt", encoding="utf-8") as handle:
        handle.write(
            "origin,rank\nhttps://www.wikipedia.org,1000\nhttps://both.example,1000000\n"
            "https://shop.hacked-site.com,500000\nhttps://www.known.com,1000\n"
        )
    return phiusiil, phishing, crux


def test_build_keeps_phiusiil_rows_and_cleans_new_labels(tmp_path):
    phiusiil, phishing, crux = write_inputs(tmp_path)
    output = tmp_path / "dataset"
    report = feeds.build(
        phiusiil,
        phishing,
        crux,
        output,
        phishing_sample=10,
        legit_sample=10,
        seed=1,
        sources={"a": {"license": "MIT"}, "b": {"license": "CC BY 4.0"}},
    )
    counts = report["counts"]
    assert counts["listed_by_both_dropped"] == 2  # both.example, known.com
    assert counts["phishing_bare_domain_serving_crux_site_dropped"] == 1  # hacked-site.com
    assert counts["phishing_already_in_phiusiil"] == 1  # known.com
    assert counts["phishing_added"] == 2 and counts["crux_added"] == 2
    with (output / "combined.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # PhiUSIIL rows come first and unchanged, including its conflicting pair.
    assert [(r["URL"], r["label"]) for r in rows[:4]] == [
        ("https://www.known.com/a", "1"),
        ("http://phish.known-bad.com/x", "0"),
        ("conflict.com", "1"),
        ("conflict.com", "0"),
    ]
    added = {(r["URL"], r["label"], r["source"]) for r in rows[4:]}
    assert added == {
        ("paypal-login.weebly.com", "0", "phishing_database"),
        ("secure-verify.xyz", "0", "phishing_database"),
        ("https://www.wikipedia.org", "1", "crux"),
        ("https://shop.hacked-site.com", "1", "crux"),
    }
    sidecar = json.loads((output / "combined.sources.json").read_text())
    assert sidecar["counts"] == counts and sidecar["seed"] == 1
    hosts, labels = training.load_csv(output / "combined.csv", "URL", "label", "1", "0")
    assert dict(zip(hosts, labels.tolist(), strict=True)) == {
        "known.com": 0,
        "phish.known-bad.com": 1,
        "paypal-login.weebly.com": 1,
        "secure-verify.xyz": 1,
        "wikipedia.org": 0,
        "shop.hacked-site.com": 0,
    }


def test_sampling_is_seeded(tmp_path):
    phiusiil, phishing, crux = write_inputs(tmp_path)

    def sampled(name, seed):
        out = tmp_path / name
        feeds.build(
            phiusiil, phishing, crux, out, phishing_sample=1, legit_sample=1, seed=seed, sources={}
        )
        return (out / "combined.csv").read_text()

    assert sampled("a", 3) == sampled("b", 3)


@pytest.mark.parametrize(
    ("today", "month"),
    [(date(2026, 11, 24), "202610"), (date(2026, 11, 3), "202609"), (date(2026, 1, 5), "202511")],
)
def test_latest_crux_month(today, month):
    assert feeds.latest_crux_month(today) == month


def test_reuse_splits_keeps_assignments_and_domain_groups():
    hosts, y = training.synthetic_hostnames(600, 3)
    groups = np.array([registrable_domain(h) or h for h in hosts])
    earlier = grouped_splits(y[:400], groups[:400], 42)
    previous = {name: [canonical_hash(hosts[i]) for i in idx] for name, idx in earlier.items()}
    # A new subdomain of a domain that was in the earlier test split.
    test_group = groups[earlier["test"][0]]
    new_host = "zz9-login." + test_group
    all_hosts = [*hosts, new_host]
    all_y = np.append(y, 1)
    all_groups = np.append(groups, registrable_domain(new_host) or new_host)
    splits, kept = training.reuse_splits(all_hosts, all_y, all_groups, previous, 42)
    assert kept == 400
    where = {int(i): name for name, idx in splits.items() for i in idx}
    assert len(where) == len(all_hosts)  # disjoint and complete
    for name, idx in earlier.items():
        assert all(where[int(i)] == name for i in idx)
    assert where[len(all_hosts) - 1] == "test"
    group_split = {}
    for i, group in enumerate(all_groups):
        assert group_split.setdefault(group, where[i]) == where[i]  # no domain spans splits
    with pytest.raises(ValueError, match="needs the splits"):
        training.reuse_splits(all_hosts, all_y, all_groups, {"train": []}, 42)


def test_training_cli_reuses_an_earlier_runs_splits(tmp_path):
    pytest.importorskip("torch")
    small = ["--synthetic-smoke", "--epochs", "1", "--channels", "6", "--hidden", "8"]
    small += ["--embedding-dim", "4", "--dilations", "1", "--threads", "2", "--device", "cpu"]
    first, second = tmp_path / "first", tmp_path / "second"
    training.main([*small, "--max-rows", "300", "--output", str(first)])
    training.main(
        [*small, "--max-rows", "500", "--output", str(second)]
        + ["--reuse-splits", str(first / "splits.json")]
    )
    before = json.loads((first / "splits.json").read_text())
    after = json.loads((second / "splits.json").read_text())
    old = {digest: name for name, digests in before.items() for digest in digests}
    new = {digest: name for name, digests in after.items() for digest in digests}
    shared = old.keys() & new.keys()
    assert shared and all(old[d] == new[d] for d in shared)
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["reused_splits"]["kept_hostnames"] == len(shared)


@pytest.mark.parametrize(
    ("host", "valid"),
    [
        ("example.com", True),
        ("a.b.github.io", True),
        ("xn--80ak6aa92e.com", True),
        ("not a host!!", False),
        ("a..b", False),
        ("-x.com", False),
        ("1.2.3.4", False),
        ("localhost", False),
        ("_dmarc.example.com", False),
        ("a" * 64 + ".com", False),
    ],
)
def test_only_valid_dns_names_come_from_new_sources(host, valid):
    assert feeds.valid_hostname(host) is valid

"""Phase 1 audit of the deployed hostname CNN (runs/cnn-v3-fresh) and its predecessor
(runs/cnn-v3-ens3). Read-only: it never writes inside the M-PHISH repository.

Only the VALIDATION split, a 30,000-hostname TRAINING sample and synthetic probes derived
from validation hostnames are scored. The test split stays locked.

Run from the M-PHISH folder:
    .venv\\Scripts\\python.exe ..\\cnn-research\\phase1\\scripts\\phase1_audit.py
Writes ..\\cnn-research\\phase1\\results\\phase1_audit.json
"""

import csv
import hashlib
import json
import math
import os
import string
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parents[1] / "results"
FEEDS = ROOT / "data" / "feeds" / "2026-11-24"
RUNS = {"fresh": ROOT / "runs" / "cnn-v3-fresh", "ens3": ROOT / "runs" / "cnn-v3-ens3"}
FPRS = (0.001, 0.0025, 0.005, 0.01)
PROCS = int(os.environ.get("AUDIT_PROCS", "14"))
_models: dict = {}


def _init():
    from backend.app.services.ml.url_cnn import UrlCnnModel

    for name, run in RUNS.items():
        path = run / "url_cnn.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _models[name] = UrlCnnModel(str(path), digest, allow_ungated=True)


def _score(host):
    """Per-member raw logits for every model: {model: [m0, m1, m2]}."""
    return {
        name: [model.member_logit(i, host) for i in range(len(model.members))]
        for name, model in _models.items()
    }


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def recall_at_fpr(y, s, fpr):
    """Recall when at most floor(fpr * negatives) negatives score above the cut."""
    negatives = np.sort(s[y == 0])[::-1]
    allowed = int(math.floor(fpr * len(negatives)))
    cut = negatives[allowed] if allowed < len(negatives) else -np.inf
    return float((s[y == 1] > cut).mean()), allowed


def ece(y, p, bins=15):
    index = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = index == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def summary(y, s, p, flag, *, bootstrap=0, seed=0):
    from sklearn.metrics import average_precision_score, roc_auc_score

    row = {"n": int(len(y)), "phishing": int(y.sum()), "legitimate": int((y == 0).sum())}
    if row["phishing"]:
        row["recall_at_threshold"] = float(flag[y == 1].mean())
    if row["legitimate"]:
        row["fpr_at_threshold"] = float(flag[y == 0].mean())
    row["brier"] = float(np.mean((p - y) ** 2))
    row["ece"] = ece(y, p)
    row["mean_probability"] = float(p.mean())
    row["observed_rate"] = float(y.mean())
    if min(row["phishing"], row["legitimate"]) >= 20:
        row["roc_auc"] = float(roc_auc_score(y, s))
        row["pr_auc"] = float(average_precision_score(y, s))
        row["pauc_fpr_1pct_mcclish"] = float(roc_auc_score(y, s, max_fpr=0.01))
        for f in FPRS:
            value, allowed = recall_at_fpr(y, s, f)
            row[f"recall_at_{f:g}_fpr"] = value
            row[f"false_positives_allowed_at_{f:g}"] = allowed
        if bootstrap:
            rng = np.random.default_rng(seed)
            draws = {f: [] for f in FPRS}
            auc_draws = []
            for _ in range(bootstrap):
                i = rng.integers(0, len(y), len(y))
                for f in FPRS:
                    draws[f].append(recall_at_fpr(y[i], s[i], f)[0])
                auc_draws.append(roc_auc_score(y[i], s[i]))
            for f in FPRS:
                row[f"recall_at_{f:g}_fpr_ci95"] = [
                    float(np.percentile(draws[f], 2.5)),
                    float(np.percentile(draws[f], 97.5)),
                ]
            row["roc_auc_ci95"] = [
                float(np.percentile(auc_draws, 2.5)),
                float(np.percentile(auc_draws, 97.5)),
            ]
    return row


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from backend.app.services.ml import hostname_feeds as feeds
    from backend.app.services.ml.offline import canonical_hash
    from backend.app.services.ml.train_url_cnn import hostname_of, load_csv
    from backend.app.services.ml.url_cnn import ALPHABET, char_roles
    from backend.app.services.observations import _suffix_extractor, registrable_domain

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "scope": "validation split only"}
    hosts, y = load_csv(FEEDS / "combined.csv", "URL", "label", "1", "0")
    index_of = {h: i for i, h in enumerate(hosts)}
    source = {}
    with (FEEDS / "combined.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            host = hostname_of(row["URL"])
            if host:
                source.setdefault(host, row["source"])
    splits = json.loads((RUNS["fresh"] / "splits.json").read_text())
    split_of = {d: name for name, digests in splits.items() for d in digests}
    where = np.array([split_of[canonical_hash(h)] for h in hosts])
    src = np.array([source[h] for h in hosts])
    report["split_composition"] = {
        name: {
            s: {
                "phishing": int(((where == name) & (src == s) & (y == 1)).sum()),
                "legitimate": int(((where == name) & (src == s) & (y == 0)).sum()),
            }
            for s in ("phiusiil", "phishing_database", "crux")
        }
        for name in ("train", "calibration", "validation", "test")
    }
    print("loaded", len(hosts), "hosts", round(time.time() - started), "s", flush=True)

    # ---------------------------------------------------------------- descriptors
    extract = _suffix_extractor()

    def tld_class(h):
        tld = h.rsplit(".", 1)[-1]
        if tld == "com":
            return "com"
        if tld in ("net", "org"):
            return "net/org"
        if tld.startswith("xn--"):
            return "idn tld"
        if len(tld) == 2:
            return "ccTLD"
        return "other gTLD"

    def describe(h):
        r = extract(h)
        reg = registrable_domain(h) or h
        length = len(h)
        return {
            "shape": "bare" if reg == h else "subdomain",
            "platform_suffix": "hosting-platform suffix" if r.is_private else "ICANN suffix",
            "suffix_known": "known suffix" if r.suffix else "unknown suffix",
            "length": next(
                f"{lo}-{hi}"
                for lo, hi in ((1, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 253))
                if lo <= length <= hi
            ),
            "labels": str(min(h.count(".") + 1, 5)) + ("+" if h.count(".") + 1 >= 5 else ""),
            "digit": "has digit" if any(c.isdigit() for c in h) else "no digit",
            "hyphen": "has hyphen" if "-" in h else "no hyphen",
            "idn": "punycode (xn--)" if "xn--" in h else "ascii",
            "tld": tld_class(h),
            "stem": r.domain,
            "suffix": r.suffix,
            "unknown_chars": any(c not in ALPHABET for c in h),
            "all_name_roles": len(set(char_roles(h))) == 1,
        }

    val = np.flatnonzero(where == "validation")
    train = np.flatnonzero(where == "train")
    rng = np.random.default_rng(0)
    train_sample = np.sort(rng.choice(train, 30_000, replace=False))
    info = {hosts[i]: describe(hosts[i]) for i in np.concatenate([val, train_sample])}
    train_stems = {describe(hosts[i])["stem"] for i in train}
    for i in val:
        d = info[hosts[i]]
        d["stem_seen"] = "stem seen in train" if d["stem"] in train_stems else "stem unseen"
    lengths = np.array([len(h) for h in hosts])
    report["representation"] = {
        "length_percentiles_all": {
            str(q): float(np.percentile(lengths, q)) for q in (5, 25, 50, 75, 95, 99, 100)
        },
        "length_median_by_class": {
            "phishing": float(np.median(lengths[y == 1])),
            "legitimate": float(np.median(lengths[y == 0])),
        },
        "hosts_longer_than_64": int((lengths > 64).sum()),
        "hosts_longer_than_129_receptive_field": int((lengths > 129).sum()),
        "validation_unknown_char_hosts": int(sum(info[hosts[i]]["unknown_chars"] for i in val)),
        "validation_all_name_role_hosts": int(sum(info[hosts[i]]["all_name_roles"] for i in val)),
        "validation_unknown_suffix_hosts": int(
            sum(info[hosts[i]]["suffix_known"] == "unknown suffix" for i in val)
        ),
        "validation_punycode_hosts": int(sum("xn--" in hosts[i] for i in val)),
    }
    print("described", round(time.time() - started), "s", flush=True)

    # ---------------------------------------------------------------- scoring
    todo = sorted({hosts[i] for i in np.concatenate([val, train_sample])})
    with Pool(PROCS, initializer=_init) as pool:
        scored = dict(zip(todo, pool.map(_score, todo, chunksize=250), strict=True))
    print("scored", len(todo), round(time.time() - started), "s", flush=True)
    artifacts = {k: json.loads((v / "url_cnn.json").read_text()) for k, v in RUNS.items()}

    def arrays(model, idx):
        members = np.array([scored[hosts[i]][model] for i in idx])  # (n, m)
        z = members.mean(1)
        a = artifacts[model]
        p = sigmoid(a["calibration_slope"] * z + a["calibration_intercept"])
        return members, z, p, p >= a["threshold"]

    yv = y[val]
    report["validation"] = {}
    report["subgroups"] = {}
    report["ensemble"] = {}
    report["memorisation"] = {}
    for model in RUNS:
        members, z, p, flag = arrays(model, val)
        report["validation"][model] = {
            "all": summary(yv, z, p, flag, bootstrap=300),
            "by_source": {
                s: summary(yv[m], z[m], p[m], flag[m], bootstrap=200 if m.sum() > 5000 else 0)
                for s in ("phiusiil", "phishing_database", "crux")
                if (m := src[val] == s).any()
            },
        }
        # Subgroups on the whole validation split and on the fresh part only.
        dims = ["shape", "platform_suffix", "suffix_known", "length", "labels", "digit"]
        dims += ["hyphen", "idn", "tld", "stem_seen"]
        groups = {}
        for dim in dims:
            values = np.array([info[hosts[i]][dim] for i in val])
            groups[dim] = {
                str(v): summary(yv[m], z[m], p[m], flag[m])
                for v in np.unique(values)
                if (m := values == v).sum() >= 50
            }
        shape = np.array([info[hosts[i]]["shape"] for i in val])
        for s in ("phiusiil", "phishing_database", "crux"):
            for sh in ("bare", "subdomain"):
                m = (src[val] == s) & (shape == sh)
                if m.sum() >= 50:
                    groups.setdefault("source_x_shape", {})[f"{s} / {sh}"] = summary(
                        yv[m], z[m], p[m], flag[m]
                    )
        report["subgroups"][model] = groups

        # Ensemble diversity.
        a = artifacts[model]
        z_cut = (math.log(a["threshold"] / (1 - a["threshold"])) - a["calibration_intercept"]) / a[
            "calibration_slope"
        ]
        member_flags = members >= z_cut
        wrong = member_flags != (yv == 1)[:, None]
        ens_wrong = flag != (yv == 1)
        m_count = members.shape[1]
        pairs = []
        for i in range(m_count):
            for j in range(i + 1, m_count):
                n11 = np.sum(~wrong[:, i] & ~wrong[:, j])
                n00 = np.sum(wrong[:, i] & wrong[:, j])
                n10 = np.sum(~wrong[:, i] & wrong[:, j])
                n01 = np.sum(wrong[:, i] & ~wrong[:, j])
                pairs.append(
                    {
                        "members": [i, j],
                        "logit_pearson": float(np.corrcoef(members[:, i], members[:, j])[0, 1]),
                        "disagreement_at_threshold": float(
                            (member_flags[:, i] != member_flags[:, j]).mean()
                        ),
                        "yule_q_errors": float(
                            (n11 * n00 - n01 * n10) / max(1, n11 * n00 + n01 * n10)
                        ),
                        "p_both_wrong_given_one_wrong": float(n00 / max(1, n00 + n01 + n10)),
                    }
                )
        report["ensemble"][model] = {
            "member_auc": [float(roc_auc_score(yv, members[:, i])) for i in range(m_count)],
            "member_recall_at_1pct_fpr": [
                recall_at_fpr(yv, members[:, i], 0.01)[0] for i in range(m_count)
            ],
            "ensemble_auc": float(roc_auc_score(yv, z)),
            "ensemble_recall_at_1pct_fpr": recall_at_fpr(yv, z, 0.01)[0],
            "pairs": pairs,
            "share_of_ensemble_errors_all_members_wrong": float(wrong[ens_wrong].all(1).mean()),
            "oracle_any_member_right_error_rate": float(wrong.all(1).mean()),
            "ensemble_error_rate": float(ens_wrong.mean()),
        }

        # Memorisation: clean scores on training hostnames vs validation hostnames.
        sample = train_sample
        if model == "ens3":  # ens3 only trained on PhiUSIIL hostnames
            sample = train_sample[src[train_sample] == "phiusiil"]
            vmask = src[val] == "phiusiil"
        else:
            vmask = np.ones(len(val), dtype=bool)
        t_members, t_z, t_p, t_flag = arrays(model, sample)
        yt = y[sample]

        def logloss(yy, pp):
            pp = np.clip(pp, 1e-7, 1 - 1e-7)
            return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))

        report["memorisation"][model] = {
            "train_sample": {
                "n": int(len(sample)),
                "auc": float(roc_auc_score(yt, t_z)),
                "log_loss": logloss(yt, t_p),
                "recall_at_1pct_fpr": recall_at_fpr(yt, t_z, 0.01)[0],
            },
            "validation": {
                "n": int(vmask.sum()),
                "auc": float(roc_auc_score(yv[vmask], z[vmask])),
                "log_loss": logloss(yv[vmask], p[vmask]),
                "recall_at_1pct_fpr": recall_at_fpr(yv[vmask], z[vmask], 0.01)[0],
            },
        }
        print("analysed", model, round(time.time() - started), "s", flush=True)

    # ---------------------------------------------------------------- hard examples (fresh)
    members, z, p, flag = arrays("fresh", val)
    order = np.argsort(p)
    legit = [i for i in order[::-1] if yv[i] == 0][:40]
    phish = [i for i in order if yv[i] == 1][:40]
    report["hard_examples_fresh"] = {
        "highest_scoring_legitimate": [
            {"host": hosts[val[i]], "p": float(p[i]), "source": src[val][i]} for i in legit
        ],
        "lowest_scoring_phishing": [
            {"host": hosts[val[i]], "p": float(p[i]), "source": src[val][i]} for i in phish
        ],
        "legitimate_with_p_over_0.99": int(((p > 0.99) & (yv == 0)).sum()),
        "phishing_with_p_under_0.05": int(((p < 0.05) & (yv == 1)).sum()),
    }
    # What separates missed from caught bare-domain phishing?
    shape = np.array([info[hosts[i]]["shape"] for i in val])
    bare_phish = (yv == 1) & (shape == "bare")
    bare_legit = (yv == 0) & (shape == "bare")

    def profile(mask):
        hs = [hosts[val[i]] for i in np.flatnonzero(mask)]
        return {
            "n": len(hs),
            "median_length": float(np.median([len(h) for h in hs])),
            "share_digit": float(np.mean([any(c.isdigit() for c in h) for h in hs])),
            "share_hyphen": float(np.mean(["-" in h for h in hs])),
            "share_com": float(np.mean([h.endswith(".com") for h in hs])),
            "share_hosting_platform": float(
                np.mean([info[h]["platform_suffix"] != "ICANN suffix" for h in hs])
            ),
            "share_stem_seen_in_train": float(
                np.mean([info[h]["stem_seen"] == "stem seen in train" for h in hs])
            ),
        }

    report["bare_domain_profiles_fresh"] = {
        "phishing_caught": profile(bare_phish & flag),
        "phishing_missed": profile(bare_phish & ~flag),
        "legitimate_flagged": profile(bare_legit & flag),
        "legitimate_passed": profile(bare_legit & ~flag),
    }

    # ---------------------------------------------------------------- label conflicts
    phishing_list = feeds.phishing_hostnames(FEEDS / "raw" / "phishing-domains-ACTIVE.txt")
    crux = feeds.crux_hostnames(FEEDS / "raw" / "crux-202610.csv.gz")
    crux_regs = {registrable_domain(h) or h for h in crux}
    phi = [(h, int(y[index_of[h]])) for h in hosts if source[h] == "phiusiil"]
    report["label_conflicts"] = {
        "phiusiil_legitimate": sum(1 for _, lab in phi if lab == 0),
        "phiusiil_phishing": sum(1 for _, lab in phi if lab == 1),
        "phiusiil_legitimate_listed_as_phishing_by_phishing_database": sum(
            1 for h, lab in phi if lab == 0 and h in phishing_list
        ),
        "phiusiil_phishing_listed_in_crux_top_1m": sum(
            1 for h, lab in phi if lab == 1 and h in crux
        ),
        "phiusiil_phishing_whose_registrable_domain_serves_a_crux_site": sum(
            1 for h, lab in phi if lab == 1 and (registrable_domain(h) or h) in crux_regs
        ),
        "examples_legit_listed_as_phishing": [
            h for h, lab in phi if lab == 0 and h in phishing_list
        ][:15],
        "examples_phishing_in_crux": [h for h, lab in phi if lab == 1 and h in crux][:15],
    }
    print("label conflicts", round(time.time() - started), "s", flush=True)

    # ---------------------------------------------------------------- structural probes
    def probe_hosts(mask, limit=3000):
        chosen = [hosts[val[i]] for i in np.flatnonzero(mask)]
        chosen = [h for h in chosen if info[h]["suffix_known"] == "known suffix"]
        picks = np.random.default_rng(1).choice(len(chosen), min(limit, len(chosen)), replace=False)
        return [chosen[i] for i in sorted(picks)]

    letters = string.ascii_lowercase + string.digits

    def random_label(h):
        r = np.random.default_rng(int(hashlib.sha256(h.encode()).hexdigest()[:8], 16))
        return "".join(r.choice(list(letters), 6))

    transforms = {
        "prefix shop.": lambda h: "shop." + h,
        "prefix login.": lambda h: "login." + h,
        "prefix random 6-char label": lambda h: random_label(h) + "." + h,
        "name + -login": lambda h: extract(h).domain + "-login." + extract(h).suffix,
        "name + digit 7": lambda h: extract(h).domain + "7." + extract(h).suffix,
        "suffix -> .xyz": lambda h: extract(h).domain + ".xyz",
        "suffix -> .com": lambda h: extract(h).domain + ".com",
    }
    probe_sets = {
        "legitimate bare (validation)": probe_hosts(bare_legit),
        "phishing bare (validation)": probe_hosts(bare_phish),
    }
    variants = sorted(
        {t(h) for hs in probe_sets.values() for h in hs for t in transforms.values()}
        | {h for hs in probe_sets.values() for h in hs}
    )
    variants = [v for v in variants if v not in scored]
    with Pool(PROCS, initializer=_init) as pool:
        scored.update(zip(variants, pool.map(_score, variants, chunksize=250), strict=True))

    def prob(model, h):
        a = artifacts[model]
        zz = float(np.mean(scored[h][model]))
        return float(sigmoid(a["calibration_slope"] * zz + a["calibration_intercept"])), a[
            "threshold"
        ]

    report["probes"] = {}
    for model in RUNS:
        for set_name, hs in probe_sets.items():
            base = [prob(model, h) for h in hs]
            for t_name, t in transforms.items():
                changed = [prob(model, t(h)) for h in hs]
                flags_before = np.array([b >= thr for b, thr in base])
                flags_after = np.array([c >= thr for c, thr in changed])
                report["probes"].setdefault(model, {}).setdefault(set_name, {})[t_name] = {
                    "n": len(hs),
                    "flagged_before": float(flags_before.mean()),
                    "flagged_after": float(flags_after.mean()),
                    "mean_probability_change": float(
                        np.mean([c[0] - b[0] for c, b in zip(changed, base, strict=True)])
                    ),
                }
    print("probes", round(time.time() - started), "s", flush=True)

    # ---------------------------------------------------------------- non-neural reference
    # Character n-gram TF-IDF + logistic regression on the same training split. If it
    # matches the CNN, the bottleneck is information in the data, not the architecture.
    cal = np.flatnonzero(where == "calibration")

    def structured(h):
        r = extract(h)
        return f"<{r.subdomain}|{r.domain}|{r.suffix}>" if r.suffix else f"<|{h}|>"

    baselines = {}
    for variant, text in (("plain", lambda h: f"<{h}>"), ("component markers", structured)):
        vec = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 5), min_df=3, sublinear_tf=True, dtype=np.float32
        )
        Xtr = vec.fit_transform([text(hosts[i]) for i in train])
        Xcal = vec.transform([text(hosts[i]) for i in cal])
        Xval = vec.transform([text(hosts[i]) for i in val])
        best = None
        for c in (1.0, 4.0, 16.0):
            clf = LogisticRegression(C=c, max_iter=2000, solver="liblinear").fit(Xtr, y[train])
            auc = roc_auc_score(y[cal], clf.decision_function(Xcal))
            if best is None or auc > best[0]:
                best = (auc, c, clf)
        s = best[2].decision_function(Xval)
        row = {"chosen_C_on_calibration_split": best[1], "features": int(Xtr.shape[1])}
        row["validation_auc"] = float(roc_auc_score(yv, s))
        row["validation_recall_at_1pct_fpr"] = recall_at_fpr(yv, s, 0.01)[0]
        row["validation_recall_at_0.1pct_fpr"] = recall_at_fpr(yv, s, 0.001)[0]
        row["by_source_auc"] = {
            k: float(roc_auc_score(yv[m], s[m]))
            for k in ("phiusiil", "phishing_database", "crux")
            if (m := src[val] == k).any() and len(np.unique(yv[m])) == 2
        }
        _, zf, _, _ = arrays("fresh", val)
        row["pearson_with_cnn_logit"] = float(np.corrcoef(s, zf)[0, 1])
        both = (s + 0) / s.std() + zf / zf.std()
        row["auc_of_cnn_plus_ngram_sum"] = float(roc_auc_score(yv, both))
        row["recall_at_1pct_fpr_of_cnn_plus_ngram_sum"] = recall_at_fpr(yv, both, 0.01)[0]
        baselines[variant] = row
        print(
            "baseline",
            variant,
            row["validation_auc"],
            round(time.time() - started),
            "s",
            flush=True,
        )
    report["ngram_logistic_reference"] = baselines

    report["seconds"] = round(time.time() - started)
    (OUT / "phase1_audit.json").write_text(json.dumps(report, indent=1, default=float))
    print("done", report["seconds"], "s")


if __name__ == "__main__":
    main()

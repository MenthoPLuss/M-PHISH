# M-PHISH X v0.2.0 — Project Analysis

*Independent review of the uploaded `M-PHISH-0.2.0v.zip` (226 zip entries, 174 files). Every claim in this document was checked by reading the code or running it.*

---

## 1. What this project is

Despite the name, **M-PHISH X is a defensive anti-phishing tool, not a phishing kit.** It looks at a website and the sensitive actions the site asks for (passwords, card numbers, OTPs, downloads), then writes an evidence-based risk report. Each report says what was observed and also what could not be checked.

Its design principle: **missing evidence never becomes a "safe" verdict.** When checks can't run, the result is `UNKNOWN` / `INSUFFICIENT_EVIDENCE`, not `LOW`.

It is a **local release candidate, not GA**. The authors say this themselves in `docs/RELEASE_READINESS.md`.

## 2. Architecture

```
Dashboard (Next.js)  /  Chromium MV3 extension
              │  authenticated requests
              ▼
         FastAPI API ───────► PostgreSQL (SQLite in dev)
              │ commits job       users, sessions, jobs, reports
              ▼
   Outbox dispatcher ──► Redis/RQ ──► Worker
                                       │ private RPC
                                       ▼
                             Isolated collector  ──► target website
                                                     (passive HTML only)
   observations → analyzers → evidence → risk policy → owner-scoped report
```

| Component | Path | Tech | Role |
|---|---|---|---|
| API | `backend/app/api.py`, `main.py` | FastAPI, Pydantic | Auth, investigations, exports, admin (31 routes) |
| Analysis pipeline | `backend/app/analysis.py` | Python | Combines all analyzer outputs into one report |
| Analyzers | `backend/app/services/*/` | Python | domain, webpage, identity, privacy, download, claims, synthetic, campaign, trust, context, ML, AI |
| Risk policy | `backend/app/services/risk/engine.py` | Python | Deterministic scoring (`evidence-policy-v2`) |
| Collector | `backend/app/collector.py` | stdlib http/ssl | Sandboxed, SSRF-hardened page fetcher |
| Job system | `backend/app/jobs.py`, `worker.py` | RQ + DB leases | Durable outbox, retries (max 3), cancellation |
| Persistence | `backend/app/core/storage.py`, `alembic/` | SQLAlchemy, Alembic | 4 migrations |
| Dashboard | `web/` | Next.js 15.5, React 18 | Investigations, reports, research, settings, admin |
| Extension | `extension/src/*.ts` | MV3, TypeScript, esbuild | Page observations, warnings, action intervention |
| Deployment | `docker-compose.yml`, `deploy/` | Docker, iptables | Local stack + production compose + collector firewall |

Size: about 4,500 lines of backend Python and 620 lines of backend tests. The TypeScript is compact (dense, few lines).

## 3. How detection works

### 3.1 Risk scoring (`services/risk/engine.py`)
- Evidence is grouped by **category** (URL, DOMAIN, CONTENT, BEHAVIOR, IDENTITY, PRIVACY, CONTEXT). Only the **strongest item in each category** counts, so correlated signals are not counted twice.
- Each category is capped at 40 points, and the total at 100.
- Some evidence is excluded from scoring on purpose: CONTEXT items, the synthetic-language analyzer, unavailable evidence, PRIVACY items with no destination concern, and CONTENT items below HIGH severity.
- There are severity floors: any HIGH evidence with weight ≥ 20 raises the score to at least 50, and CRITICAL raises it to at least 75.
- The **ML model can only raise the score to a floor of 50**. It is never added on top of the rule score.
- Levels: ≥75 CRITICAL, ≥50 HIGH, ≥25 MEDIUM. `LOW` requires **full coverage**; anything else is `UNKNOWN`.

### 3.2 Results on the bundled sample pages (run offline with this codebase)

| Sample | Score | Class | Top factor |
|---|---|---|---|
| `suspicious-login.html` | 75 | CRITICAL | Sensitive form has an unrelated destination |
| `identity-mismatch.html` | 75 | CRITICAL | Unrelated form destination + branding/destination mismatch |
| `privacy-heavy.html` | 75 | CRITICAL | Sensitive form has an unrelated destination |
| `download-risk.html` | 50 | HIGH | Executable or script link observed |
| `benign-login.html` | 0 | UNKNOWN | — |
| `simple-site.html` | 0 | UNKNOWN | — |
| `redirect-example.html` | 0 | UNKNOWN | — |
| `synthetic-content.html` | 0 | UNKNOWN | — (synthetic language is excluded from scoring by design) |

Benign pages come out as `UNKNOWN`, not `LOW`. That is because offline mode has no DNS/TLS coverage, and the policy is doing what it was written to do.

### 3.3 Strongest signal
In practice, the strongest detector is **"a sensitive form posts to an unrelated domain."** It is the classic credential-harvesting pattern, and here it is checked against the page's origin and a known-SSO allowlist (Google, Microsoft, Apple, PayPal, …) to avoid false positives.

### 3.4 ML and AI
- **ML** (`services/ml/model.py`): loads only a checksum-pinned, pure-JSON linear model (`m-phish-linear-v1`) that has `release_gate: "passed"`. No pickle, so there is no code-execution risk. **No trained model is bundled**, so ML is inactive by default.
- **AI** (`services/ai/grounded.py`): optional Groq call. It is off by default, needs consent, and sends only evidence IDs, titles and severities (never URLs or page content). The output is validated so the model can only *pick up to 3 existing evidence IDs*. It cannot change scores or invent claims, and a circuit breaker opens after 3 failures.

## 4. Security review

### Strengths (verified in code)
| Area | Implementation |
|---|---|
| SSRF protection | `core/network.py` rejects non-HTTP(S) schemes, embedded credentials, non-80/443 ports, `localhost`/`.local`/`.internal`, and private, loopback, link-local, multicast and reserved IPs (including IPv4-mapped IPv6). It validates **every** DNS answer and **every redirect hop**. |
| DNS-rebinding defense | The collector connects to the pre-validated numeric IP, re-checks `getpeername()`, and uses TLS SNI with hostname verification. |
| Resource bounds | 15s deadline, 1 MB body limit, 6 redirect hops, 2 concurrent slots, killable subprocess (24s hard timeout). HTML only, compressed responses refused. |
| Collector isolation | No DB, Redis or provider credentials. `deploy/collector-firewall.sh` restricts egress. |
| Auth | Argon2 passwords (pwdlib). Opaque tokens stored as SHA-256 hashes. 8h web / 30min extension sessions. Strict Origin checks on mutating requests. |
| Extension pairing | Single-use 5-minute code exchanged for a bearer token bound to an allowlisted `chrome-extension://` origin. |
| Data minimization | Stored URLs are reduced to the **origin** (paths and queries can hold tokens). No page HTML, input values or screenshots are stored in reports. |
| Extension privacy | The content script explicitly never reads `.value`, cookies or the clipboard. Requests use `credentials:"omit"` and `redirect:"error"`, and go only to the user-configured API. |
| Multi-tenancy | Every investigation query is owner-scoped. Campaign similarity never crosses accounts. |
| Job integrity | Lease tokens prevent stale or duplicate publication. Cancelled or deleted jobs can't be published late. |
| API surface | OpenAPI/docs endpoints are disabled. Production refuses to start without PostgreSQL, Redis and HTTPS origins. |

### Supply chain and integrity checks (done during this analysis)
- **No obfuscation:** no `eval`, `new Function`, `atob`, base64 blobs or pickle loading. `subprocess` is used only for the bounded collector/visual runner and `scripts/verify.py`.
- **Outbound hosts:** the only third-party network call in the code is `api.groq.com` (opt-in). The other URLs are SSO allowlist entries, docs links and `.example` test fixtures.
- **Release zip:** `extension/release/m-phish-0.2.0.zip` matches its `SHA256SUMS` (`2fc9b530…85a2cb`), and its files are identical to `extension/*.js`.
- **Reproducible build:** rebuilding the extension from `extension/src/*.ts` with esbuild gives **byte-identical** output to the shipped JS, so the shipped extension contains only what the readable source says.
- Dependency versions are pinned through lockfiles (`requirements*.lock`, `package-lock.json`).

### Weaknesses and residual risks
Most of these are acknowledged in `docs/threat-model.md`.
1. **Page-injected warnings can be spoofed or removed** by a malicious page. Data sent through `fetch`/XHR/WebSocket bypasses the extension's native-submission intervention.
2. **Top-frame only** (`all_frames: false`), so iframe-embedded forms are not observed client-side.
3. **Domain age/registration is always unavailable.** No WHOIS/RDAP source is wired in, and young domains are one of the most useful phishing signals.
4. **Identity catalog is small** (about 11 brands, some India-specific such as SBI and scholarships.gov.in).
5. **The server collector doesn't run JavaScript.** JS-rendered phishing kits look empty to the server, so it depends on extension observations.
6. **No bundled ML model**, so ML contributes nothing out of the box.
7. **Cross-owner queue admission race** (documented). Capacity has to be enforced at the infrastructure level.
8. **Collector egress firewall is unverified** in real deployment (stated by the authors).

## 5. Code-quality observations

- **Clean:** `ruff check` passes and all 56 files are formatted. Code is consistently annotated with *why* comments explaining security decisions.
- **Config is read at import time** (`core/config.py` is a frozen dataclass with `os.getenv` defaults). Environment variables must be set **before** importing the app. The app does **not** auto-load `.env`; VS Code's `envFile` (configured here) or `set -a; . ./.env` handles it.
- **Python version:** the project targets 3.12. Everything passed on **3.13** in this environment.
- **Deprecation warning:** Starlette warns that using `httpx` with `TestClient` is deprecated (harmless for now).
- **TypeScript style** is very dense (many statements per line), which makes the extension harder to review and diff. Running a formatter such as Prettier would help maintainability.
- **Duplicate legacy routes:** every endpoint is served at both `/api/...` and `/api/v1/...`.

## 6. Verification results (run during this analysis)

| Check | Result |
|---|---|
| `ruff check backend/app backend/tests` | ✅ All checks passed |
| `ruff format --check` | ✅ 56 files already formatted |
| `pytest backend/tests` | ✅ **44 passed** (1 deprecation warning) |
| `web: npm test` | ✅ **6/6** |
| `web: npm run typecheck` | ✅ clean |
| `web: npm run build` (Next.js production) | ✅ 10 routes built |
| `extension: npm run typecheck` | ✅ clean |
| `extension: npm test` | ✅ **13/13** |
| Extension rebuild vs shipped JS | ✅ byte-identical |
| Alembic migrations (SQLite) | ✅ 4/4 applied |
| API boot + `GET /api/v1/health` | ✅ `{"status":"ok","version":"0.2.0"}` |
| Unauthenticated `POST /quick-check` | ✅ correctly rejected with 401 |

These numbers match the project's own claims in `CHANGELOG_v0.2.0.md` (44 backend / 6 web / 13 extension).

**Not run:** Playwright browser journeys (`web/tests/browser_smoke.py`, `extension/tests/browser.py`), the full Docker Compose stack with PostgreSQL and Redis, and the optional visual/OCR runtime.

## 7. Running it locally

### Windows (PowerShell): run each line separately

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.lock
cd web; npm ci; cd ..\extension; npm ci; cd ..
Copy-Item .env.example .env
$env:DATABASE_URL = "sqlite:///backend/m_phish.db"
.\.venv\Scripts\python.exe -m alembic -c backend\alembic.ini upgrade head
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

If `py -3.12` fails, use `py -m venv .venv` (any Python 3.12 or newer works). If `py` isn't found, install Python from python.org and tick **"Add python.exe to PATH"**.

### Linux / macOS

```bash
# one-time (already done in this folder)
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.lock
(cd web && npm ci) && (cd extension && npm ci)
cp .env.example .env

# database + API
set -a && . ./.env && set +a
.venv/bin/python -m alembic -c backend/alembic.ini upgrade head
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# create a user (interactive password prompt)
.venv/bin/python -m backend.app.manage create-admin --email you@example.com

# dashboard
cd web && npm run dev     # http://localhost:3000
```

Deep investigations need **Redis** plus the **dispatcher** (`python -m backend.app.jobs`) and **worker** (`python -m backend.app.worker`). The easiest way to get the whole stack is `docker compose up --build`.

**Extension:** open `chrome://extensions`, turn on Developer mode, choose *Load unpacked* and select the `extension/` folder. Then add its ID to `EXTENSION_IDS` in `.env`.

## 8. Suggested next steps (if you plan to develop this)

1. Add an **RDAP/WHOIS domain-age source**. It is one of the highest-value missing signals.
2. Expand the **identity/brand catalog**, or load it from data instead of code.
3. Consider `all_frames: true`, with care, to cover iframe-embedded login forms.
4. Train and gate a real **ML artifact** using `services/ml/offline.py` and labeled data (see `docs/DATASETS.md`, `docs/ML_MODEL_CARD.md`). **Added:** a hostname character CNN with a PhiUSIIL training pipeline; see `docs/URL_CNN.md`.
5. Add Prettier to `web/` and `extension/` for readability.
6. Pin Python to 3.12 in a `.python-version` file, or officially test against 3.13.

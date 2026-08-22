"""Fetch HCP-YA non-imaging data (behavioural, demographic, session metadata).

ConnectomeDB (``db.humanconnectome.org``) has been retired -- the host 301s
wholesale to BALSA, including the ``/REST/search/dict/...`` endpoint that older
HCP tutorials use for the behavioural CSV. Everything now comes from the BALSA
HCP_YA project page, behind a Spring Security form login.

Two sources are pulled:

1. **The subject table** -- BALSA's "Export CSV" button, which is
   ``POST /project/subjectCSV`` with opaque per-variable ids. The ids are
   discovered by scraping the column-selection menus on the project page, where
   each checkbox carries the real variable name in its ``name`` attribute and
   the restricted ones are tagged ``.subjectVariableR``. This is the table
   holding ``NEOFAC_N`` and the rest of the plan's targets.
2. **Project files** -- the Files tab, ``/project/downloadProjectFile/<id>``:
   session summaries, the data dictionary, FreeSurfer phenotypes, in-scanner
   task performance, and the restricted raw instrument dumps.

Restricted columns and restricted files are gated server-side on the account's
``viewRestricted`` flag; both are attempted and a 403 is reported rather than
raised, so an entitlement that is approved but not yet linked shows up plainly.

Credentials, in order of precedence:
  1. ``BALSA_USER`` / ``BALSA_PASS`` environment variables
  2. an ini file (default ``~/.balsa/credentials``) with::

         [balsa]
         username = ...
         password = ...

Usage::

    python src/00_acquisition/fetch_hcp_ya_phenotypes.py --discover
    python src/00_acquisition/fetch_hcp_ya_phenotypes.py
    python src/00_acquisition/fetch_hcp_ya_phenotypes.py --no-files
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common_provenance import stamp  # noqa: E402

BASE = "https://balsa.wustl.edu"
PROJECT = "HCP_YA"
PROJECT_URL = f"{BASE}/project?project={PROJECT}"
LOGIN_URL = f"{BASE}/j_spring_security_check"
SUBJECTS_URL = f"{BASE}/project/subjectsForSubjectGroup"
CSV_URL = f"{BASE}/project/subjectCSV"

DEST = ROOT / "data" / "HCP_young_adults" / "demographics"

# "All Family Subjects" -- the full 1206-subject roster, superset of the
# imaging subgroups. Other options are listed by --discover.
SUBJECT_GROUP = "HCP_1200_all"

# Imaging packages and stimulus bundles are Aspera-gated and out of scope; the
# few large stimulus zips that do appear as project files are skipped by name.
SKIP_FILES = re.compile(r"stimulus|movie_|TFMRI_scripts", re.I)


def credentials(ini_path: Path) -> tuple[str, str]:
    user, password = os.environ.get("BALSA_USER"), os.environ.get("BALSA_PASS")
    if not (user and password) and ini_path.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini_path)
        if cfg.has_section("balsa"):
            user = user or cfg["balsa"].get("username")
            password = password or cfg["balsa"].get("password")
    if not (user and password):
        sys.exit(
            f"No BALSA credentials. Set BALSA_USER/BALSA_PASS, or write {ini_path} "
            "with a [balsa] section containing username and password."
        )
    return user, password


def login(user: str, password: str) -> tuple[requests.Session, BeautifulSoup, bool]:
    """Log in and return the session, the parsed project page, and the
    restricted-access flag.

    The login modal stays in the page layout even when authenticated, so the
    presence of a password field proves nothing. Authentication is confirmed by
    the username appearing in the page chrome.
    """
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; hcp-ya-phenotype-fetch)"
    session.get(PROJECT_URL, timeout=60)  # seed JSESSIONID
    session.post(
        LOGIN_URL,
        data={"j_username": user, "j_password": password, "_spring_security_remember_me": "on"},
        timeout=60,
    )
    page = session.get(PROJECT_URL, timeout=180)
    page.raise_for_status()
    if user not in page.text or "Logout" not in page.text:
        sys.exit("BALSA login rejected -- the project page came back unauthenticated.")

    soup = BeautifulSoup(page.text, "html.parser")
    flag = soup.find(id="viewRestricted")
    view_restricted = bool(flag) and flag.get("value") == "true"
    print(f"logged in as {user} | restricted access: {'yes' if view_restricted else 'NO'}")
    return session, soup, view_restricted


def column_map(soup: BeautifulSoup) -> list[dict]:
    """Variable name -> opaque BALSA column id, from the selection menus."""
    cols = []
    for menu in soup.select("[id^=menuFor]"):
        category = menu.get("id")[len("menuFor") :].replace("_", " ")
        for cb in menu.select(".subjectVariable"):
            label = cb.parent.find("label")
            cols.append(
                {
                    "name": cb.get("name"),
                    "balsa_id": cb.get("value"),
                    "category": category,
                    "restricted": "subjectVariableR" in (cb.get("class") or []),
                    "label": " ".join(label.get_text().split()) if label else "",
                }
            )
    return cols


def subject_ids(session: requests.Session, subject_col: str, group: str) -> list[str]:
    """BALSA-internal row ids for the group (not the 6-digit HCP subject ids)."""
    r = session.post(
        SUBJECTS_URL,
        data={"project": PROJECT, "subjectGroup": group, "cols": subject_col},
        timeout=600,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    ids = [td.get("data-subjectid") for td in soup.select("#subjectTable tbody tr td:first-child")]
    return [i for i in ids if i]


def subject_csv(session: requests.Session, ids: list[str], cols: list[dict]) -> str | None:
    r = session.post(
        CSV_URL,
        data={
            "project": PROJECT,
            "subjects": ",".join(ids),
            "cols": ",".join(c["balsa_id"] for c in cols),
        },
        timeout=900,
    )
    if r.status_code == 403:
        return None
    r.raise_for_status()
    return r.text


def project_files(soup: BeautifulSoup) -> list[dict]:
    out = []
    for a in soup.find_all("a", href=True):
        if "/project/downloadProjectFile/" not in a["href"]:
            continue
        row = a.find_parent("tr")
        cells = [" ".join(c.get_text().split()) for c in row.find_all("td")] if row else []
        text = " ".join(cells)
        # The row reads "<n> <filename> <description>". Filenames may contain
        # spaces, so take everything from the row number up to the extension
        # rather than the last whitespace-delimited token.
        m = re.match(
            r"\s*\d+\s+(?:\(Restricted Data\)\s*)?(.+?\.(?:csv|zip|xlsx|docx|pdf|txt|tsv))(?:\s|$)",
            text,
        )
        out.append(
            {
                "url": BASE + a["href"],
                "filename": m.group(1) if m else a["href"].rsplit("/", 1)[-1],
                "restricted": "(Restricted Data)" in text,
                "description": text[:300],
            }
        )
    return out


def write_csv(text: str, path: Path) -> dict:
    path.write_text(text)
    rows = list(csv.reader(io.StringIO(text)))
    print(f"  {path.name}  {len(rows) - 1} subjects x {len(rows[0])} columns  ({len(text) / 1e6:.1f} MB)")
    return {
        "file": path.name,
        "subjects": len(rows) - 1,
        "columns": len(rows[0]),
        "bytes": len(text.encode()),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def download_file(session: requests.Session, entry: dict, dest: Path) -> dict | None:
    with session.get(entry["url"], stream=True, timeout=1800) as r:
        if r.status_code == 403:
            print(f"  403 FORBIDDEN  {entry['filename']}  (restricted)")
            return None
        r.raise_for_status()
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
        name = unquote(m.group(1)) if m else entry["filename"]
        path = dest / name
        digest, size = hashlib.sha256(), 0
        with open(path, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    print(f"  {name}  ({size / 1e6:.1f} MB){'  [restricted]' if entry['restricted'] else ''}")
    return {
        "file": name,
        "url": entry["url"],
        "restricted": entry["restricted"],
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--discover", action="store_true", help="report what is available, download nothing")
    ap.add_argument("--no-files", action="store_true", help="subject table only, skip the Files tab")
    ap.add_argument("--group", default=SUBJECT_GROUP, help="BALSA subject group")
    ap.add_argument("--out", type=Path, default=DEST)
    ap.add_argument("--credentials", type=Path, default=Path.home() / ".balsa" / "credentials")
    args = ap.parse_args()

    user, password = credentials(args.credentials)
    session, soup, view_restricted = login(user, password)

    cols = column_map(soup)
    files = project_files(soup)
    n_restricted = sum(c["restricted"] for c in cols)
    print(f"{len(cols)} subject variables ({n_restricted} restricted) | {len(files)} project files")

    if args.discover:
        print("\nsubject groups:")
        for o in soup.select("#subjectGroup option"):
            print(f"    {o.get('value'):<24} {' '.join(o.get_text().split())}")
        print("\nvariables by category:")
        cats: dict[str, list[dict]] = {}
        for c in cols:
            cats.setdefault(c["category"], []).append(c)
        for cat, cc in cats.items():
            print(f"    {cat:<32} {len(cc):>4}  ({sum(x['restricted'] for x in cc)} restricted)")
        print("\nproject files:")
        for f in files:
            skip = " [skipped: stimulus/script bundle]" if SKIP_FILES.search(f["filename"]) else ""
            print(f"    {'R' if f['restricted'] else ' '} {f['filename']}{skip}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source": f"BALSA {PROJECT} project",
        "subject_group": args.group,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "restricted_access": view_restricted,
    }

    # Column dictionary: the name <-> BALSA id mapping this whole route depends
    # on, saved so a later run can be reproduced without re-scraping.
    dict_path = args.out / "hcp_ya_column_map.csv"
    with open(dict_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "balsa_id", "category", "restricted", "label"])
        w.writeheader()
        w.writerows(cols)
    print(f"\n  {dict_path.name}  {len(cols)} variables")

    ids = subject_ids(session, next(c["balsa_id"] for c in cols if c["name"] == "Subject"), args.group)
    print(f"  subject group {args.group}: {len(ids)} subjects")

    print("\nsubject table:")
    open_cols = [c for c in cols if not c["restricted"]]
    manifest["unrestricted"] = write_csv(
        subject_csv(session, ids, open_cols), args.out / "hcp_ya_unrestricted.csv"
    )

    text = subject_csv(session, ids, cols)
    if text is None:
        print("  403 FORBIDDEN  restricted columns -- account lacks the Restricted Access entitlement")
        manifest["restricted"] = {"status": "forbidden"}
    else:
        manifest["restricted"] = write_csv(text, args.out / "hcp_ya_restricted.csv")

    if not args.no_files:
        print("\nproject files:")
        wanted = [f for f in files if not SKIP_FILES.search(f["filename"])]
        manifest["files"] = [r for f in wanted if (r := download_file(session, f, args.out))]
        manifest["files_forbidden"] = [
            f["filename"] for f in wanted if f["restricted"] and not view_restricted
        ]

    (args.out / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    stamp(args.out, "src/00_acquisition/fetch_hcp_ya_phenotypes.py", ROOT, **{
        k: v for k, v in manifest.items() if k != "files"
    })


if __name__ == "__main__":
    main()

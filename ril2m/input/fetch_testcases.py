#!/usr/bin/env python3
"""
Generates one ragtestcases_<repo>.json per configured repository.

Usage:
    poetry run python ril2m/input/fetch_testcases.py

The script discovers all Java files under src/test/, parses test methods annotated
with @Test or @ParameterizedTest, extracts their @AccessMode annotations, and skips
any method or class annotated with @Disabled.

To add a new repository, append its GitHub URL to the REPOS list below.
"""

import base64
import json
import logging
import os
import re
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXT_DIR = os.path.join(BASE_DIR, "context")

# ──────────────────────────────────────────────────────────────────────────────
# Add / remove repositories here to control which test suites are included.
# Each entry is a public GitHub URL; the output file is named after the repo.
# ──────────────────────────────────────────────────────────────────────────────
REPOS = [
    "https://github.com/giis-uniovi/retorch-st-petclinic",
    "https://github.com/giis-uniovi/retorch-st-fullteaching",
    "https://github.com/giis-uniovi/retorch-st-eShopContainers",
]

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"


def _parse_github_url(url: str) -> tuple[str, str]:
    """Parse owner and repo name from a GitHub repository URL."""
    parts = url.rstrip("/").split("/")
    return parts[-2], parts[-1]


def _github_api_get(path: str) -> dict:
    """Fetch JSON from GitHub API, using GITHUB_TOKEN env var when available."""
    url = f"{GITHUB_API}/{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "retorch-ril2m-rp",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _fetch_raw_file(owner: str, repo: str, path: str, branch: str = "main") -> str:
    """Fetch raw file content from GitHub (avoids API rate limits)."""
    url = f"{GITHUB_RAW}/{owner}/{repo}/{branch}/{path}"
    headers = {"User-Agent": "retorch-ril2m-rp"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _list_java_test_files(owner: str, repo: str) -> tuple[list[str], str | None]:
    """
    Return ``(java_paths, system_resources_path)`` for a repository.

    *java_paths* — paths of every ``.java`` file under ``src/test/``.
    *system_resources_path* — path of the ``*SystemResources.json`` file inside
    ``.retorch/configurations/``, or ``None`` if not found.
    """
    data = _github_api_get(f"repos/{owner}/{repo}/git/trees/main?recursive=1")
    java_paths = []
    system_resources_path = None
    for item in data.get("tree", []):
        path = item.get("path", "")
        if item.get("type") != "blob":
            continue
        if path.endswith(".java") and "/test/" in path:
            java_paths.append(path)
        if ".retorch/configurations/" in path and path.endswith("SystemResources.json"):
            system_resources_path = path
    return java_paths, system_resources_path


def _remove_block_comments(source: str) -> str:
    """Remove /* ... */ block comments while preserving line count."""
    return re.sub(
        r"/\*.*?\*/",
        lambda m: "\n" * m.group().count("\n"),
        source,
        flags=re.DOTALL,
    )


def _extract_class_name(source: str) -> str:
    m = re.search(r"\bclass\s+(\w+)", source)
    return m.group(1) if m else "Unknown"


def _is_class_disabled(source: str, class_name: str) -> bool:
    """Return True if the class declaration is preceded by an active @Disabled annotation."""
    m = re.search(rf"\bclass\s+{re.escape(class_name)}\b", source)
    if not m:
        return False
    for line in reversed(source[: m.start()].split("\n")[-25:]):
        s = line.strip()
        if re.match(r"@Disabled\b", s):
            return True
        if "}" in s:
            break
    return False


def _collect_annotation_block(
    lines: list[str], start: int
) -> tuple[list[str], list[str], bool, bool, int]:
    """
    Collect consecutive @-annotation lines beginning at `start`.

    Returns:
        access_modes   – list of @AccessMode annotation strings
        other_anns     – list of other annotation strings (as they appear in source)
        has_test       – True if @Test or @ParameterizedTest was found
        is_disabled    – True if @Disabled was found
        next_idx       – index of the first non-annotation line after the block
    """
    access_modes, other_anns = [], []
    has_test = is_disabled = False
    i, n = start, len(lines)

    while i < n:
        s = lines[i].strip()
        if not s or s.startswith("//"):
            i += 1
            continue
        if not s.startswith("@"):
            break

        # Handle potentially multi-line annotation via parenthesis depth
        ann = s
        depth = ann.count("(") - ann.count(")")
        j = i + 1
        while depth > 0 and j < n:
            nxt = lines[j].strip()
            ann += " " + nxt
            depth += nxt.count("(") - nxt.count(")")
            j += 1
        i = j

        if ann.startswith("@AccessMode"):
            access_modes.append(ann)
        elif re.match(r"@Disabled\b", ann):
            is_disabled = True
        elif re.match(r"@(Test|ParameterizedTest)\b", ann):
            has_test = True
            other_anns.append(s)
        else:
            other_anns.append(s)

    return access_modes, other_anns, has_test, is_disabled, i


def _find_code_start(lines: list[str], block_start: int, method_start: int) -> int:
    """Return the index of the first non-@AccessMode annotation in the block.

    Falls back to *method_start* when no such annotation exists.
    """
    for k in range(block_start, method_start):
        s = lines[k].strip()
        if s and not s.startswith("//") and s.startswith("@") and not s.startswith("@AccessMode"):
            return k
    return method_start


def _find_test_blocks(lines: list[str]) -> list[dict]:
    """
    Scan `lines` and return a list of test-block descriptors, one per test method.

    Each descriptor contains:
        block_start  – line index where the @AccessMode annotations begin
        code_start   – line index where the non-@AccessMode annotations begin
                       (falls back to method_start if there are none)
        method_start – line index of the method signature
        access_modes – list of @AccessMode annotation strings
        is_disabled  – whether the test is marked @Disabled
    """
    n = len(lines)
    tests = []
    i = 0

    while i < n:
        s = lines[i].strip()
        if not s or s.startswith("//"):
            i += 1
            continue

        if s.startswith("@"):
            block_start = i
            access_modes, _, has_test, is_disabled, next_i = _collect_annotation_block(lines, i)

            if has_test:
                tests.append(
                    {
                        "block_start": block_start,
                        "code_start": _find_code_start(lines, block_start, next_i),
                        "method_start": next_i,
                        "access_modes": access_modes,
                        "is_disabled": is_disabled,
                    }
                )
            i = next_i
        else:
            i += 1

    return tests


def _extract_method_name(lines: list[str]) -> str:
    """Extract the method name from the first lines of a method signature."""
    skip = {"if", "while", "for", "catch", "switch", "new", "return", "super", "this"}
    for line in lines[:8]:
        m = re.search(r"\w+\s+(\w+)\s*\(", line)
        if m and m.group(1) not in skip:
            return m.group(1)
    return ""


def _parse_java_file(content: str) -> list[dict]:
    """
    Parse one Java source file and return a list of test-case dicts
    {code, annotations, testname} for each enabled test method with @AccessMode.
    """
    class_name = _extract_class_name(content)
    clean = _remove_block_comments(content)
    orig_lines = content.split("\n")
    clean_lines = clean.split("\n")

    if _is_class_disabled(clean, class_name):
        logger.debug("  Class %s is @Disabled — skipping", class_name)
        return []

    tests = _find_test_blocks(clean_lines)
    results = []

    for idx, test in enumerate(tests):
        if test["is_disabled"] or not test["access_modes"]:
            continue

        # Code spans from code_start to just before the next test's annotation block
        code_start = test["code_start"]
        code_end = (
            tests[idx + 1]["block_start"] if idx + 1 < len(tests) else len(orig_lines)
        )

        code_lines = orig_lines[code_start:code_end]
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()

        code = "\n".join(code_lines).rstrip() + "\n"
        method_name = _extract_method_name(orig_lines[test["method_start"] : test["method_start"] + 8])
        if not method_name:
            continue

        results.append(
            {
                "code": code,
                "annotations": "\n".join(test["access_modes"]),
                "testname": method_name[0].upper() + method_name[1:],
            }
        )

    return results


def extract_test_cases_from_url(repo_url: str) -> tuple[list[dict], str | None]:
    """
    Given a GitHub repository URL, fetch all Java test files and return
    ``(test_cases, system_resources_content)``.

    *test_cases* — list of dicts ``{code, annotations, testname}`` for every enabled
    test method that has ``@AccessMode`` annotations.  Works regardless of the
    internal package/folder structure of the repository.

    *system_resources_content* — raw JSON string of the ``*SystemResources.json`` file
    found in ``.retorch/configurations/``, or ``None`` if absent.
    """
    owner, repo = _parse_github_url(repo_url)
    logger.info("── %s/%s ──────────────────────────────────────", owner, repo)
    results = []
    system_resources: str | None = None

    try:
        java_paths, sr_path = _list_java_test_files(owner, repo)
        logger.info("  Java test files found : %d", len(java_paths))
        if sr_path:
            logger.info("  SystemResources file  : %s", sr_path)
        else:
            logger.warning("  SystemResources file  : not found in .retorch/configurations/")
    except urllib.error.URLError:
        logger.exception("  Failed to list files for %s", repo)
        return [], None

    # Fetch SystemResources
    if sr_path:
        try:
            system_resources = _fetch_raw_file(owner, repo, sr_path)
        except Exception as exc:
            logger.warning("  Could not fetch %s: %s", sr_path, exc)

    # Fetch and parse Java test files
    per_file: dict[str, int] = {}
    for i, path in enumerate(java_paths, 1):
        logger.info("  [%d/%d] Parsing %s", i, len(java_paths), path.split("/")[-1])
        try:
            content = _fetch_raw_file(owner, repo, path)
            cases = _parse_java_file(content)
            if cases:
                per_file[path] = len(cases)
                logger.info("    → %d test case(s) found", len(cases))
            results.extend(cases)
        except Exception as exc:
            logger.warning("    → error: %s", exc)

    # Per-file breakdown at DEBUG only (avoid test-case data in INFO output)
    for path, count in per_file.items():
        logger.debug("  %-60s %3d test(s)", path.split("/")[-1], count)

    access_counts = [len(tc["annotations"].strip().splitlines()) for tc in results if tc["annotations"].strip()]
    logger.info(
        "  Done — %d test case(s) extracted, @AccessMode: min=%s max=%s avg=%s",
        len(results),
        min(access_counts) if access_counts else "n/a",
        max(access_counts) if access_counts else "n/a",
        f"{sum(access_counts)/len(access_counts):.1f}" if access_counts else "n/a",
    )

    return results, system_resources


def _repo_short_name(repo_url: str) -> str:
    """
    Derive a short identifier from a GitHub URL.

    The last segment of the URL is the repository name (e.g. ``retorch-st-fullteaching``).
    The part after the last ``-`` separator is returned as the short name, so that
    ``retorch-st-fullteaching`` → ``fullteaching``.
    If no ``-`` is present the full repository name is used.
    """
    repo_segment = repo_url.rstrip("/").split("/")[-1]
    # Keep everything after the last '-' delimiter
    parts = repo_segment.rsplit("-", 1)
    return parts[-1] if len(parts) > 1 else repo_segment


def _output_path_for(repo_url: str, context_dir: str = CONTEXT_DIR) -> str:
    """Return the output JSON path for a given repository URL."""
    return os.path.join(context_dir, f"ragtestcases_{_repo_short_name(repo_url)}.json")


def generate_ragtestcases(
    repo_urls: list[str] | None = None,
    context_dir: str = CONTEXT_DIR,
) -> dict[str, list[dict]]:
    """
    Fetch test cases and SystemResources from each repository URL and write one
    JSON file per repo to *context_dir*.

    Files written per repository:
    - ``ragtestcases_<shortname>.json``   — test cases (IDs reset per repo)
    - ``systemresources_<shortname>.json`` — SystemResources from .retorch/configurations/

    Returns a mapping of ``{short_name: [test_case_dicts]}``.
    """
    if repo_urls is None:
        repo_urls = REPOS

    os.makedirs(context_dir, exist_ok=True)
    results: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}

    for url in repo_urls:
        short = _repo_short_name(url)
        cases, system_resources = extract_test_cases_from_url(url)

        for i, tc in enumerate(cases, 1):
            tc["id"] = f"TC-{i:03d}"

        ordered = [
            {
                "id": tc["id"],
                "code": tc["code"],
                "annotations": tc["annotations"],
                "testname": tc["testname"],
            }
            for tc in cases
        ]

        # Write test cases
        tc_path = _output_path_for(url, context_dir)
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)
        logger.info("  Written: %s (%d test cases)", tc_path, len(ordered))

        # Write SystemResources and count resources
        resource_count: int | None = None
        if system_resources is not None:
            sr_path = os.path.join(context_dir, f"systemresources_{short}.json")
            with open(sr_path, "w", encoding="utf-8") as f:
                f.write(system_resources)
            logger.info("  Written: %s", sr_path)
            try:
                resource_count = len(json.loads(system_resources))
            except json.JSONDecodeError:
                logger.warning("  Could not parse SystemResources JSON for [%s]", short)
        else:
            logger.warning("  No SystemResources written for [%s]", short)

        results[short] = ordered
        summary[short] = {"test_cases": len(ordered), "resources": resource_count}

    _log_summary(summary)
    return results


def _log_summary(summary: dict[str, dict]) -> None:
    """Print a human-readable generation summary to the log."""
    sep = "─" * 54
    logger.info(sep)
    logger.info("  GENERATION SUMMARY")
    logger.info(sep)
    logger.info("  %-22s %12s %12s", "SUT", "Test cases", "Resources")
    logger.info("  %-22s %12s %12s", "─" * 22, "─" * 10, "─" * 10)
    for sut, data in summary.items():
        resources = str(data["resources"]) if data["resources"] is not None else "n/a"
        logger.info("  %-22s %12d %12s", sut, data["test_cases"], resources)
    logger.info(sep)


def load_all_ragtestcases(context_dir: str = CONTEXT_DIR) -> list[dict]:
    """
    Load and merge all ``ragtestcases_*.json`` files from *context_dir*.
    Used by core.py to build the full RAG knowledge base across all repos.
    """
    all_cases: list[dict] = []
    for fname in sorted(os.listdir(context_dir)):
        if fname.startswith("ragtestcases_") and fname.endswith(".json"):
            path = os.path.join(context_dir, fname)
            with open(path, encoding="utf-8") as f:
                all_cases.extend(json.load(f))
    return all_cases


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, "..", ".."))
    from ril2m.helpers.logging_config import setup_logging

    setup_logging()
    logger.info("Generating per-repo test case JSONs...")
    generate_ragtestcases()
    logger.info("Done.")

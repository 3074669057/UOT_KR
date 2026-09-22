# normalize_eol.py -- make every text file in the release LF-only.
#
# Why this is necessary
# --------------------
# The curated source tree inherited a mix of line endings: most files were LF, but files
# produced by Windows tooling (PowerShell Set-Content, some experiment writers) were CRLF.
# That mix plus a verifying machine's global `core.autocrlf=true` broke byte-exact
# verification: the clone rewrote the 1304 LF files to CRLF, so 633 files failed their
# SHA256 comparison against RELEASE_SHA256SUMS.txt even though Git content was identical.
#
# .gitattributes now pins `eol=lf`, and this script converts the working tree to match, so
# a checkout is byte-identical on every platform and the hashes always verify.
#
# Binary formats are skipped by extension.  Files that already contain a NUL byte are also
# skipped, as a cheap extra guard.
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

BINARY_EXT = {
    ".zip", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".npz", ".npy",
    ".pkl", ".bin", ".so", ".dll", ".exe", ".xlsx", ".gz", ".tar", ".woff", ".woff2",
    ".ico", ".ttf", ".otf", ".pyc",
}


def main() -> int:
    converted = 0
    skipped_binary = 0
    already_lf = 0
    total = 0

    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() in BINARY_EXT:
            skipped_binary += 1
            continue
        data = path.read_bytes()
        total += 1
        if b"\x00" in data[:8192]:
            skipped_binary += 1
            continue
        if b"\r\n" not in data:
            already_lf += 1
            continue
        # normalise: CRLF -> LF, and a lone CR -> LF
        new = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        path.write_bytes(new)
        converted += 1

    print(f"text files examined : {total}")
    print(f"  already LF        : {already_lf}")
    print(f"  converted to LF   : {converted}")
    print(f"  skipped (binary)  : {skipped_binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

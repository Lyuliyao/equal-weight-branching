#!/usr/bin/env python3
"""Apply the revised Section 5.2 localized-growth text to cmame-main.tex.

This helper replaces the current localized-growth subsection in
paper/cmame/cmame-main.tex by the standalone replacement stored in
paper/cmame/section5_2_revised.tex.  It is intentionally marker-based so it does not
touch the rest of the manuscript.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "cmame" / "cmame-main.tex"
SECTION = ROOT / "paper" / "cmame" / "section5_2_revised.tex"

START_MARKERS = [
    r"\subsection{Stationary localized growth: reaction representation at equal transport}",
    r"\subsection{Localized-growth benchmarks: weighted particles, resampling, and branching}",
]
END_MARKER = r"\subsection{Two-dimensional Keller--Segel: coupled solver and pre-singular concentration}"


def main() -> None:
    text = MAIN.read_text(encoding="utf-8")
    replacement = SECTION.read_text(encoding="utf-8").rstrip() + "\n\n"

    start = -1
    used_marker = None
    for marker in START_MARKERS:
        start = text.find(marker)
        if start >= 0:
            used_marker = marker
            break
    if start < 0:
        raise RuntimeError("Could not find the start of the localized-growth subsection.")

    end = text.find(END_MARKER, start)
    if end < 0:
        raise RuntimeError("Could not find the next Keller--Segel subsection marker.")

    new_text = text[:start] + replacement + text[end:]
    if new_text == text:
        print("No change: revised Section 5.2 is already installed.")
        return

    MAIN.write_text(new_text, encoding="utf-8")
    print(f"Replaced subsection starting at marker: {used_marker}")
    print(f"Updated: {MAIN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Before/after-oversampling class balance for the built TRAIN split.

Repeatable — re-run after any `make build` to see how oversampling reshapes the
distribution as datasets are added/curated/blacklisted:

    make balance            # or: python scripts/analyze_balance.py

"before" counts only original images; "after" counts everything written to
train (originals + `__osN` oversample copies). val/test are never oversampled,
so they're reported once as the honest eval distribution.
"""
from __future__ import annotations

import re
from collections import Counter

from common import FINAL, unified_classes

COPY = re.compile(r"__os\d+$")  # marks an oversample duplicate


def count_split(split: str):
    before, after = Counter(), Counter()
    n_before = n_after = 0
    d = FINAL / split / "labels"
    if not d.exists():
        return before, after, n_before, n_after
    for t in sorted(d.glob("*.txt")):
        is_copy = bool(COPY.search(t.stem))
        n_after += 1
        if not is_copy:
            n_before += 1
        for ln in t.read_text().splitlines():
            if not ln.strip():
                continue
            c = int(ln.split()[0])
            after[c] += 1
            if not is_copy:
                before[c] += 1
    return before, after, n_before, n_after


def main() -> None:
    names = unified_classes()
    before, after, ib, ia = count_split("train")
    tb = sum(before.values()) or 1
    ta = sum(after.values()) or 1

    print("=" * 64)
    print("TRAIN class balance — before vs after oversampling")
    print("=" * 64)
    print(f"images: {ib:>7} -> {ia:>7}   (+{ia - ib} oversample copies)")
    print(f"boxes:  {tb:>7} -> {ta:>7}")
    print(f"\n{'class':<15}{'before':>9}{'b%':>7}{'after':>9}{'a%':>7}{'x':>6}")
    print("-" * 53)
    for c in sorted(range(len(names)), key=lambda i: -after.get(i, 0)):
        b, a = before.get(c, 0), after.get(c, 0)
        if a == 0:
            continue
        mult = (a / b) if b else 0.0
        flag = "  <-- oversampled" if round(mult, 1) > 1.0 else ""
        print(f"{names[c]:<15}{b:>9}{100*b/tb:>6.1f}{a:>9}{100*a/ta:>6.1f}{mult:>6.1f}{flag}")

    for split in ("val", "test"):
        _, af, _, ia2 = count_split(split)
        print(f"\n{split}: {ia2} images, {sum(af.values())} boxes "
              f"(no oversampling — honest eval distribution)")


if __name__ == "__main__":
    main()

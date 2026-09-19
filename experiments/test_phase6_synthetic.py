"""Synthetic regression check for Phase 6: adaptive key resizing and
suggestion acceptance. No webcam needed.

Run from the repo root:

    .venv/Scripts/python.exe -m experiments.test_phase6_synthetic
"""

from __future__ import annotations

import sys

from backend.adaptive_layout import accept_suggestion, compute_adaptive_layout
from decoder.char_lm import CharLM
from decoder.nearest_key import KeyboardLayout
from decoder.text_buffer import TextBuffer


def check_row_bounds_preserved(layout: KeyboardLayout, lm: CharLM) -> bool:
    adaptive = compute_adaptive_layout(layout, lm, "th")
    for y in {k.y for k in layout.keys}:
        orig_row = sorted([k for k in layout.keys if k.y == y], key=lambda k: k.x)
        new_row = sorted([k for k in adaptive.keys if k.y == y], key=lambda k: k.x)
        orig_left, orig_right = orig_row[0].x, orig_row[-1].x + orig_row[-1].width
        new_left, new_right = new_row[0].x, new_row[-1].x + new_row[-1].width
        if abs(orig_left - new_left) > 1e-9 or abs(orig_right - new_right) > 1e-9:
            print(f"[FAIL] row y={y} bounds changed: ({orig_left},{orig_right}) -> ({new_left},{new_right})")
            return False
        # no gaps or overlaps within the row
        for i in range(len(new_row) - 1):
            if abs((new_row[i].x + new_row[i].width) - new_row[i + 1].x) > 1e-9:
                print(f"[FAIL] row y={y} has a gap/overlap between {new_row[i].id} and {new_row[i+1].id}")
                return False
    print("[OK] adaptive layout preserves each row's overall bounds, no gaps/overlaps")
    return True


def check_likely_key_grows(layout: KeyboardLayout, lm: CharLM) -> bool:
    adaptive = compute_adaptive_layout(layout, lm, "th")
    e_key = next(k for k in adaptive.keys if k.id == "E")
    q_key = next(k for k in adaptive.keys if k.id == "Q")
    orig_e = next(k for k in layout.keys if k.id == "E")
    if not (e_key.width > orig_e.width):
        print(f"[FAIL] E (highly likely after 'th') should grow: {orig_e.width} -> {e_key.width}")
        return False
    if not (e_key.width > q_key.width):
        print(f"[FAIL] E should end up bigger than Q after 'th': E={e_key.width} Q={q_key.width}")
        return False
    print(f"[OK] likely next key (E after 'th') grows relative to both its own baseline and unlikely keys")
    return True


def check_no_key_vanishes(layout: KeyboardLayout, lm: CharLM) -> bool:
    adaptive = compute_adaptive_layout(layout, lm, "zzq")  # an unusual context
    if any(k.width <= 0.0 for k in adaptive.keys):
        print("[FAIL] a key shrank to zero or negative width")
        return False
    print("[OK] no key ever shrinks to zero width, even for an unusual context")
    return True


def check_accept_suggestion_replaces_word() -> bool:
    buffer = TextBuffer()
    for key in ["H", "E", "L", "L"]:
        buffer.apply_key(key)
    buffer.apply_key("SPACE")  # S0's normal flow: SPACE already applied -> "hell "
    if buffer.text != "hell ":
        print(f"[FAIL] setup wrong: expected 'hell ', got {buffer.text!r}")
        return False

    accept_suggestion(buffer, pending_prefix="hell", suggestion="hello")
    if buffer.text != "hello ":
        print(f"[FAIL] expected 'hello ', got {buffer.text!r}")
        return False
    print("[OK] accept_suggestion replaces the in-progress word with the suggestion")
    return True


def check_accept_suggestion_mid_sentence() -> bool:
    buffer = TextBuffer()
    for key in ["I", "SPACE", "N", "E", "E", "D"]:
        buffer.apply_key(key)
    buffer.apply_key("SPACE")  # "i need "
    accept_suggestion(buffer, pending_prefix="need", suggestion="need")  # identical -- should be a no-op-ish replace
    if buffer.text != "i need ":
        print(f"[FAIL] expected 'i need ', got {buffer.text!r}")
        return False
    print("[OK] accept_suggestion works correctly mid-sentence, preceding words untouched")
    return True


def main() -> int:
    layout = KeyboardLayout.from_json()
    lm = CharLM.load_default()

    checks = [
        lambda: check_row_bounds_preserved(layout, lm),
        lambda: check_likely_key_grows(layout, lm),
        lambda: check_no_key_vanishes(layout, lm),
        check_accept_suggestion_replaces_word,
        check_accept_suggestion_mid_sentence,
    ]
    results = [check() for check in checks]
    if all(results):
        print(f"\n[OK] all {len(results)} Phase 6 checks passed")
        return 0
    print(f"\n[FAIL] {results.count(False)}/{len(results)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Text buffer shared by all S0-S4 systems: turns a sequence of selected key
ids into text. SPACE and BACKSPACE are handled specially; any other key id is
appended lowercased, matching config/vocabulary.txt and
config/evaluation_phrases.txt conventions.
"""

from __future__ import annotations


class TextBuffer:
    def __init__(self) -> None:
        self._chars: list[str] = []

    def reset(self) -> None:
        self._chars.clear()

    def apply_key(self, key_id: str) -> None:
        if key_id == "SPACE":
            self._chars.append(" ")
        elif key_id == "BACKSPACE":
            if self._chars:
                self._chars.pop()
        else:
            self._chars.append(key_id.lower())

    @property
    def text(self) -> str:
        return "".join(self._chars)

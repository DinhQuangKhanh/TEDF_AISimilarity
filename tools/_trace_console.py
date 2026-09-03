"""Khung in ấn dùng chung cho các script "xem AI làm gì" (``trace_tuning.py``, ``trace_pipeline.py``).

Mọi bước đều in theo cùng một dạng, để đọc quen mắt:

    ══════════════════════════════════════════
     BƯỚC 3/8 · Tên bước
    ══════════════════════════════════════════
      INPUT     │ …
      CÔNG THỨC │ …
      OUTPUT    │ …
"""

from __future__ import annotations

WIDTH = 92
_LABEL = 10


def banner(*lines: str) -> None:
    print("╔" + "═" * (WIDTH - 2) + "╗")
    for line in lines:
        print("║" + line.center(WIDTH - 2) + "║")
    print("╚" + "═" * (WIDTH - 2) + "╝")


def head(n: int, total: int, title: str) -> None:
    print("\n" + "═" * WIDTH)
    print(f" BƯỚC {n}/{total} · {title}")
    print("═" * WIDTH)


def io(label: str, *lines: str) -> None:
    """In một khối nhãn → nội dung; các dòng sau thẳng hàng với dòng đầu."""
    if not lines:
        return
    print(f"  {label:<{_LABEL}}│ " + f"\n  {'':<{_LABEL}}│ ".join(lines))


def note(*lines: str) -> None:
    io("", *lines)


def rule(indent: int = 4) -> None:
    print(" " * indent + "─" * (WIDTH - indent - 2))


def bar(value: float, width: int = 28, lo: float = 0.0, hi: float = 1.0) -> str:
    span = max(hi - lo, 1e-9)
    filled = max(0, min(width, round((value - lo) / span * width)))
    return "█" * filled + "·" * (width - filled)


def wtuple(weights) -> str:
    return "(" + ", ".join(f"{x:.2f}" for x in weights) + ")"


def status_mark(status: str) -> str:
    return {"success": "✓", "warning": "!", "error": "✕"}.get(status, "·")


def wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Ngắt dòng thủ công (không dùng textwrap để giữ nguyên các ký tự · │ trong nội dung)."""
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(indent + current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(indent + current)
    return lines or [indent]

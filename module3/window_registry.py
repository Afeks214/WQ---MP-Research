from __future__ import annotations

WINDOWS: tuple[int, ...] = (5, 15, 30, 60)


def normalize_structural_windows(
    structural_windows: tuple[int, ...] | list[int],
    *,
    fallback: tuple[int, ...] = WINDOWS,
) -> tuple[int, ...]:
    src = tuple(int(x) for x in structural_windows)
    if len(src) == 0:
        src = tuple(int(x) for x in fallback)
    out: list[int] = []
    seen: set[int] = set()
    for w in src:
        if w <= 0:
            raise RuntimeError(f"Invalid structural window: {w}")
        if w in seen:
            continue
        seen.add(w)
        out.append(int(w))
    if not out:
        raise RuntimeError("structural_windows must be non-empty")
    return tuple(out)


def build_window_index_map(structural_windows: tuple[int, ...] | list[int]) -> dict[int, int]:
    wins = normalize_structural_windows(structural_windows)
    return {int(w): int(i) for i, w in enumerate(wins)}


def resolve_window_index(selected_window: int, structural_windows: tuple[int, ...] | list[int]) -> int:
    idx_map = build_window_index_map(structural_windows)
    sw = int(selected_window)
    if sw not in idx_map:
        raise RuntimeError(
            f"selected_window={sw} is not present in structural_windows={tuple(int(x) for x in structural_windows)}"
        )
    return int(idx_map[sw])

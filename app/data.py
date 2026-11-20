"""Generate and prepare reproducible basketball possession sequences."""
from __future__ import annotations
import csv
import random
from pathlib import Path

ACTIONS = [
    "defensive_rebound", "outlet_pass", "transition_dribble", "pick_and_roll",
    "screen", "drive", "kick_out", "three_point_attempt", "three_point_make",
    "mid_range_attempt", "mid_range_make", "layup_attempt", "layup_make",
    "turnover", "steal", "foul", "free_throw_make", "offensive_rebound",
    "post_up", "assist"
]

PATTERNS = [
    (["defensive_rebound", "outlet_pass", "transition_dribble"], "layup_attempt"),
    (["pick_and_roll", "screen", "drive", "kick_out"], "three_point_attempt"),
    (["post_up", "drive"], "foul"),
    (["three_point_attempt"], "defensive_rebound"),
    (["layup_attempt"], "offensive_rebound"),
    (["steal", "transition_dribble"], "layup_attempt"),
]

def generate_games(path: str | Path, games: int = 240, seed: int = 42) -> Path:
    """Create synthetic possession-level sequences with planted basketball patterns."""
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for game_id in range(games):
        for possession in range(1, 21):
            pattern, outcome = rng.choice(PATTERNS)
            events = list(pattern)
            if rng.random() < 0.72:
                events.append(outcome)
            else:
                events.append(rng.choice(["turnover", "mid_range_attempt", "layup_attempt"]))
            if rng.random() < 0.25:
                events.insert(0, rng.choice(["steal", "defensive_rebound", "foul"]))
            for step, action in enumerate(events, 1):
                rows.append({
                    "game_id": game_id,
                    "possession_id": f"{game_id}-{possession}",
                    "step": step,
                    "action": action
                })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["game_id", "possession_id", "step", "action"])
        writer.writeheader()
        writer.writerows(rows)
    return path

def load_sequences(path: str | Path) -> list[list[str]]:
    """Read play-by-play rows and return ordered event baskets per possession."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["possession_id"], []).append((int(row["step"]), row["action"]))
    return [[action for _, action in sorted(events)] for _, events in sorted(grouped.items())]

def split_sequences(sequences: list[list[str]], test_fraction: float = 0.2):
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    cut = max(1, int(len(sequences) * (1 - test_fraction)))
    return sequences[:cut], sequences[cut:]

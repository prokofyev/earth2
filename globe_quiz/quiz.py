"""Проход уровня: вопросы, попытки, звёзды. Без pygame — чтобы легко тестировать."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from . import DATA_DIR

MAX_QUESTIONS = 8
MAX_ATTEMPTS = 3
STARS_FOR_ATTEMPT = {1: 3, 2: 2, 3: 1}
PASS_RATIO = 2 / 3
HINT_AFTER_MISSES = 2


@dataclass(frozen=True)
class Question:
    text: str
    accept: tuple[str, ...]
    fact: str | None = None
    describes: bool = False  # вопрос-описание (рассуждение), объект в тексте не назван


@dataclass(frozen=True)
class Level:
    id: int  # место в списке уровней (1, 2, 3…) — для показа в меню
    key: str  # постоянное имя: к нему привязан прогресс, номер может меняться
    title: str
    questions: tuple[Question, ...]
    hint_highlight: bool = False
    riddles: bool = False


def load_levels(path: Path = DATA_DIR / "levels.json") -> list[Level]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    levels = []
    for n, raw in enumerate(data["levels"], start=1):
        if raw["id"] != n:
            raise ValueError(f"уровень «{raw['title']}»: номер {raw['id']}, а по порядку должен быть {n}")
        if any(lv.key == raw["key"] for lv in levels):
            raise ValueError(f"повторяется имя уровня {raw['key']!r}")
        questions = tuple(
            Question(q["text"], tuple(q["accept"]), q.get("fact"), q.get("describes", False))
            for q in raw["questions"]
        )
        levels.append(
            Level(
                id=raw["id"],
                key=raw["key"],
                title=raw["title"],
                questions=questions,
                hint_highlight=raw.get("hint_highlight", False),
                riddles=raw.get("riddles", False),
            )
        )
    return levels


def next_level(levels: list[Level], level: Level) -> Level | None:
    """Следующий уровень по порядку или None для последнего."""
    i = levels.index(level)
    return levels[i + 1] if i + 1 < len(levels) else None


def previous_level(levels: list[Level], level: Level) -> Level | None:
    i = levels.index(level)
    return levels[i - 1] if i > 0 else None


class Result(Enum):
    CORRECT = "correct"
    MISS = "miss"  # можно попробовать ещё
    FAILED = "failed"  # попытки кончились


@dataclass(frozen=True)
class Outcome:
    result: Result
    stars: int
    attempts_left: int
    show_hint: bool


class LevelRun:
    """Один проход уровня."""

    def __init__(self, level: Level, rng: random.Random | None = None,
                 max_questions: int = MAX_QUESTIONS):
        rng = rng or random.Random()
        count = min(max_questions, len(level.questions))
        self.level = level
        self.questions: list[Question] = rng.sample(list(level.questions), count)
        self.index = 0
        self.misses = 0
        self.resolved = False  # на текущий вопрос уже ответили (или показали ответ)
        self.stars: list[int] = []

    @property
    def current(self) -> Question:
        return self.questions[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.questions)

    @property
    def is_last(self) -> bool:
        return self.index == len(self.questions) - 1

    @property
    def total_stars(self) -> int:
        return sum(self.stars)

    @property
    def max_stars(self) -> int:
        return 3 * len(self.questions)

    @property
    def passed(self) -> bool:
        return self.total_stars >= self.max_stars * PASS_RATIO - 1e-9

    @property
    def show_hint(self) -> bool:
        return self.level.hint_highlight and self.misses >= HINT_AFTER_MISSES and not self.resolved

    def submit(self, correct: bool) -> Outcome:
        if self.finished or self.resolved:
            raise RuntimeError("на этот вопрос уже ответили")
        if correct:
            stars = STARS_FOR_ATTEMPT[self.misses + 1]
            self.stars.append(stars)
            self.resolved = True
            return Outcome(Result.CORRECT, stars, MAX_ATTEMPTS - self.misses - 1, False)
        self.misses += 1
        if self.misses >= MAX_ATTEMPTS:
            self.stars.append(0)
            self.resolved = True
            return Outcome(Result.FAILED, 0, 0, False)
        return Outcome(Result.MISS, 0, MAX_ATTEMPTS - self.misses, self.show_hint)

    def next_question(self) -> None:
        if not self.resolved:
            raise RuntimeError("вопрос ещё не решён")
        self.index += 1
        self.misses = 0
        self.resolved = False

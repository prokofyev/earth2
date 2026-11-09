"""Игроки и их прогресс: лучшие результаты уровней и открытые уровни."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import PROJECT_DIR

PROGRESS_PATH = PROJECT_DIR / "progress.json"
VERSION = 3  # 3 — восемь уровней и рельефная модель; файлы прежних версий не переносятся
MAX_NAME_LENGTH = 16


@dataclass
class Progress:
    """Прогресс одного игрока."""

    best: dict[int, int] = field(default_factory=dict)
    unlocked: int = 1  # номер последнего открытого уровня

    def is_unlocked(self, level_id: int) -> bool:
        return level_id <= self.unlocked

    def best_for(self, level_id: int) -> int:
        return self.best.get(level_id, 0)

    def record(self, level_id: int, stars: int, passed: bool, level_count: int) -> None:
        """Записать завершённый проход уровня."""
        self.best[level_id] = max(self.best.get(level_id, 0), stars)
        if passed:
            self.unlocked = max(self.unlocked, min(level_id + 1, level_count))

    def to_json(self) -> dict:
        return {"best": {str(k): v for k, v in self.best.items()}, "unlocked": self.unlocked}

    @classmethod
    def from_json(cls, data: dict) -> "Progress":
        best = {int(k): int(v) for k, v in data.get("best", {}).items()}
        unlocked = int(data.get("unlocked", 1))
        if unlocked < 1 or any(v < 0 for v in best.values()):
            raise ValueError("неверные данные прогресса")
        return cls(best=best, unlocked=unlocked)


def clean_name(name: str) -> str:
    """Имя без лишних пробелов, не длиннее MAX_NAME_LENGTH."""
    return " ".join(name.split())[:MAX_NAME_LENGTH]


@dataclass
class Players:
    """Все игроки, их прогресс и текущий игрок."""

    players: dict[str, Progress] = field(default_factory=dict)
    current: str | None = None

    @property
    def names(self) -> list[str]:
        return list(self.players)

    @property
    def progress(self) -> Progress | None:
        return self.players.get(self.current) if self.current else None

    def find(self, name: str) -> str | None:
        """Существующее имя без учёта регистра."""
        key = clean_name(name).casefold()
        return next((n for n in self.players if n.casefold() == key), None)

    def add(self, name: str) -> str:
        """Создать игрока и сделать его текущим. ValueError — пустое или занятое имя."""
        name = clean_name(name)
        if not name:
            raise ValueError("Напиши имя")
        if self.find(name) is not None:
            raise ValueError(f"Игрок «{self.find(name)}» уже есть")
        self.players[name] = Progress()
        self.current = name
        return name

    def select(self, name: str) -> None:
        if name not in self.players:
            raise KeyError(name)
        self.current = name

    def to_json(self) -> dict:
        return {
            "version": VERSION,
            "current": self.current,
            "players": {n: p.to_json() for n, p in self.players.items()},
        }

    @classmethod
    def from_json(cls, data: dict) -> "Players":
        if data.get("version") != VERSION:
            # Прогресс прежних версий игры (семь уровней, старая модель Земли) не переносим:
            # начинаем с чистого листа, а старый файл заменится при первом сохранении.
            raise ValueError("файл прогресса другой версии")
        players = {clean_name(str(n)): Progress.from_json(p) for n, p in data.get("players", {}).items()}
        players.pop("", None)
        current = data.get("current")
        return cls(players, current if current in players else None)


def load_players(path: Path | None = None) -> Players:
    """Загрузить игроков. Нет файла или он испорчен — начинаем без игроков."""
    path = PROGRESS_PATH if path is None else path
    try:
        return Players.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, AttributeError):
        return Players()


def save_players(players: Players, path: Path | None = None) -> None:
    """Атомарная запись: сначала во временный файл, потом замена."""
    path = Path(PROGRESS_PATH if path is None else path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".progress-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(players.to_json(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

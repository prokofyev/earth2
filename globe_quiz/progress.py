"""Игроки и их прогресс: лучшие результаты уровней и открытые уровни."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import PROJECT_DIR

PROGRESS_PATH = PROJECT_DIR / "progress.json"
VERSION = 4  # 4 — прогресс по постоянным именам уровней; 3 переносится, 1 и 2 — нет
MAX_NAME_LENGTH = 16


# Перенос версии 3 (восемь уровней с номерами): старый номер → новые имена уровней.
# Звёзды переносятся только у уровней, которые не делились.
V3_LEVELS = {
    1: ["continents"],
    2: ["oceans"],
    3: ["poles_equator"],
    4: ["islands", "peninsulas"],
    5: ["seas", "gulfs"],
    6: ["mountains", "plains_deserts"],
    7: ["peaks"],
    8: ["riddles"],
}


@dataclass
class Progress:
    """Прогресс одного игрока: лучшие звёзды и открытые уровни по постоянным именам."""

    best: dict[str, int] = field(default_factory=dict)
    opened: set[str] = field(default_factory=set)  # первый уровень открыт всегда, сюда не пишется

    def is_unlocked(self, level) -> bool:
        return level.id == 1 or level.key in self.opened

    def best_for(self, key: str) -> int:
        return self.best.get(key, 0)

    def record(self, key: str, stars: int, passed: bool, next_key: str | None) -> None:
        """Записать завершённый проход уровня; при прохождении открыть следующий."""
        self.best[key] = max(self.best.get(key, 0), stars)
        if passed and next_key is not None:
            self.opened.add(next_key)

    def to_json(self) -> dict:
        return {"best": dict(self.best), "opened": sorted(self.opened)}

    @classmethod
    def from_json(cls, data: dict) -> "Progress":
        best = {str(k): int(v) for k, v in data.get("best", {}).items()}
        opened = {str(k) for k in data.get("opened", [])}
        if any(v < 0 for v in best.values()):
            raise ValueError("неверные данные прогресса")
        return cls(best=best, opened=opened)

    @classmethod
    def from_v3(cls, data: dict) -> "Progress":
        """Прогресс версии 3: номера восьми уровней → постоянные имена."""
        best_v3 = {int(k): int(v) for k, v in data.get("best", {}).items()}
        unlocked = int(data.get("unlocked", 1))
        if unlocked < 1 or any(v < 0 for v in best_v3.values()):
            raise ValueError("неверные данные прогресса")
        best = {V3_LEVELS[n][0]: v for n, v in best_v3.items() if n in V3_LEVELS and len(V3_LEVELS[n]) == 1}
        # Открыты все новые уровни, выделенные из открытых старых (первый открыт и так).
        opened = {key for n, keys in V3_LEVELS.items() if 1 < n <= unlocked for key in keys}
        return cls(best=best, opened=opened)


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
        version = data.get("version")
        if version == 3:  # восемь уровней с номерами — переносим по именам
            read = Progress.from_v3
        elif version == VERSION:
            read = Progress.from_json
        else:
            # Прогресс самых старых версий (семь уровней, старая модель Земли) не переносим:
            # начинаем с чистого листа, а старый файл заменится при первом сохранении.
            raise ValueError("файл прогресса другой версии")
        players = {clean_name(str(n)): read(p) for n, p in data.get("players", {}).items()}
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

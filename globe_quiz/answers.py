"""Проверка тыка по глобусу и тексты ответа игры."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .geo import angular_distance, direction_hint
from .model import SURFACE_WATER
from .quiz import Question
from .regions import Fact, Region, RegionSet
from .renderer import GlobeRenderer

HIT_TOLERANCE_PX = 22
POINT_TOLERANCE_DEG = 12.0  # полюса: тык не дальше 12° от полюса
PEAK_TOLERANCE_DEG = 5.0  # вершины: на модели нет отдельных конусов — тык не дальше 5°
LINE_TOLERANCE_DEG = 8.0  # экватор: не дальше 8° по широте
# Для этих объектов промах «рядом» засчитывается: они маленькие или узкие.
TOLERANT_KINDS = {"island", "peninsula", "sea", "gulf", "lake", "mountains", "desert", "plain",
                  "point", "peak", "line"}

PRAISE = ["Молодец!", "Отлично!", "Здорово!", "Так держать!", "Верно!", "Ты настоящий географ!"]


@dataclass
class Tap:
    face: int
    latlon: tuple[float, float]
    regions: list[Region]
    specific: Region | None
    is_water: bool


def cap(text: str) -> str:
    """Первая буква — заглавная (названия хранятся так, как пишутся внутри фразы)."""
    return text[:1].upper() + text[1:]


def inspect_tap(renderer: GlobeRenderer, regions: RegionSet, pos) -> Tap | None:
    face = renderer.pick_face(pos)
    if face is None:
        return None
    latlon = renderer.screen_to_latlon(pos)
    return Tap(
        face=face,
        latlon=latlon,
        regions=regions.regions_of_face(face),
        specific=regions.most_specific(face),
        is_water=bool(renderer.model.surface[face] == SURFACE_WATER),
    )


def region_hit(renderer: GlobeRenderer, region: Region, tap: Tap, pos,
               tolerance: float = HIT_TOLERANCE_PX) -> bool:
    if region.kind in ("point", "peak"):
        # У полюсов нет видимых ориентиров, а вершины не выглядят отдельными конусами:
        # засчитываем по углу на сфере (у полюсов допуск щедрее).
        limit = POINT_TOLERANCE_DEG if region.kind == "point" else PEAK_TOLERANCE_DEG
        return (angular_distance(tap.latlon, region.at) <= limit
                or renderer.distance_to_points(pos, *region.at) <= tolerance)
    if region.kind == "line":
        return (abs(tap.latlon[0] - region.lat) <= LINE_TOLERANCE_DEG
                or renderer.distance_to_parallel(pos, region.lat) <= tolerance)
    if any(r.id == region.id for r in tap.regions):
        return True
    if region.kind in TOLERANT_KINDS:
        return renderer.distance_to_faces(pos, region.faces) <= tolerance
    return False


def check_answer(renderer: GlobeRenderer, regions: RegionSet, question: Question,
                 tap: Tap, pos) -> Region | None:
    """Найденный объект, если ответ верный, иначе None."""
    for rid in question.accept:
        region = regions[rid]
        if region_hit(renderer, region, tap, pos):
            return region
    return None


def nearest_target(regions: RegionSet, question: Question, latlon) -> tuple[Region, tuple[float, float]]:
    """Ближайший из допустимых ответов и его ближайшая точка — для подсказки."""
    best = None
    for rid in question.accept:
        region = regions[rid]
        point = regions.nearest_target_point(region, latlon)
        d = angular_distance(latlon, point)
        if best is None or d < best[0]:
            best = (d, region, point)
    return best[1], best[2]


def miss_message(regions: RegionSet, question: Question, tap: Tap, riddle: bool) -> str:
    if tap.specific is not None:
        first = f"Это {tap.specific.name}."
    else:
        first = "Это вода." if tap.is_water else "Это суша."
    target, point = nearest_target(regions, question, tap.latlon)
    direction = direction_hint(tap.latlon, point)
    if direction is None:
        second = "Совсем рядом, посмотри внимательнее!"
    elif riddle or len(question.accept) > 1:
        second = f"Ищи {direction}!"
    else:
        second = f"{cap(target.name)} — {direction}!"
    return f"{first} {second}"


class FactPicker:
    """Выбирает факт об объекте без повторов, пока не показаны все его факты.

    У каждого объекта — перемешанный «мешок» номеров фактов. Когда он пустеет,
    набирается новый, и первым в нём не идёт только что показанный факт.
    Один выборщик живёт весь запуск игры.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self._bags: dict[str, list[int]] = {}
        self._last: dict[str, int] = {}

    def pick(self, region: Region) -> Fact:
        if not region.facts:
            return Fact("", "")
        bag = self._bags.get(region.id)
        if not bag:
            bag = list(range(len(region.facts)))
            self.rng.shuffle(bag)
            if len(bag) > 1 and bag[-1] == self._last.get(region.id):
                bag[0], bag[-1] = bag[-1], bag[0]  # берём с конца — не повторяем последний
            self._bags[region.id] = bag
        i = bag.pop()
        self._last[region.id] = i
        return region.facts[i]


def _fact_for(region: Region, question: Question, picker: FactPicker | None) -> Fact:
    """У вопроса на рассуждение — его пояснение, иначе случайный факт об объекте."""
    if question.fact:
        return Fact("", question.fact)
    return (picker or FactPicker()).pick(region)


def correct_message(region: Region, question: Question, picker: FactPicker | None = None,
                    rng: random.Random | None = None) -> tuple[str, Fact]:
    """Похвала с названием и факт для правильного ответа."""
    rng = rng or random
    title = f"{rng.choice(PRAISE)} Это {region.name}."
    return title, _fact_for(region, question, picker)


def reveal_message(region: Region, question: Question, picker: FactPicker | None = None) -> tuple[str, Fact]:
    """Название показанного объекта и факт о нём."""
    return f"Вот {region.name}.", _fact_for(region, question, picker)

"""Проверка тыка по глобусу и тексты ответа игры."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .geo import angular_distance, direction_hint
from .model import SURFACE_WATER
from .quiz import Question
from .regions import Region, RegionSet
from .renderer import GlobeRenderer

HIT_TOLERANCE_PX = 22
POINT_TOLERANCE_DEG = 12.0  # полюса: тык не дальше 12° от полюса
LINE_TOLERANCE_DEG = 8.0  # экватор: не дальше 8° по широте
# Для этих объектов промах «рядом» засчитывается: они маленькие или узкие.
TOLERANT_KINDS = {"island", "peninsula", "sea", "gulf", "lake", "mountains", "desert", "point", "line"}

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
    if region.kind == "point":
        # У полюсов нет видимых ориентиров, поэтому допуск щедрый: по углу на сфере.
        return (angular_distance(tap.latlon, region.at) <= POINT_TOLERANCE_DEG
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


def correct_message(region: Region, question: Question, rng: random.Random | None = None) -> tuple[str, str]:
    """Заголовок и пояснение для правильного ответа."""
    rng = rng or random
    title = f"{rng.choice(PRAISE)} Это {region.name}."
    return title, question.fact or region.fact


def reveal_message(region: Region, question: Question) -> tuple[str, str]:
    return f"Вот {region.name}.", question.fact or region.fact

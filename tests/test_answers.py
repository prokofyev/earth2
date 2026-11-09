"""Проверка ответов на настоящем глобусе, включая прогон всех вопросов всех уровней."""

import numpy as np
import pygame
import pytest

from globe_quiz.answers import check_answer, correct_message, inspect_tap, miss_message, reveal_message
from globe_quiz.quiz import Question, load_levels
from globe_quiz.regions import RegionSet
from globe_quiz.renderer import GlobeRenderer, View

SIZE = 700


@pytest.fixture(scope="module")
def regions(earth):
    return RegionSet.load().assign(earth)


@pytest.fixture(scope="module")
def levels():
    return load_levels()


def _render(earth, lat, lon, zoom=1.0):
    r = GlobeRenderer(earth, View(lat, lon, zoom))
    r.view.clamp()
    surf = pygame.Surface((SIZE, SIZE))
    r.draw(surf, surf.get_rect())
    return r


def _screen_point_for(r, region):
    """Экранная точка, куда надо ткнуть, чтобы попасть в регион."""
    if region.kind in ("point", "peak"):
        xy, vis = r.project_latlon(*region.at)
        return tuple(xy)
    if region.kind == "line":
        return (SIZE / 2, r.project_latlon(region.lat, r.view.center_lon)[0][1])
    fr = r.frame
    sel = region.faces[fr.order]
    # Грань цели, которая видна сверху в своём центре и ближе всего к центру экрана.
    best, best_d = None, np.inf
    for k in np.nonzero(sel)[0]:
        c = fr.pts[k].mean(axis=0)
        if r.pick_face(c) == fr.order[k]:
            d = np.hypot(*(c - SIZE / 2))
            if d < best_d:
                best, best_d = c, d
    assert best is not None, region.id
    return tuple(best)


def test_every_question_can_be_answered(earth, regions, levels):
    for level in levels:
        for q in level.questions:
            target = regions[q.accept[0]]
            lat, lon = target.center
            zoom = 3.0 if target.radius_deg and target.radius_deg < 12 else 1.0
            r = _render(earth, lat, lon, zoom)
            pos = _screen_point_for(r, target)
            tap = inspect_tap(r, regions, pos)
            assert tap is not None, q.text
            found = check_answer(r, regions, q, tap, pos)
            assert found is not None and found.id in q.accept, q.text


def test_coast_water_is_not_the_continent(earth, regions):
    r = _render(earth, 0, 20)
    q = Question("Найди Африку", ("africa",))
    xy, _ = r.project_latlon(-5, 8)  # Атлантика у берега Африки
    tap = inspect_tap(r, regions, tuple(xy))
    assert tap.is_water
    assert check_answer(r, regions, q, tap, tuple(xy)) is None


def test_near_small_island_counts(earth, regions):
    r = _render(earth, 8, 80, zoom=2.0)
    q = Question("Найди остров Шри-Ланка", ("sri_lanka",))
    xy, _ = r.project_latlon(7.5, 83.0)  # вода к востоку от острова
    pos = tuple(xy)
    tap = inspect_tap(r, regions, pos)
    assert tap.is_water
    assert check_answer(r, regions, q, tap, pos).id == "sri_lanka"


def test_far_from_small_island_misses(earth, regions):
    r = _render(earth, 8, 80, zoom=2.0)
    q = Question("Найди остров Шри-Ланка", ("sri_lanka",))
    pos = tuple(r.project_latlon(0, 88)[0])
    assert check_answer(r, regions, q, inspect_tap(r, regions, pos), pos) is None


def test_any_of_several_answers(earth, regions):
    r = _render(earth, -10, -60)
    q = Question("Найди материк, через который проходит экватор", ("africa", "south_america"))
    pos = tuple(r.project_latlon(-10, -55)[0])
    assert check_answer(r, regions, q, inspect_tap(r, regions, pos), pos).id == "south_america"


def test_miss_message_names_place_and_direction(earth, regions):
    r = _render(earth, -10, -40)
    q = Question("Найди Африку", ("africa",))
    pos = tuple(r.project_latlon(-10, -55)[0])
    msg = miss_message(regions, q, inspect_tap(r, regions, pos), riddle=False)
    assert msg.startswith("Это Южная Америка.")
    assert "Африка — " in msg and "восточнее" in msg


def test_miss_message_most_specific_place(earth, regions):
    r = _render(earth, 42, 5, zoom=3)
    q = Question("Найди Апеннинский полуостров", ("apennine_peninsula",))
    pos = tuple(r.project_latlon(42, -5)[0])
    msg = miss_message(regions, q, inspect_tap(r, regions, pos), riddle=False)
    assert msg.startswith("Это Пиренейский полуостров.") and "восточнее" in msg


def test_riddle_hint_does_not_reveal_answer(earth, regions):
    r = _render(earth, -10, -40)
    q = Question("Найди самый жаркий материк", ("africa",))
    pos = tuple(r.project_latlon(-10, -55)[0])
    msg = miss_message(regions, q, inspect_tap(r, regions, pos), riddle=True)
    assert "Африка" not in msg and "Ищи" in msg


def test_correct_and_reveal_messages(regions):
    q = Question("Найди Каспийское море", ("caspian_sea",))
    title, fact = correct_message(regions["caspian_sea"], q)
    assert "Каспийское море" in title and "самое большое озеро" in fact
    title, _ = reveal_message(regions["pacific_ocean"], q)
    assert title == "Вот Тихий океан."


@pytest.mark.parametrize("lat,lon", [(80, 30), (80, -100)])  # в 10° от полюса, как в сценарии
def test_roughly_at_north_pole_counts(earth, regions, lat, lon):
    r = _render(earth, 70, lon)
    q = Question("Найди Северный полюс", ("north_pole",))
    pos = tuple(r.project_latlon(lat, lon)[0])
    tap = inspect_tap(r, regions, pos)
    assert check_answer(r, regions, q, tap, pos) is not None


def test_far_from_pole_misses(earth, regions):
    r = _render(earth, 60, 30)
    q = Question("Найди Северный полюс", ("north_pole",))
    pos = tuple(r.project_latlon(70, 30)[0])
    assert check_answer(r, regions, q, inspect_tap(r, regions, pos), pos) is None


@pytest.mark.parametrize("lat,ok", [(6, True), (-7, True), (12, False)])
def test_roughly_at_equator(earth, regions, lat, ok):
    r = _render(earth, 0, 20)
    q = Question("Найди экватор", ("equator",))
    pos = tuple(r.project_latlon(lat, 20)[0])
    found = check_answer(r, regions, q, inspect_tap(r, regions, pos), pos)
    assert (found is not None) == ok


@pytest.mark.parametrize("dist,ok", [(3.0, True), (8.0, False)])
def test_peak_tolerance(earth, regions, dist, ok):
    # Вершина на модели не отдельный конус: засчитываем тык не дальше 5° от неё.
    kili = regions["kilimanjaro"]
    lat, lon = kili.at
    r = _render(earth, lat, lon, 2.5)
    q = Question("Найди Килиманджаро", ("kilimanjaro",))
    pos = tuple(r.project_latlon(lat - dist, lon)[0])
    tap = inspect_tap(r, regions, pos)
    assert tap is not None
    assert (check_answer(r, regions, q, tap, pos) is not None) == ok


def test_miss_on_elbrus_names_alps_and_points_east(earth, regions):
    r = _render(earth, 46.6, 9.9, 2.0)
    q = Question("Найди Эльбрус", ("elbrus",))
    pos = tuple(r.project_latlon(*regions["alps"].center)[0])
    tap = inspect_tap(r, regions, pos)
    assert check_answer(r, regions, q, tap, pos) is None
    text = miss_message(regions, q, tap, riddle=False)
    assert text.startswith("Это Альпы.") and "восточнее" in text

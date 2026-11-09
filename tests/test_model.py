import numpy as np
import pytest

from globe_quiz.model import (
    HEIGHT_BANDS,
    ICE_COLOR,
    SAND_COLOR,
    SURFACE_ICE,
    SURFACE_LAND,
    SURFACE_WATER,
    WATER_COLOR,
)

from .conftest import face_at


def test_model_size(earth):
    assert 10000 <= len(earth) <= 12000
    assert earth.sea_radius == pytest.approx(0.762, abs=0.01)


def test_highest_point_on_unit_radius(earth):
    r = np.linalg.norm(earth.tris.reshape(-1, 3), axis=1)
    assert r.max() == pytest.approx(1.0, abs=1e-5)
    assert r.min() == pytest.approx(earth.sea_radius, abs=1e-5)


def test_normals_point_outward(earth):
    # В исходном STL ~7 % граней — складки у обрывов, повёрнутые внутрь; упрощение
    # их наследует, а при отрисовке их скрывает отсечение задних граней.
    outward = (earth.normals * earth.centers).sum(axis=1) > 0
    assert outward.mean() > 0.9


def test_water_is_at_sea_level(earth):
    r = np.linalg.norm(earth.tris[earth.is_water], axis=2)
    assert np.allclose(r, earth.sea_radius, atol=1e-5)
    assert (earth.colors[earth.is_water] == WATER_COLOR).all()


@pytest.mark.parametrize("lat, lon, surface", [
    (-30, -70, SURFACE_LAND),  # Анды
    (0, -150, SURFACE_WATER),  # Тихий океан
    (25, 5, SURFACE_LAND),  # Сахара
    (-25, 125, SURFACE_LAND),  # Австралия (не зеркально)
    (-25, -125, SURFACE_WATER),
    (-80, 0, SURFACE_ICE),  # Антарктида
    (72, -40, SURFACE_ICE),  # Гренландия
    (87, 0, SURFACE_ICE),  # шапка у Северного полюса
])
def test_known_points(earth, lat, lon, surface):
    assert earth.surface[face_at(earth, lat, lon)] == surface


def test_colors_by_height_and_place(earth):
    assert tuple(earth.colors[face_at(earth, 25, 5)]) == SAND_COLOR  # Сахара
    assert tuple(earth.colors[face_at(earth, 23, 47)]) == SAND_COLOR  # Аравия
    assert tuple(earth.colors[face_at(earth, -80, 0)]) == ICE_COLOR
    assert tuple(earth.colors[face_at(earth, 55, 40)]) == HEIGHT_BANDS[0][1]  # Восточно-Европейская равнина
    snow = HEIGHT_BANDS[-1][1]
    tibet = (abs(earth.latlon[:, 0] - 34) < 6) & (abs(earth.latlon[:, 1] - 88) < 10)
    assert (earth.colors[tibet] == snow).all(axis=1).any()  # белые вершины Тибета
    andes = (abs(earth.latlon[:, 0] + 20) < 10) & (abs(earth.latlon[:, 1] + 68) < 5)
    assert (earth.height[andes] > 6).any()  # Анды — горы


# --- Знаменитые вершины, поднятые при переводе модели ---------------------------


def _vertex_heights(earth):
    """Радиусы всех вершин модели и их направления (единичные)."""
    v = earth.tris.reshape(-1, 3)
    r = np.linalg.norm(v, axis=1)
    return r, v / r[:, None]


@pytest.mark.parametrize("peak", ["everest", "aconcagua", "denali", "kilimanjaro", "elbrus", "mont_blanc"])
def test_peak_is_highest_around_and_snowy(earth, peak):
    from globe_quiz.geo import latlon_to_vec
    from tools.convert_stl import PEAKS

    _, lat, lon, _ = next(p for p in PEAKS if p[0] == peak)
    r, u = _vertex_heights(earth)
    d = np.degrees(np.arccos(np.clip(u @ latlon_to_vec(lat, lon), -1, 1)))
    around = np.nonzero(d < 3.0)[0]
    top = around[np.argmax(r[around])]
    assert d[top] < 0.05  # самая высокая точка в 3° — макушка ровно на месте вершины
    assert tuple(earth.colors[face_at(earth, lat, lon)]) == HEIGHT_BANDS[-1][1]  # верхушка белая


def test_peak_regions_match_converter():
    import json

    from globe_quiz import DATA_DIR
    from tools.convert_stl import PEAKS

    regions = {r["id"]: r for r in json.loads((DATA_DIR / "regions.json").read_text(encoding="utf-8"))["regions"]}
    for pid, lat, lon, _ in PEAKS:
        assert regions[pid]["at"] == [lat, lon]

import json

import numpy as np
import pygame
import pytest

from globe_quiz import ASSETS_DIR
from globe_quiz.geo import latlon_to_vec, vec_to_latlon
from globe_quiz.islands import (
    ISLAND_BASE,
    ISLAND_TOP,
    ISLANDS_PATH,
    LAND_COLOR,
    MAX_EDGE_DEG,
    SHORE_COLOR,
    build_island,
    is_simple,
    island_groups,
    load_island_outlines,
    project_outline,
    refine,
    signed_area,
    triangulate,
)
from globe_quiz.model import SURFACE_LAND, load_earth

from .conftest import face_at


def _area_of(xy, tris):
    return sum(abs(signed_area(xy[list(t)])) for t in tris)


@pytest.mark.parametrize("reverse", [False, True])
def test_square_triangulates(reverse):
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    if reverse:
        xy = xy[::-1].copy()
    tris = triangulate(xy)
    assert len(tris) == 2
    assert _area_of(xy, tris) == pytest.approx(1.0)
    assert all(signed_area(xy[list(t)]) > 0 for t in tris)  # против часовой


def test_horseshoe_triangulates():
    # Невыпуклая «подкова» (буква U): веер из центра здесь не годится.
    xy = np.array([[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]], float)
    tris = triangulate(xy)
    assert len(tris) == len(xy) - 2
    assert _area_of(xy, tris) == pytest.approx(abs(signed_area(xy)))
    for t in tris:  # ни один треугольник не лежит в вырезе подковы
        c = xy[list(t)].mean(axis=0)
        assert not (1 < c[0] < 2 and c[1] > 1)


def test_self_intersection_detected():
    bow = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
    assert not is_simple(bow)
    assert is_simple(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))


def test_refine_keeps_area_and_limits_edges():
    vecs = latlon_to_vec(np.array([0.0, 0.0, 10.0]), np.array([0.0, 10.0, 0.0]))
    verts, tris = refine(list(vecs), [(0, 1, 2)], 2.0)
    v = np.asarray(verts)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            assert np.degrees(np.arccos(np.clip(v[a] @ v[b], -1, 1))) <= 2.0 + 1e-9
    # Площадь на касательной плоскости сохраняется (дробление без дыр и наложений).
    xy = project_outline(v, vecs.mean(axis=0))
    whole = abs(signed_area(project_outline(vecs, vecs.mean(axis=0))))
    assert _area_of(xy, tris) == pytest.approx(whole, rel=0.02)


TEST_OUTLINE = np.array([[0, 0], [0, 8], [4, 8], [4, 5], [2, 5], [2, 3], [6, 3], [6, 0]], float)


def test_test_island_geometry():
    mesh = build_island(TEST_OUTLINE)
    top, walls = mesh.tris[mesh.is_top], mesh.tris[~mesh.is_top]
    assert len(top) and len(walls)
    r_top = np.linalg.norm(top.reshape(-1, 3), axis=1)
    assert np.allclose(r_top, ISLAND_TOP)
    r_wall = np.linalg.norm(walls.reshape(-1, 3), axis=1)
    assert np.isclose(r_wall.min(), ISLAND_BASE) and np.isclose(r_wall.max(), ISLAND_TOP)
    # Цвета — из палитры модели.
    assert (mesh.colors[mesh.is_top] == LAND_COLOR).all()
    assert (mesh.colors[~mesh.is_top] == SHORE_COLOR).all()
    # Нормали верха — от центра Земли.
    tc = top.mean(axis=1)
    assert ((mesh.normals[mesh.is_top] * tc).sum(axis=1) / np.linalg.norm(tc, axis=1) > 0.999).all()


def test_wall_normals_point_away_from_island():
    mesh = build_island(TEST_OUTLINE)
    walls = mesh.tris[~mesh.is_top]
    wn = mesh.normals[~mesh.is_top]
    wc = walls.mean(axis=1)
    # Стенка вертикальна: нормаль перпендикулярна радиусу.
    radial = (wn * wc).sum(axis=1) / np.linalg.norm(wc, axis=1)
    assert np.abs(radial).max() < 0.05
    # Шаг вдоль нормали выводит из острова, шаг против — внутрь.
    poly = TEST_OUTLINE

    def inside(lat, lon):
        x, y = lon, lat
        c = False
        for (y1, x1), (y2, x2) in zip(poly, np.roll(poly, -1, axis=0)):
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                c = not c
        return c

    for c, n in zip(wc, wn):
        u = c / np.linalg.norm(c)
        out = vec_to_latlon(u + 0.003 * n)
        inn = vec_to_latlon(u - 0.003 * n)
        assert not inside(*out) and inside(*inn)


def test_top_edges_are_small():
    mesh = build_island(TEST_OUTLINE)
    top = mesh.tris[mesh.is_top] / ISLAND_TOP
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ang = np.degrees(np.arccos(np.clip((top[:, a] * top[:, b]).sum(axis=1), -1, 1)))
        assert ang.max() <= MAX_EDGE_DEG + 1e-9


# --- Данные islands.json ---------------------------------------------------


def test_islands_file():
    islands = load_island_outlines()
    assert [i.id for i in islands] == ["hokkaido", "honshu", "shikoku", "kyushu", "madagascar", "iceland"]
    assert island_groups(islands) == ["Японские острова", "Мадагаскар", "Исландия"]
    for isl in islands:
        assert 10 <= len(isl.outline) <= 25, isl.id
        assert is_simple(isl.outline[:, ::-1]), f"{isl.id}: самопересечение"
        # Триангуляция полная: площадь треугольников равна площади контура.
        vecs = latlon_to_vec(isl.outline[:, 0], isl.outline[:, 1])
        xy = project_outline(vecs, vecs.mean(axis=0))
        tris = triangulate(xy)
        assert len(tris) == len(xy) - 2
        assert _area_of(xy, tris) == pytest.approx(abs(signed_area(xy)))


def test_missing_islands_file(tmp_path):
    assert load_island_outlines(tmp_path / "nope.json") == []


# --- Модель с дополнениями -------------------------------------------------


@pytest.fixture(scope="module")
def bare():
    return load_earth(with_islands=False)


def test_islands_appended_without_touching_original(earth, bare):
    assert len(bare) == 9187
    assert len(earth) > len(bare)
    n = len(bare)
    assert np.array_equal(earth.tris[:n], bare.tris)
    assert np.array_equal(earth.colors[:n], bare.colors)
    assert np.array_equal(earth.normals[:n], bare.normals)
    assert (earth.surface[n:] == SURFACE_LAND).all()
    assert len(earth.latlon) == len(earth)


def test_island_colors_from_palette(earth):
    palette = pygame.surfarray.array3d(pygame.image.load(str(ASSETS_DIR / "earth.png")))
    palette_colors = {tuple(c) for c in palette.reshape(-1, 3)}
    assert {tuple(c) for c in earth.colors[9187:].tolist()} <= palette_colors


@pytest.mark.parametrize("lat, lon", [
    (36.2, 138.3),   # Хонсю
    (43.4, 142.8),   # Хоккайдо
    (33.7, 133.5),   # Сикоку
    (32.6, 130.9),   # Кюсю
    (-19.0, 46.7),   # Мадагаскар
    (64.9, -18.6),   # Исландия
])
def test_islands_visible_from_above(earth, bare, lat, lon):
    f = face_at(earth, lat, lon)
    assert f >= len(bare) and earth.surface[f] == SURFACE_LAND


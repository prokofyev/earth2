import numpy as np
import pygame

from globe_quiz import ASSETS_DIR
from globe_quiz.model import SURFACE_LAND, SURFACE_WATER, load_earth

from .conftest import face_at


def test_model_has_all_faces():
    # Исходная модель без дополнений (острова — в test_islands.py).
    assert len(load_earth(with_islands=False)) == 9187


def test_each_face_has_one_palette_color(earth):
    palette = pygame.surfarray.array3d(pygame.image.load(str(ASSETS_DIR / "earth.png")))
    palette_colors = {tuple(c) for c in palette.reshape(-1, 3)}
    assert earth.colors.shape == (len(earth), 3)
    assert {tuple(c) for c in earth.colors.tolist()} <= palette_colors


def test_sphere_is_centered_and_unit(earth):
    r = np.linalg.norm(earth.tris.reshape(-1, 3), axis=1)
    assert abs(np.median(r) - 1.0) < 0.05
    # Если центр сферы подобран неверно, радиусы расходятся на ±0.14.
    assert r.min() > 0.95 and r.max() < 1.15


def test_known_points_land_and_water(earth):
    assert earth.surface[face_at(earth, -30, -70)] == SURFACE_LAND  # Анды
    assert earth.surface[face_at(earth, 0, -150)] == SURFACE_WATER  # Тихий океан
    assert earth.surface[face_at(earth, 15, 20)] == SURFACE_LAND  # Африка (Сахара)


def test_map_is_not_mirrored(earth):
    # Австралия на востоке (+125°), Анды на западе (−70°).
    assert earth.surface[face_at(earth, -25, 125)] == SURFACE_LAND
    assert earth.surface[face_at(earth, -25, -125)] == SURFACE_WATER


def test_normals_point_outward(earth):
    # У береговых стенок островов нормаль перпендикулярна радиусу — допускаем ноль.
    assert ((earth.normals * earth.centers).sum(axis=1) >= -1e-9).all()

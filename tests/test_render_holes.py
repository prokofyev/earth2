"""У глобуса не должно быть просветов фона: ни внутри круга, ни у его края."""

import numpy as np
import pygame
import pytest

from globe_quiz.renderer import GlobeRenderer, View

BG = (1, 2, 3)
VIEWS = [(lat, lon) for lat in (-60, -30, 0, 20, 50, 70) for lon in range(-180, 180, 10)]


def render(earth, lat, lon, size):
    r = GlobeRenderer(earth, View(lat, lon, 1.0))
    surf = pygame.Surface(size)
    surf.fill(BG)
    r.draw(surf, surf.get_rect(), outside=BG)
    bg = (pygame.surfarray.array3d(surf) == BG).all(axis=2).T
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    cx, cy = r.frame.center
    dist_px = np.hypot(xx - cx, yy - cy) - r.frame.scale  # < 0 — внутри круга
    return bg, dist_px


@pytest.mark.parametrize("lat,lon", VIEWS)
def test_globe_is_solid_disk(earth, lat, lon):
    bg, d = render(earth, lat, lon, (500, 500))
    assert int((bg & (d < -1.5)).sum()) == 0  # внутри круга нет дыр
    assert int((~bg & (d > 5)).sum()) == 0  # за кругом ничего не торчит


@pytest.mark.parametrize("lon", range(-180, 180, 15))
def test_splash_sized_globe_has_no_holes(earth, lon):
    bg, d = render(earth, 20, lon, (1792, 600))  # область глобуса на заставке
    assert int((bg & (d < -1.5)).sum()) == 0

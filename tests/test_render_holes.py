"""У глобуса не должно быть просветов фона: ни внутри круга океана, ни внутри гор за его краем."""

import numpy as np
import pygame
import pytest

from globe_quiz.renderer import GlobeRenderer, View

BG = (1, 2, 3)
VIEWS = [(lat, lon) for lat in (-60, -30, 0, 20, 50, 70) for lon in range(-180, 180, 10)]


def render(earth, lat, lon, size, zoom=1.0):
    r = GlobeRenderer(earth, View(lat, lon, zoom))
    surf = pygame.Surface(size)
    surf.fill(BG)
    r.draw(surf, surf.get_rect(), outside=BG)
    bg = (pygame.surfarray.array3d(surf) == BG).all(axis=2)  # (w, h)
    xx, yy = np.mgrid[0:size[0], 0:size[1]]
    cx, cy = r.frame.center
    dist = np.hypot(xx - cx, yy - cy)
    return bg, dist, r.frame.scale, earth.sea_radius


def enclosed_background(bg: np.ndarray) -> np.ndarray:
    """Пиксели фона, не связанные с открытым космосом у края картинки, — просветы."""
    w, h = bg.shape
    mask = pygame.mask.Mask((w, h))
    for x, y in zip(*np.nonzero(bg)):
        mask.set_at((int(x), int(y)))
    space = pygame.mask.Mask((w, h))
    for corner in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if mask.get_at(corner):
            space.draw(mask.connected_component(corner), (0, 0))
    open_space = np.zeros((w, h), bool)
    sx, sy = np.nonzero(bg)
    keep = np.array([space.get_at((int(x), int(y))) for x, y in zip(sx, sy)], bool)
    open_space[sx[keep], sy[keep]] = True
    # Зазубрины растеризации на самом силуэте (одиночные пиксели фона не дальше
    # 2 пикселей от открытого космоса) глазом не видны и просветами не считаются.
    # Настоящий просвет — фон глубже внутри горы или круга океана.
    near = open_space.copy()
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            near |= np.roll(np.roll(open_space, dx, axis=0), dy, axis=1)
    return bg & ~near


def check(bg, dist, scale, sea):
    assert int((bg & (dist < scale * sea - 1.5)).sum()) == 0  # внутри круга океана нет дыр
    assert int((~bg & (dist > scale + 5)).sum()) == 0  # за кругом радиуса 1 ничего нет
    assert int(enclosed_background(bg).sum()) == 0  # нет просветов внутри гор у края


@pytest.mark.parametrize("lat,lon", VIEWS)
def test_globe_has_no_holes(earth, lat, lon):
    check(*render(earth, lat, lon, (300, 300)))


@pytest.mark.parametrize("lon", range(-180, 180, 15))
def test_splash_sized_globe_has_no_holes(earth, lon):
    check(*render(earth, 20, lon, (1200, 500)))


@pytest.mark.parametrize("lat,lon", [(33, 0), (33, 180), (-20, 20), (-20, -160), (33, 88)])
def test_mountains_at_limb_have_no_holes(earth, lat, lon):
    # Тибет и Анды у края, а также крупно в центре.
    check(*render(earth, lat, lon, (700, 700)))
    check(*render(earth, lat, lon, (700, 700), zoom=2.5))

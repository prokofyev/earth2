"""Скорость отрисовки при сильном зуме: океан не должен тормозить, пока масштаб доезжает до цели."""

import math
import time

import pygame

from globe_quiz.renderer import GlobeRenderer, View

SIZE = (1792, 982)  # развёрнутое окно
OCEAN_MS = 30  # с большим запасом: ожидается несколько миллисекунд
FRAME_MS = 60


def test_zoom_in_to_max_stays_fast(earth):
    r = GlobeRenderer(earth, View(20, 30, 1.0))
    surf = pygame.Surface(SIZE)
    rect = surf.get_rect()
    r.draw(surf, rect)  # прогрев: шаблон океана, сетка радиусов
    ocean_ms = []
    name = "_blit_ocean" if hasattr(r, "_blit_ocean") else "_ocean_disk"
    real = getattr(r, name)

    def timed(*a, **kw):
        t = time.perf_counter()
        out = real(*a, **kw)
        ocean_ms.append(1000 * (time.perf_counter() - t))
        return out

    setattr(r, name, timed)
    frame_ms = []
    k = 1.0 - math.exp(-6.0 / 30)  # то же сглаживание, что в GlobeWidget.update при 30 кадрах/с
    for _ in range(30):
        r.view.zoom += (5.0 - r.view.zoom) * k
        r.view.center_lon += 0.5  # и сразу вращаем
        t = time.perf_counter()
        r.draw(surf, rect)
        frame_ms.append(1000 * (time.perf_counter() - t))
    assert r.view.zoom > 4.5
    assert max(ocean_ms) < OCEAN_MS, [round(x) for x in ocean_ms]
    assert max(frame_ms) < FRAME_MS, [round(x) for x in frame_ms]
    assert len(getattr(r, "_disk_cache", {})) <= 2  # круги под каждый радиус не копятся


def _old_ocean_color(dx, dy, key):
    """Цвет океана, как его считал прежний круг под каждый радиус."""
    from globe_quiz.model import WATER_COLOR
    from globe_quiz.renderer import _AMBIENT, _LIGHT

    x, y = dx / key, dy / key
    nz = math.sqrt(max(0.0, 1.0 - x * x - y * y))
    light = _AMBIENT + (1 - _AMBIENT) * max(0.0, x * _LIGHT[0] - y * _LIGHT[1] + nz * _LIGHT[2])
    return [min(255, int(c * light)) for c in WATER_COLOR]


def test_ocean_looks_as_before_at_zoom_one(earth):
    import random

    r = GlobeRenderer(earth, View(0, 0, 1.0))
    surf = pygame.Surface((800, 600))
    surf.fill((1, 2, 3))
    center, radius = (400, 300), 250.0
    r._blit_ocean(surf, center, radius)
    key = radius + 2
    rng = random.Random(3)
    for _ in range(20):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.95 * key)
        dx, dy = d * math.cos(a), d * math.sin(a)
        got = surf.get_at((round(center[0] + dx), round(center[1] + dy)))[:3]
        want = _old_ocean_color(round(dx), round(dy), key)
        assert all(abs(g - w) <= 3 for g, w in zip(got, want)), (dx, dy, got, want)
    # край точный: чуть внутри — вода, чуть снаружи — фон
    assert surf.get_at((round(center[0] + key - 2), center[1]))[:3] != (1, 2, 3)
    assert surf.get_at((round(center[0] + key + 2), center[1]))[:3] == (1, 2, 3)

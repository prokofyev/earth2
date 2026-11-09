import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402

from globe_quiz.geo import latlon_to_vec  # noqa: E402
from globe_quiz.model import load_earth  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _pygame():
    pygame.init()
    pygame.display.set_mode((800, 600))
    yield
    pygame.quit()


@pytest.fixture(scope="session")
def earth():
    return load_earth()


def face_at(model, lat: float, lon: float) -> int:
    """Грань, которая видна сверху в точке (lat, lon): самая внешняя вдоль луча из центра."""
    d = latlon_to_vec(lat, lon)
    t = model.tris
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    # Пересечение луча s·d (s > 0) с плоскостями треугольников (Мёллер — Трумбор).
    e1, e2 = b - a, c - a
    pvec = np.cross(d, e2)
    det = (e1 * pvec).sum(axis=1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = -a
    u = (tvec * pvec).sum(axis=1) * inv
    qvec = np.cross(tvec, e1)
    v = (qvec @ d) * inv
    s = (e2 * qvec).sum(axis=1) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (s > 0)
    assert hit.any(), f"нет грани в точке {lat}, {lon}"
    return int(np.nonzero(hit)[0][np.argmax(s[hit])])

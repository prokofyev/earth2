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
    """Грань, которая видна сверху в точке (lat, lon): самая внешняя вдоль луча."""
    d = latlon_to_vec(lat, lon)
    dirs = model.centers / np.linalg.norm(model.centers, axis=1)[:, None]
    near = np.nonzero(dirs @ d > np.cos(np.radians(6)))[0]
    best, best_r = None, -1.0
    for f in near:
        a, b, c = model.tris[f]
        m = np.column_stack([a - c, b - c, -d])
        try:
            u, v, t = np.linalg.solve(m, -c)
        except np.linalg.LinAlgError:
            continue
        if u >= 0 and v >= 0 and u + v <= 1 and t > 0 and t > best_r:
            best, best_r = int(f), t
    assert best is not None, f"нет грани в точке {lat}, {lon}"
    return best

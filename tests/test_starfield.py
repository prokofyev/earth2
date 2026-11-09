import numpy as np
import pygame
import pytest

from globe_quiz.geo import latlon_to_vec
from globe_quiz.renderer import View, view_matrix
from globe_quiz.ui import STAR_AREA_PX, Starfield

SIZE = (1200, 800)


def _star_shift(sky, lat0, lon0, lat1, lon1):
    """Сдвиг звезды, ближайшей к центру экрана, при смене вида; в долях радиуса неба."""
    xy0, idx0 = sky.star_positions(SIZE, lat0, lon0)
    c = np.array(SIZE) / 2
    k = int(np.argmin(np.linalg.norm(xy0 - c, axis=1)))
    assert np.linalg.norm(xy0[k] - c) < 0.15 * sky.radius
    xy1, idx1 = sky.star_positions(SIZE, lat1, lon1)
    j = np.nonzero(idx1 == idx0[k])[0]
    assert len(j) == 1
    return (xy1[j[0]] - xy0[k]) / sky.radius


def _globe_shift(lat0, lon0, lat1, lon1):
    """Сдвиг точки поверхности глобуса из центра экрана; в радиусах глобуса."""
    p = latlon_to_vec(lat0, lon0)
    a, b = p @ view_matrix(lat0, lon0).T, p @ view_matrix(lat1, lon1).T
    return np.array([b[0] - a[0], -(b[1] - a[1])])


@pytest.mark.parametrize("lat1, lon1", [(10, 23), (10, 17), (14, 20), (6, 20)])
def test_stars_move_with_globe(lat1, lon1):
    """Поворот по долготе и наклон по широте: навстречу материкам, на тот же угол."""
    sky = Starfield()
    star = _star_shift(sky, 10, 20, lat1, lon1)
    globe = _globe_shift(10, 20, lat1, lon1)
    assert np.linalg.norm(globe) > 0.03
    assert np.dot(star, globe) < 0  # в обратную сторону
    assert np.allclose(star, -globe, atol=0.1 * np.linalg.norm(globe))  # на тот же угол


def test_drag_right_moves_stars_left():
    # Перетаскивание вправо уменьшает center_lon (globe_widget._rotate_by).
    sky = Starfield()
    shift = _star_shift(sky, 0, 0, 0, -3)
    assert shift[0] < 0 and abs(shift[1]) < abs(shift[0]) * 0.2


def test_tilt_down_moves_stars_up():
    # Тянем глобус вниз — center_lat растёт, к зрителю поворачивается север.
    sky = Starfield()
    shift = _star_shift(sky, 0, 0, 3, 0)
    assert shift[1] < 0 and abs(shift[0]) < abs(shift[1]) * 0.2


def _render(sky, view):
    surf = pygame.Surface(SIZE)
    sky.view = view
    sky.draw(surf)
    return pygame.image.tobytes(surf, "RGB")


def test_zoom_does_not_move_stars():
    sky = Starfield()
    a = _render(sky, View(20, 15, 1.0))
    b = _render(sky, View(20, 15, 4.0))
    assert a == b


def test_return_to_same_view_restores_stars():
    sky = Starfield()
    a = _render(sky, View(20, 15))
    b = _render(sky, View(35, 80))
    c = _render(sky, View(20, 15))
    assert a != b and a == c


def test_still_globe_reuses_sky():
    sky = Starfield()
    view = View(20, 15)
    frames = [_render(sky, view) for _ in range(2)]  # второй кадр кладёт небо в кеш
    count = sky.render_count
    frames += [_render(sky, view) for _ in range(3)]
    assert sky.render_count == count  # глобус стоит — небо не перерисовывается
    assert len(set(frames)) == 1  # и картинка та же
    view.center_lon += 1
    moved = _render(sky, view)
    assert sky.render_count == count + 1 and moved != frames[0]


def test_star_density_even_and_as_before():
    sky = Starfield()
    w, h = SIZE
    box = 200
    center, corners, total = 0, 0, 0
    views = [(lat, lon) for lat in (-60, -20, 20, 60) for lon in range(0, 360, 30)]
    for lat, lon in views:
        xy, _ = sky.star_positions(SIZE, lat, lon)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = xy[inside]
        total += len(xy)
        cx, cy = w / 2, h / 2
        center += np.sum((abs(xy[:, 0] - cx) < box / 2) & (abs(xy[:, 1] - cy) < box / 2))
        for x0, y0 in ((0, 0), (w - box, 0), (0, h - box), (w - box, h - box)):
            corners += np.sum((xy[:, 0] >= x0) & (xy[:, 0] < x0 + box) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + box))
    corners /= 4
    assert abs(corners - center) / center < 0.25
    expected = w * h / STAR_AREA_PX
    assert abs(total / len(views) - expected) / expected < 0.15


def test_default_view_without_globe():
    sky = Starfield()
    surf = pygame.Surface(SIZE)
    sky.draw(surf)  # экран загрузки: глобуса ещё нет
    assert sky.render_count == 1


def test_stars_vanish_only_beyond_window():
    """Звезда, видимая в одном кадре, в следующем не пропадает посреди экрана."""
    sky = Starfield()
    w, h = SIZE
    for step in range(60):
        xy0, idx0 = sky.star_positions(SIZE, 15, step * 0.5)
        _, idx1 = sky.star_positions(SIZE, 15, step * 0.5 + 0.5)
        gone = ~np.isin(idx0, idx1)
        p = xy0[gone]
        inner = (p[:, 0] > 20) & (p[:, 0] < w - 20) & (p[:, 1] > 20) & (p[:, 1] < h - 20)
        assert not inner.any()

import numpy as np
import pygame
import pytest

from globe_quiz.globe_widget import GlobeWidget, is_tap
from globe_quiz.renderer import MAX_PITCH, Frame, GlobeRenderer, View


@pytest.fixture
def widget(earth):
    w = GlobeWidget(GlobeRenderer(earth, View(0.0, 0.0, 1.0)))
    w.set_rect(pygame.Rect(0, 0, 600, 600))
    w.draw(pygame.display.get_surface())
    return w


def _click(w, pos, to=None):
    to = to or pos
    w.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    if to != pos:
        w.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=to, rel=(0, 0), buttons=(1, 0, 0)))
    return w.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=to))


def test_is_tap_threshold():
    assert is_tap(0) and is_tap(8)
    assert not is_tap(8.5)


def test_tap_on_globe_returns_point(widget):
    assert _click(widget, (300, 300)) == (300, 300)


def test_drag_rotates_and_is_not_a_tap(widget):
    lon0 = widget.view.center_lon
    assert _click(widget, (300, 300), (360, 300)) is None
    assert widget.view.center_lon < lon0  # тянем вправо — в центр приходит запад


def test_tap_in_space_is_ignored(widget):
    assert _click(widget, (3, 3)) is None


def test_pitch_is_limited(widget):
    for _ in range(20):
        _click(widget, (300, 100), (300, 500))
    assert widget.view.center_lat == pytest.approx(MAX_PITCH)


def test_americas_left_africa_right(earth):
    r = GlobeRenderer(earth, View(0.0, -30.0, 1.0))
    surf = pygame.Surface((600, 600))
    r.draw(surf, surf.get_rect())
    (ax, _), _ = r.project_latlon(0, -60)  # Южная Америка
    (fx, _), _ = r.project_latlon(10, 15)  # Африка
    assert ax < 300 < fx


def test_screen_to_latlon_center(earth):
    r = GlobeRenderer(earth, View(30.0, 40.0, 1.0))
    surf = pygame.Surface((600, 600))
    r.draw(surf, surf.get_rect())
    lat, lon = r.screen_to_latlon((300, 300))
    assert lat == pytest.approx(30, abs=0.01) and lon == pytest.approx(40, abs=0.01)


def test_pick_topmost_of_overlapping_triangles(earth):
    r = GlobeRenderer(earth)
    pts = np.array([[[0, 0], [100, 0], [0, 100]], [[10, 10], [90, 10], [10, 90]]], float)
    r.frame = Frame(order=np.array([7, 42]), pts=pts, rot=np.eye(3), scale=1000.0, center=(0, 0))
    assert r.pick_face((20, 20)) == 42  # нарисована последней — сверху
    assert r.pick_face((3, 3)) == 7
    assert r.pick_face((200, 200)) is None


def test_idle_globe_is_not_rerendered(widget):
    surf = pygame.display.get_surface()
    widget.draw(surf)
    count = widget.render_count
    for _ in range(5):
        widget.update(0.016)
        widget.draw(surf)
    assert widget.render_count == count


def test_zoom_limits(widget):
    for _ in range(30):
        widget.zoom_by(2)
    for _ in range(200):
        widget.update(0.05)
    assert widget.view.zoom == pytest.approx(5.0)
    for _ in range(30):
        widget.zoom_by(0.5)
    for _ in range(200):
        widget.update(0.05)
    assert widget.view.zoom == pytest.approx(1.0)


def test_fly_to_centers_target(widget):
    widget.fly_to(-19, 46.5)  # Мадагаскар
    for _ in range(300):
        widget.update(0.05)
    assert not widget.busy
    assert widget.view.center_lat == pytest.approx(-19, abs=0.1)
    assert widget.view.center_lon == pytest.approx(46.5, abs=0.1)


def test_pick_mountain_beyond_ocean_edge(earth):
    # Тибет за правым краем: в 88° от центра вида он выглядывает за круг океана.
    r = GlobeRenderer(earth, View(33, 0, 1.0))
    surf = pygame.Surface((600, 600))
    surf.fill((1, 2, 3))
    r.draw(surf, surf.get_rect())
    cx, cy = r.frame.center
    sea_px = r.frame.scale * earth.sea_radius
    px = pygame.surfarray.array3d(surf)
    xx, yy = np.mgrid[0:600, 0:600]
    beyond = ((px != (1, 2, 3)).any(axis=2) & (np.hypot(xx - cx, yy - cy) > sea_px + 4) & (xx > cx))
    assert beyond.any(), "за краем океана справа нет гор"
    d = np.where(beyond, np.hypot(xx - cx, yy - cy), 0).ravel()
    found, face = None, None
    for k in np.argsort(-d)[:200]:  # самые выступающие точки, у края — внутри грани
        x, y = np.unravel_index(k, (600, 600))
        face = r.pick_face((int(x), int(y)))
        if face is not None:
            found = (int(x), int(y))
            break
    assert face is not None
    assert np.hypot(found[0] - cx, found[1] - cy) > sea_px + 4
    lat, lon = earth.latlon[face]
    assert 20 < lat < 45 and 70 < lon < 110 and not earth.is_water[face]
    tap_lat, tap_lon = r.screen_to_latlon(found)
    assert abs(tap_lat - lat) < 5 and abs(tap_lon - lon) < 5

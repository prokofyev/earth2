"""Замочек закрытого уровня — по форме образца assets/lock.jpg."""

import pytest

from globe_quiz.ui import LOCK_FILL, LOCK_HOLE, LOCK_LINE, lock_image

H = 300  # крупно, чтобы точки внутри деталей попадали уверенно


@pytest.fixture(scope="module")
def img():
    return lock_image(H)


def at(img, fx, fy):
    """Пиксель по долям ширины и высоты (в единицах образца 684 × 879)."""
    w, h = img.get_size()
    return img.get_at((int(fx / 684 * (w - 1)), int(fy / 879 * (h - 1))))


def close(c, ref, tol=12):
    return all(abs(a - b) <= tol for a, b in zip(c, ref))


def test_proportions_like_sample(img):
    w, h = img.get_size()
    assert h == H and w / h == pytest.approx(684 / 879, abs=0.03)


def test_background_and_shackle_opening_are_transparent(img):
    assert at(img, 5, 5).a == 0  # угол вне дужки
    assert at(img, 680, 5).a == 0
    assert at(img, 342, 230).a == 0  # просвет дужки
    assert at(img, 342, 50).a == 0  # между внешним и внутренним контуром дужки, как на образце


def test_lines_body_and_keyhole_colors(img):
    edge = at(img, 15, 600)  # левая стенка корпуса
    assert edge.a >= 250 and close(edge, LOCK_LINE)
    shackle = at(img, 113, 330)  # внешняя линия дужки
    assert shackle.a >= 250 and close(shackle, LOCK_LINE)
    body = at(img, 150, 450)  # корпус между стенкой и скважиной
    assert abs(body.a - LOCK_FILL[3]) <= 10 and close(body, LOCK_FILL[:3])
    hole = at(img, 342, 569)  # центр кружка скважины
    assert hole.a >= 250 and close(hole, LOCK_HOLE)
    slot = at(img, 342, 700)  # прорезь скважины
    assert close(slot, LOCK_HOLE)


def test_cached_per_size_and_color():
    assert lock_image(46) is lock_image(46)
    assert lock_image(46) is not lock_image(47)
    assert lock_image(46, (255, 0, 0)) is not lock_image(46)


def test_small_lock_keeps_its_lines():
    small = lock_image(25)
    w, h = small.get_size()
    opaque = sum(small.get_at((x, y)).a > 128 for x in range(w) for y in range(h))
    assert opaque > 0.15 * w * h  # тонкие линии не растворились при уменьшении

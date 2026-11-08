import numpy as np
import pytest

from globe_quiz.geo import angular_distance, direction_hint, latlon_to_vec, lon_delta, vec_to_latlon


def test_roundtrip_latlon():
    pts = np.array([[0, 0], [45, 90], [-30, -120], [80, 179]], float)
    back = vec_to_latlon(latlon_to_vec(pts[:, 0], pts[:, 1]))
    assert np.allclose(back, pts)


def test_lon_delta_crosses_dateline():
    assert lon_delta(170, -170) == pytest.approx(20)
    assert lon_delta(-170, 170) == pytest.approx(-20)


def test_angular_distance():
    assert angular_distance((0, 0), (0, 90)) == pytest.approx(90)
    assert angular_distance((90, 0), (-90, 0)) == pytest.approx(180)


@pytest.mark.parametrize(
    "src,dst,expected",
    [
        ((-15, -60), (5, 20), "севернее и восточнее"),  # Южная Америка → Африка
        ((0, 20), (0, -60), "западнее"),
        ((50, 10), (-30, 12), "южнее"),
        ((0, 170), (0, -170), "восточнее"),  # через линию перемены дат
        ((10, 10), (12, 13), None),
    ],
)
def test_direction_hint(src, dst, expected):
    assert direction_hint(src, dst) == expected

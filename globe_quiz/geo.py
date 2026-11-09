"""Перевод между 3D-векторами модели и широтой/долготой, подсказки направления.

Система координат модели (после центрирования в model.py):
  ось Y смотрит на север,
  вектор(lat, lon) = (cos lat * cos lon, sin lat, -cos lat * sin lon).
"""

from __future__ import annotations

import math

import numpy as np

# Направление засчитывается в подсказку, если разница по своей оси больше этого порога.
DIRECTION_THRESHOLD_DEG = 8.0


def latlon_to_vec(lat: float | np.ndarray, lon: float | np.ndarray) -> np.ndarray:
    """Единичный вектор (или массив векторов) по широте и долготе в градусах."""
    la = np.radians(lat)
    lo = np.radians(lon)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.sin(la), -np.cos(la) * np.sin(lo)], axis=-1
    )


def vec_to_latlon(v: np.ndarray) -> np.ndarray:
    """Широта и долгота в градусах для вектора (..., 3). Возвращает массив (..., 2)."""
    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(v[..., 1] / r, -1.0, 1.0)))
    lon = -np.degrees(np.arctan2(v[..., 2], v[..., 0]))
    return np.stack([lat, lon], axis=-1)


def wrap_lon(lon: float) -> float:
    """Привести долготу к диапазону [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def lon_delta(from_lon: float, to_lon: float) -> float:
    """Разница долгот по кратчайшей дуге: положительная — на восток."""
    return wrap_lon(to_lon - from_lon)


def angular_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Угловое расстояние между двумя точками (lat, lon) в градусах."""
    va = latlon_to_vec(*a)
    vb = latlon_to_vec(*b)
    return math.degrees(math.acos(float(np.clip(np.dot(va, vb), -1.0, 1.0))))


def direction_hint(src: tuple[float, float], dst: tuple[float, float]) -> str | None:
    """Слова-подсказка, куда двигаться от src к dst.

    Возвращает «севернее», «восточнее», «южнее и западнее» и т. п.
    или None, если цель совсем рядом.
    """
    dlat = dst[0] - src[0]
    dlon = lon_delta(src[1], dst[1])
    # Ближе к полюсам градус долготы короче — учитываем это для честного сравнения.
    dlon_scaled = dlon * math.cos(math.radians((src[0] + dst[0]) / 2))
    parts = []
    if abs(dlat) > DIRECTION_THRESHOLD_DEG:
        parts.append("севернее" if dlat > 0 else "южнее")
    if abs(dlon_scaled) > DIRECTION_THRESHOLD_DEG:
        parts.append("восточнее" if dlon > 0 else "западнее")
    if not parts:
        return None
    return " и ".join(parts)


def point_in_polygon(lat: np.ndarray, lon: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Чётно-нечётное правило на плоскости (долгота — x, широта — y)."""
    x, y = lon, lat
    inside = np.zeros(len(x), dtype=bool)
    py, px = poly[:, 0], poly[:, 1]
    n = len(poly)
    for i in range(n):
        j = (i - 1) % n
        yi, yj, xi, xj = py[i], py[j], px[i], px[j]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cond & (x < x_cross)
    return inside

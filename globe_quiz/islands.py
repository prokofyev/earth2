"""Острова, которых нет в исходной модели: строятся при загрузке из контуров.

Контур острова (широта, долгота) проецируется на касательную плоскость,
разбивается на треугольники (ear clipping), треугольники дробятся, пока
стороны не станут короче MAX_EDGE_DEG, а вершины поднимаются на радиус
ISLAND_TOP. По краю острова опускаются береговые стенки до ISLAND_BASE —
ниже водной оболочки, чтобы у берега не было щели.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import DATA_DIR
from .geo import latlon_to_vec

ISLANDS_PATH = DATA_DIR / "islands.json"

ISLAND_TOP = 1.033  # радиус верха острова: медиана соседней суши 1,033–1,037
ISLAND_BASE = 0.96  # низ береговой стенки — ниже воды (≈ 0,97)
MAX_EDGE_DEG = 2.5  # грани верха не крупнее, чем у соседней суши
LAND_COLOR = (166, 204, 52)  # зелёный из палитры модели
SHORE_COLOR = (244, 163, 128)  # песочный берег из палитры модели


@dataclass
class Island:
    id: str
    name: str
    group: str  # подпись для заставки: несколько островов могут быть одной группой
    outline: np.ndarray  # (N, 2) широта и долгота в градусах


@dataclass
class IslandMesh:
    tris: np.ndarray  # (F, 3, 3)
    normals: np.ndarray  # (F, 3)
    colors: np.ndarray  # (F, 3) uint8
    is_top: np.ndarray  # (F,) bool — верх (True) или стенка (False)


def load_island_outlines(path: Path = ISLANDS_PATH) -> list[Island]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Island(
            id=item["id"],
            name=item["name"],
            group=item.get("group", item["name"]),
            outline=np.asarray(item["outline"], dtype=float),
        )
        for item in data["islands"]
    ]


def island_groups(islands: list[Island]) -> list[str]:
    """Названия групп островов по порядку, без повторов — для пометки на заставке."""
    out: list[str] = []
    for isl in islands:
        if isl.group not in out:
            out.append(isl.group)
    return out


# --- Плоскость и треугольники ------------------------------------------------


def tangent_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Векторы «восток» и «север» в касательной плоскости в точке center."""
    up = center / np.linalg.norm(center)
    north_pole = np.array([0.0, 1.0, 0.0])
    east = np.cross(north_pole, up)
    if np.linalg.norm(east) < 1e-9:  # на полюсе восток не определён — берём любой
        east = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return east, north


def project_outline(vecs: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Гномоническая проекция точек сферы на касательную плоскость: (N, 2)."""
    up = center / np.linalg.norm(center)
    east, north = tangent_basis(up)
    p = vecs / (vecs @ up)[:, None]
    return np.stack([p @ east, p @ north], axis=1)


def signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cross2(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _inside(p, a, b, c) -> bool:
    return _cross2(a, b, p) >= 0 and _cross2(b, c, p) >= 0 and _cross2(c, a, p) >= 0


def triangulate(xy: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping для простого многоугольника. Треугольники против часовой стрелки."""
    n = len(xy)
    idx = list(range(n))
    if signed_area(xy) < 0:
        idx.reverse()
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3:
        m = len(idx)
        for k in range(m):
            i, j, l = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = xy[i], xy[j], xy[l]
            if _cross2(a, b, c) <= 1e-15:  # вогнутая или вырожденная вершина
                continue
            if any(_inside(xy[q], a, b, c) for q in idx if q not in (i, j, l)):
                continue
            tris.append((i, j, l))
            idx.pop(k)
            break
        else:
            guard += 1
            if guard > 1:
                raise ValueError("контур не триангулируется: самопересечение?")
            # Только вырожденные уши — выбрасываем вершину на прямой.
            idx.pop(0)
            continue
        guard = 0
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def segments_intersect(p1, p2, p3, p4) -> bool:
    d1, d2 = _cross2(p3, p4, p1), _cross2(p3, p4, p2)
    d3, d4 = _cross2(p1, p2, p3), _cross2(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def is_simple(xy: np.ndarray) -> bool:
    """Контур без самопересечений (несоседние стороны не пересекаются)."""
    n = len(xy)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(xy[i], xy[(i + 1) % n], xy[j], xy[(j + 1) % n]):
                return False
    return True


# --- Дробление на сфере ---------------------------------------------------------


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def refine(verts: list[np.ndarray], tris: list[tuple[int, int, int]], max_deg: float):
    """Дробит длинные стороны пополам (середина на сфере), пока все не короче max_deg.

    Сторона делится сразу во всех треугольниках, где она есть, поэтому сетка
    остаётся без Т-стыков и щелей.
    """
    verts = [v / np.linalg.norm(v) for v in verts]
    for _ in range(12):
        mids: dict[tuple[int, int], int] = {}
        for t in tris:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(a, b), max(a, b))
                if key not in mids and _angle(verts[a], verts[b]) > max_deg:
                    m = verts[a] + verts[b]
                    verts.append(m / np.linalg.norm(m))
                    mids[key] = len(verts) - 1
        if not mids:
            break
        out = []
        for a, b, c in tris:
            mab = mids.get((min(a, b), max(a, b)))
            mbc = mids.get((min(b, c), max(b, c)))
            mca = mids.get((min(c, a), max(c, a)))
            split = [m is not None for m in (mab, mbc, mca)]
            if all(split):
                out += [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
            elif not any(split):
                out.append((a, b, c))
            elif sum(split) == 1:
                if mab is not None:
                    out += [(a, mab, c), (mab, b, c)]
                elif mbc is not None:
                    out += [(a, b, mbc), (a, mbc, c)]
                else:
                    out += [(a, b, mca), (mca, b, c)]
            else:  # две стороны: угол между ними отрезаем, остаток — пополам
                if mab is None:  # поделены bc и ca, общий угол c
                    out += [(mca, mbc, c), (a, b, mbc), (a, mbc, mca)]
                elif mbc is None:  # поделены ab и ca, общий угол a
                    out += [(a, mab, mca), (mab, b, c), (mab, c, mca)]
                else:  # поделены ab и bc, общий угол b
                    out += [(mab, b, mbc), (a, mab, mbc), (a, mbc, c)]
        tris = out
    return verts, tris


# --- Остров целиком ----------------------------------------------------------


def build_island(outline_latlon: np.ndarray, top: float = ISLAND_TOP, base: float = ISLAND_BASE,
                 max_deg: float = MAX_EDGE_DEG) -> IslandMesh:
    vecs = latlon_to_vec(outline_latlon[:, 0], outline_latlon[:, 1])
    center = vecs.mean(axis=0)
    xy = project_outline(vecs, center)
    tri_idx = triangulate(xy)
    verts, tri_idx = refine(list(vecs), tri_idx, max_deg)
    v = np.asarray(verts)

    top_tris = v[np.asarray(tri_idx)] * top  # (T, 3, 3)
    top_normals = top_tris.mean(axis=1)
    top_normals /= np.linalg.norm(top_normals, axis=1)[:, None]

    # Края: стороны, которые принадлежат ровно одному треугольнику.
    # Обход треугольников против часовой стрелки (если смотреть сверху),
    # значит остров слева от стороны a→b, а наружу — справа.
    count: dict[tuple[int, int], int] = {}
    for t in tri_idx:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    walls, wall_normals = [], []
    for t in tri_idx:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if count[(min(a, b), max(a, b))] != 1:
                continue
            at, bt, ab, bb = v[a] * top, v[b] * top, v[a] * base, v[b] * base
            up = v[a] + v[b]
            n = np.cross(v[b] - v[a], up)  # вправо от направления a→b
            n /= np.linalg.norm(n)
            # Вершины против часовой стрелки, если смотреть снаружи, — как у верха.
            walls += [(at, ab, bb), (at, bb, bt)]
            wall_normals += [n, n]
    wall_tris = np.asarray(walls).reshape(-1, 3, 3)
    wall_normals = np.asarray(wall_normals).reshape(-1, 3)

    tris = np.concatenate([top_tris, wall_tris])
    normals = np.concatenate([top_normals, wall_normals])
    colors = np.array([LAND_COLOR] * len(top_tris) + [SHORE_COLOR] * len(wall_tris), dtype=np.uint8)
    is_top = np.r_[np.ones(len(top_tris), bool), np.zeros(len(wall_tris), bool)]
    return IslandMesh(tris=tris, normals=normals, colors=colors, is_top=is_top)


def build_islands(islands: list[Island], **kw) -> IslandMesh:
    meshes = [build_island(isl.outline, **kw) for isl in islands]
    if not meshes:
        empty = np.zeros((0, 3, 3))
        return IslandMesh(empty, np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, bool))
    return IslandMesh(
        tris=np.concatenate([m.tris for m in meshes]),
        normals=np.concatenate([m.normals for m in meshes]),
        colors=np.concatenate([m.colors for m in meshes]),
        is_top=np.concatenate([m.is_top for m in meshes]),
    )

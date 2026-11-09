"""Перевод рельефной модели Земли из STL в облегчённую сетку для игры.

Исходник — assets/World-Full-Final.STL («The Whole World», Davision3d, CC BY-NC 4.0):
231 тысяча треугольников, ось Z — север, океан — гладкая сфера уровня моря,
суша поднята над ней (высоты преувеличены), юг срезан плоско под подставку.

Шаги:
  1. сфера уровня моря, сварка вершин;
  2. срезанный подставкой юг поднимается на гладкий ледяной купол;
  3. упрощение схлопыванием рёбер с наименьшей квадратичной ошибкой (QEM):
     грани остаются там, где форма сложная, — вдоль берегов-обрывов и хребтов,
     а ровный океан и плоскогорья собираются в крупные грани;
  4. запись assets/earth_relief.npz в системе координат игры.

Запуск: uv run python -m tools.convert_stl
"""

from __future__ import annotations

import hashlib
import heapq
import math
import struct
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from globe_quiz import ASSETS_DIR  # noqa: E402

STL_PATH = ASSETS_DIR / "World-Full-Final.STL"
OUT_PATH = ASSETS_DIR / "earth_relief.npz"

LON_OFFSET = 105.0  # настоящая долгота = atan2(y, x) + LON_OFFSET (сверено по береговым линиям)
SEA_EPS = 0.15  # выше этого (в единицах STL) — суша
DOME_RISE = 2.4  # насколько купол у Южного полюса выше плоскости среза подставки
FACE_BUDGET = 12000  # больше — меньше 30 кадров в секунду в развёрнутом окне
MAX_FLIP_COS = 0.2  # схлопывание не должно поворачивать грань больше чем на ~78°


# --- 1. Чтение STL, уровень моря, сварка ---------------------------------------


def read_stl(path: Path) -> np.ndarray:
    """Треугольники двоичного STL: (F, 3, 3) float64."""
    data = Path(path).read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path}: не двоичный STL")
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dt, count=n, offset=84)["v"].astype(np.float64)


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.c_[2 * points, np.ones(len(points))]
    sol = np.linalg.lstsq(a, (points**2).sum(axis=1), rcond=None)[0]
    center = sol[:3]
    return center, float(np.sqrt(sol[3] + center @ center))


def fit_sea_sphere(tris: np.ndarray) -> tuple[np.ndarray, float]:
    """Сфера уровня моря: по крупным граням (океан), с отбрасыванием выбросов."""
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    p = tris[area >= np.percentile(area, 80)].reshape(-1, 3)
    center, radius = fit_sphere(p)
    for tol in (0.05, 0.02, 0.01, 0.005):
        r = np.linalg.norm(p - center, axis=1)
        center, radius = fit_sphere(p[np.abs(r - radius) < tol * radius])
    return center, radius


def weld(tris: np.ndarray, decimals: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Общие вершины: (V, 3) и грани (F, 3) индексами, без вырожденных граней."""
    flat = tris.reshape(-1, 3)
    _, first, inv = np.unique(np.round(flat, decimals), axis=0, return_index=True, return_inverse=True)
    verts = flat[first]
    faces = inv.reshape(-1, 3)
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return verts, faces[ok]


def stl_to_game(points: np.ndarray, center: np.ndarray, lon_offset: float = LON_OFFSET) -> np.ndarray:
    """Точки STL → система координат игры (ось Y — север, см. geo.py)."""
    p = points - center
    a = math.radians(lon_offset)
    x = p[..., 0] * math.cos(a) - p[..., 1] * math.sin(a)
    y = p[..., 0] * math.sin(a) + p[..., 1] * math.cos(a)
    return np.stack([x, p[..., 2], -y], axis=-1)


# --- 2. Купол на месте подставки ------------------------------------------------


def edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return {(int(a), int(b)): int(n) for (a, b), n in zip(u, c)}


def raise_south_cut(verts: np.ndarray, plane_z: float, rise: float = DOME_RISE,
                    tol: float = 0.05) -> np.ndarray:
    """Поднимает срезанный подставкой юг на гладкий ледяной купол.

    verts — относительно центра Земли (ось Z — север), plane_z — высота плоскости
    среза в тех же координатах. Вершины в плоскости сдвигаются только по радиусу:
    сетка остаётся той же и замкнутой. Подъём плавно растёт от края среза
    (у каждой долготы свой) к полюсу.
    """
    out = verts.copy()
    on = np.abs(verts[:, 2] - plane_z) < tol
    if not on.any():
        return out
    p = verts[on]
    r = np.linalg.norm(p, axis=1)
    alpha = np.degrees(np.arccos(np.clip(-p[:, 2] / r, -1, 1)))  # угол от Южного полюса
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    bins = ((lon + 180) // 10).astype(int) % 36
    edge = np.zeros(36)
    np.maximum.at(edge, bins, alpha)
    edge = np.maximum(edge, np.percentile(alpha, 50))  # пустые сектора у самого полюса
    edge = (np.roll(edge, 1) + 2 * edge + np.roll(edge, -1)) / 4
    t = np.clip(alpha / edge[bins], 0, 1)
    bump = rise * (1 - t * t) ** 2
    out[on] = p / r[:, None] * (r + bump)[:, None]
    return out


# --- 3. Упрощение (quadric error metrics) ---------------------------------------


def _plane_quadrics(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Квадрики плоскостей граней (F, 10), взвешенные площадью."""
    t = verts[faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area2 = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area2, 1e-12)[:, None]
    a, b, c = n[:, 0], n[:, 1], n[:, 2]
    d = -(n * t[:, 0]).sum(axis=1)
    w = area2 / 2
    return np.stack([a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d], axis=1) * w[:, None]


def _cost(q, p) -> float:
    x, y, z = p
    return (q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x
            + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9])


def _normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def decimate(verts: np.ndarray, faces: np.ndarray, target_faces: int, verbose: bool = False):
    """Схлопывает рёбра с наименьшей ошибкой, пока граней больше target_faces.

    Не схлопывает: вершины на краях и неманифолдных рёбрах; рёбра, после которых
    сетка перестала бы быть многообразием (условие связи); схлопывания, которые
    сильно поворачивают или вырождают соседние грани.
    """
    pos = [tuple(p) for p in verts.tolist()]
    flist = [list(f) for f in faces.tolist()]
    alive = [True] * len(flist)
    vfaces: list[set[int]] = [set() for _ in pos]
    for fi, f in enumerate(flist):
        for v in f:
            vfaces[v].add(fi)
    fq = _plane_quadrics(verts, faces)
    q = np.zeros((len(pos), 10))
    for k in range(3):
        np.add.at(q, faces[:, k], fq)
    quad = [list(row) for row in q.tolist()]
    counts = edge_counts(faces)
    locked = [False] * len(pos)
    for (a, b), c in counts.items():
        if c != 2:
            locked[a] = locked[b] = True
    version = [0] * len(pos)
    removed = [False] * len(pos)

    def best(u, v):
        qs = [x + y for x, y in zip(quad[u], quad[v])]
        pu, pv = pos[u], pos[v]
        pm = ((pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2, (pu[2] + pv[2]) / 2)
        cands = [(_cost(qs, p), p) for p in (pu, pv, pm)]
        return min(cands, key=lambda cp: cp[0])

    heap = []
    for (a, b) in counts:
        if locked[a] or locked[b]:
            continue
        c, _ = best(a, b)
        heap.append((c, a, b, 0, 0))
    heapq.heapify(heap)

    def neighbors(v):
        out = set()
        for fi in vfaces[v]:
            out.update(flist[fi])
        out.discard(v)
        return out

    n_faces = len(flist)
    while n_faces > target_faces and heap:
        c, u, v, su, sv = heapq.heappop(heap)
        if removed[u] or removed[v] or version[u] != su or version[v] != sv:
            continue
        shared = vfaces[u] & vfaces[v]
        if len(shared) != 2:
            continue
        # Условие связи: общие соседи — только вершины двух общих граней.
        opp = set()
        for fi in shared:
            opp.update(flist[fi])
        opp -= {u, v}
        if (neighbors(u) & neighbors(v)) != opp:
            continue
        _, p = best(u, v)
        ok = True
        for w in (u, v):
            for fi in vfaces[w]:
                if fi in shared:
                    continue
                f = flist[fi]
                old = [pos[x] for x in f]
                new = [p if x in (u, v) else pos[x] for x in f]
                no = _normal(*old)
                nn = _normal(*new)
                lo = math.sqrt(no[0] ** 2 + no[1] ** 2 + no[2] ** 2)
                ln = math.sqrt(nn[0] ** 2 + nn[1] ** 2 + nn[2] ** 2)
                if ln < 1e-9 or lo < 1e-12:
                    ok = False
                    break
                if (no[0] * nn[0] + no[1] * nn[1] + no[2] * nn[2]) / (lo * ln) < MAX_FLIP_COS:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # Схлопываем v в u.
        for fi in shared:
            alive[fi] = False
            for x in flist[fi]:
                vfaces[x].discard(fi)
        n_faces -= 2
        for fi in vfaces[v]:
            f = flist[fi]
            f[f.index(v)] = u
            vfaces[u].add(fi)
        vfaces[v] = set()
        removed[v] = True
        pos[u] = p
        quad[u] = [x + y for x, y in zip(quad[u], quad[v])]
        version[u] += 1
        for w in neighbors(u):
            if locked[w] or removed[w]:
                continue
            cw, _ = best(u, w)
            a, b = (u, w) if u < w else (w, u)
            heapq.heappush(heap, (cw, a, b, version[a], version[b]))
        if verbose and n_faces % 20000 < 2:
            print(f"  граней {n_faces}")

    keep = [i for i in range(len(pos)) if not removed[i] and vfaces[i]]
    remap = {old: new for new, old in enumerate(keep)}
    out_v = np.array([pos[i] for i in keep])
    out_f = np.array([[remap[x] for x in f] for fi, f in enumerate(flist) if alive[fi]], dtype=np.int32)
    return out_v, out_f


# --- 4. Сборка и запись --------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(stl_path: Path = STL_PATH, out_path: Path = OUT_PATH, lon_offset: float = LON_OFFSET,
            budget: int = FACE_BUDGET, verbose: bool = True) -> dict:
    tris = read_stl(stl_path)
    center, sea = fit_sea_sphere(tris)
    verts, faces = weld(tris)
    verts = verts - center
    verts = raise_south_cut(verts, float(tris[:, :, 2].min() - center[2]))
    if verbose:
        print(f"сетка STL: {len(verts)} вершин, {len(faces)} граней; упрощаем до {budget}…")
    verts, faces = decimate(verts, faces, budget, verbose=verbose)
    game = stl_to_game(verts, np.zeros(3), lon_offset)
    r = np.linalg.norm(game, axis=1)
    h = r - sea
    h[h < SEA_EPS] = 0.0  # уровень моря — ровно на сфере
    game = game / r[:, None] * (sea + h)[:, None]
    h_max = float(h.max())
    data = dict(
        vertices=(game / (sea + h_max)).astype(np.float32),
        faces=faces.astype(np.int32),
        heights=h.astype(np.float32),
        sea_radius=np.float64(sea / (sea + h_max)),
        stl_sea_radius=np.float64(sea),
        lon_offset=np.float64(lon_offset),
        source_sha256=np.array(sha256(stl_path)),
    )
    np.savez_compressed(out_path, **data)
    if verbose:
        print(f"{out_path.name}: {len(game)} вершин, {len(faces)} граней, "
              f"уровень моря {data['sea_radius']:.3f}, наибольшая высота {h_max:.2f}")
    return data


if __name__ == "__main__":
    convert()

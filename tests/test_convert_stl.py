"""Перевод STL в облегчённую модель (tools/convert_stl.py)."""

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from globe_quiz.geo import vec_to_latlon
from tools import convert_stl as cs

from .conftest import face_at


def _icosphere(level: int):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        mids, out = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in mids:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                mids[key] = len(verts) - 1
            return mids[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        f = out
    return np.array(verts), np.array(f)


def _bumpy_ball(level=4):
    """Сфера радиуса 10 с «горой» высотой 2 вокруг оси +X (центр в нуле)."""
    v, f = _icosphere(level)
    cos25 = np.cos(np.radians(25))
    bump = np.clip((v @ np.array([1.0, 0, 0]) - cos25) / (1 - cos25), 0, 1)
    return v * (10 + 2 * bump)[:, None], f


def _write_stl(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"test".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            fh.write(struct.pack("<3f", 0, 0, 0))
            fh.write(struct.pack("<9f", *np.asarray(t, float).ravel()))
            fh.write(struct.pack("<H", 0))


def _edge_hist(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return c


def test_synthetic_sphere_recovered(tmp_path):
    v, f = _bumpy_ball()
    path = tmp_path / "ball.stl"
    _write_stl(path, v[f] + np.array([1.0, 2.0, 3.0]))
    tris = cs.read_stl(path)
    center, radius = cs.fit_sea_sphere(tris)
    assert np.allclose(center, [1, 2, 3], atol=0.05)
    assert radius == pytest.approx(10, rel=0.005)
    h = np.linalg.norm(tris.reshape(-1, 3) - center, axis=1) - radius
    assert h.max() == pytest.approx(2, abs=0.1)
    verts, faces = cs.weld(tris)
    assert len(verts) == len(v) and (_edge_hist(faces) == 2).all()


def test_not_binary_stl(tmp_path):
    p = tmp_path / "bad.stl"
    p.write_bytes(b"solid ascii\n" * 10)
    with pytest.raises(ValueError):
        cs.read_stl(p)


def test_stl_axes_to_game():
    # Ось Z STL — север (ось Y игры); долгота atan2(y, x) + LON_OFFSET.
    north = cs.stl_to_game(np.array([0.0, 0, 5]), np.zeros(3), 105.0)
    assert np.allclose(north, [0, 5, 0])
    p = cs.stl_to_game(np.array([1.0, 0, 0]), np.zeros(3), 105.0)
    assert np.allclose(vec_to_latlon(p), [0, 105.0])


def test_decimate_keeps_closed_oriented_mesh_and_detail():
    v, f = _bumpy_ball(4)  # 5120 граней
    dv, df = cs.decimate(v, f, 1000)
    assert 990 <= len(df) <= 1000
    c = _edge_hist(df)
    assert (c == 2).all()  # замкнута, без Т-стыков и щелей
    assert len(dv) - len(c) + len(df) == 2
    t = dv[df]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    assert ((n * t.mean(axis=1)).sum(axis=1) > 0).all()  # обход сохранился
    # На «горе» вершины гуще, чем на ровной сфере.
    on_bump = (dv @ np.array([1.0, 0, 0])) / np.linalg.norm(dv, axis=1) > np.cos(np.radians(25))
    cap_share = (1 - np.cos(np.radians(25))) / 2  # доля площади сферы под горой
    assert on_bump.mean() > 1.5 * cap_share


def test_south_cut_raised_to_dome():
    # Плоскость среза z = −9 у шара радиуса 10: вершины в ней поднимаются по радиусу.
    v, f = _icosphere(4)
    v = v * 10
    v[:, 2] = np.maximum(v[:, 2], -9.0)
    raised = cs.raise_south_cut(v, -9.0, rise=1.0)
    moved = np.abs(v[:, 2] + 9.0) < 0.05
    assert moved.any()
    r0, r1 = np.linalg.norm(v, axis=1), np.linalg.norm(raised, axis=1)
    assert (r1[moved] >= r0[moved]).all() and np.allclose(r1[~moved], r0[~moved])
    assert np.allclose(raised / r1[:, None], v / r0[:, None])  # только по радиусу
    pole = np.argmin(v[:, 2])
    assert r1[pole] == pytest.approx(r0[pole] + 1.0, abs=0.05)  # у полюса подъём наибольший


# --- Настоящий файл -------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    tris = cs.read_stl(cs.STL_PATH)
    center, sea = cs.fit_sea_sphere(tris)
    return SimpleNamespace(tris=tris, center=center, sea=sea)


def test_real_sea_level_and_heights(real):
    assert real.sea == pytest.approx(51.96, abs=0.05)
    h = np.linalg.norm(real.tris.reshape(-1, 3) - real.center, axis=1) - real.sea
    assert h.max() == pytest.approx(16.9, abs=0.2)
    assert h.min() > -0.2  # морского дна нет: ниже уровня моря только провисание хорд


def test_real_mesh_almost_closed(real):
    _, faces = cs.weld(real.tris)
    c = _edge_hist(faces)
    assert (c == 1).sum() <= 3 and (c > 2).sum() <= 2  # крошечный шов и пара неманифолдных рёбер исходника


# --- Готовая модель ---------------------------------------------------------------


@pytest.fixture(scope="module")
def npz():
    return np.load(cs.OUT_PATH)


def test_npz_budget_radii_and_topology(npz):
    v, f = npz["vertices"].astype(float), npz["faces"]
    assert len(f) <= cs.FACE_BUDGET
    r = np.linalg.norm(v, axis=1)
    assert r.max() == pytest.approx(1.0, abs=1e-5)
    assert r.min() == pytest.approx(float(npz["sea_radius"]), abs=1e-5)
    c = _edge_hist(f)
    assert (c != 2).sum() <= 5  # швы исходника не размножились


def test_npz_south_pole_is_a_dome(npz):
    v, h, f = npz["vertices"].astype(float), npz["heights"], npz["faces"]
    plane_h = 58.95 - float(npz["stl_sea_radius"])  # высота плоскости среза над морем у полюса
    pole = face_at(_mesh(npz), -89.9, 0)
    assert h[f[pole]].mean() > plane_h + 1.0  # над полюсом купол выше плоского среза
    lat = vec_to_latlon(v)[:, 0]
    assert h[lat < -75].std() > 0.3  # и не плоский


def test_npz_matches_source_stl(npz):
    assert str(npz["source_sha256"]) == cs.sha256(cs.STL_PATH)
    assert float(npz["lon_offset"]) == cs.LON_OFFSET


def _mesh(npz):
    t = npz["vertices"].astype(float)[npz["faces"]]
    return SimpleNamespace(tris=t, centers=t.mean(axis=1))


@pytest.mark.parametrize("lat, lon, land", [
    (30.0, 31.2, True),  # Каир
    (55.7, 37.6, True),  # Москва
    (-33.9, 151.2, True),  # Сидней
    (4.6, -74.1, True),  # Богота
    (-25.0, -68.0, True),  # Анды
    (33.0, 88.0, True),  # Тибет
    (42.5, 13.0, True),  # Италия
    (0.0, -150.0, False),  # Тихий океан
    (25.0, -40.0, False),  # Атлантический
    (-20.0, 80.0, False),  # Индийский
    (43.0, 34.0, False),  # Чёрное море
])
def test_control_points(npz, lat, lon, land):
    f = face_at(_mesh(npz), lat, lon)
    is_land = (npz["heights"][npz["faces"][f]] > 0).any()
    assert bool(is_land) == land


def test_conversion_is_reproducible(npz, tmp_path):
    again = cs.convert(out_path=tmp_path / "again.npz", verbose=False)
    for key in ("vertices", "faces", "heights"):
        assert np.array_equal(again[key], npz[key])

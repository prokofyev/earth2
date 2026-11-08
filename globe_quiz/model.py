"""Загрузка модели Земли из glTF в массивы numpy.

Модель — низкополигональная Земля с палитрой: у каждой грани ровно один цвет.
Океан — отдельная оболочка под сушей; её скрытые части не удаляются,
а закрываются сушей при отрисовке (вода рисуется первым слоем).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from . import ASSETS_DIR
from .geo import vec_to_latlon
from .islands import build_islands, load_island_outlines

WATER_COLOR = (85, 177, 241)
ICE_COLOR = (245, 247, 250)

SURFACE_WATER = 0
SURFACE_LAND = 1
SURFACE_ICE = 2
SURFACE_NAMES = {"water": SURFACE_WATER, "land": SURFACE_LAND, "ice": SURFACE_ICE}

_COMPONENT = {5125: np.uint32, 5123: np.uint16, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass
class EarthModel:
    tris: np.ndarray  # (F, 3, 3) вершины граней, сфера радиуса ~1 с центром в нуле
    normals: np.ndarray  # (F, 3) внешние нормали граней
    centers: np.ndarray  # (F, 3) центры граней
    colors: np.ndarray  # (F, 3) uint8 цвет грани из палитры
    surface: np.ndarray  # (F,) int8: SURFACE_WATER / SURFACE_LAND / SURFACE_ICE
    latlon: np.ndarray  # (F, 2) широта и долгота центра грани, градусы

    @property
    def is_water(self) -> np.ndarray:
        return self.surface == SURFACE_WATER

    def __len__(self) -> int:
        return len(self.tris)


def _read_accessor(gltf: dict, buf: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMPONENT[acc["componentType"]])
    width = _WIDTH[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", width * dtype.itemsize) // dtype.itemsize
    count = acc["count"]
    raw = np.frombuffer(buf, dtype=dtype, offset=offset, count=(count - 1) * stride + width)
    return np.lib.stride_tricks.as_strided(
        raw, shape=(count, width), strides=(stride * dtype.itemsize, dtype.itemsize)
    ).copy()


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Сфера, лучше всего описывающая точки (метод наименьших квадратов)."""
    a = np.c_[2 * points, np.ones(len(points))]
    b = (points**2).sum(axis=1)
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    center = sol[:3]
    radius = float(np.sqrt(sol[3] + center @ center))
    return center, radius


def load_earth(assets_dir: Path = ASSETS_DIR, with_islands: bool = True) -> EarthModel:
    """Модель из glTF. with_islands — дописать острова из islands.json (см. islands.py)."""
    gltf = json.loads((assets_dir / "earth.gltf").read_text())
    buf = (assets_dir / gltf["buffers"][0]["uri"]).read_bytes()
    prim = gltf["meshes"][0]["primitives"][0]
    pos = _read_accessor(gltf, buf, prim["attributes"]["POSITION"]).astype(np.float64)
    uv = _read_accessor(gltf, buf, prim["attributes"]["TEXCOORD_0"])
    idx = _read_accessor(gltf, buf, prim["indices"]).reshape(-1, 3)

    image_path = assets_dir / gltf["images"][0]["uri"]
    palette = pygame.surfarray.array3d(pygame.image.load(str(image_path)))  # (W, H, 3)
    w, h = palette.shape[:2]

    center, radius = fit_sphere(pos)
    pos = (pos - center) / radius
    tris = pos[idx]

    face_uv = uv[idx].mean(axis=1)
    px = np.clip((face_uv[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip((face_uv[:, 1] * h).astype(int), 0, h - 1)
    colors = palette[px, py].astype(np.uint8)

    surface = np.full(len(tris), SURFACE_LAND, dtype=np.int8)
    surface[(colors == WATER_COLOR).all(axis=1)] = SURFACE_WATER
    surface[(colors == ICE_COLOR).all(axis=1)] = SURFACE_ICE

    centers = tris.mean(axis=1)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12  # выбрасываем вырожденные грани
    normals = normals / np.where(lengths > 0, lengths, 1.0)[:, None]
    # Материал двусторонний — разворачиваем нормали наружу.
    flip = (normals * centers).sum(axis=1) < 0
    normals[flip] *= -1

    tris, normals, centers = tris[keep], normals[keep], centers[keep]
    colors, surface = colors[keep], surface[keep]

    if with_islands:
        extra = build_islands(load_island_outlines())
        tris = np.concatenate([tris, extra.tris])
        normals = np.concatenate([normals, extra.normals])  # нормали островов — явные
        centers = np.concatenate([centers, extra.tris.mean(axis=1)])
        colors = np.concatenate([colors, extra.colors])
        surface = np.concatenate([surface, np.full(len(extra.tris), SURFACE_LAND, np.int8)])
    return EarthModel(
        tris=tris,
        normals=normals,
        centers=centers,
        colors=colors,
        surface=surface,
        latlon=vec_to_latlon(centers),
    )

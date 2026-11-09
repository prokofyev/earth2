"""Рельефная модель Земли: загрузка готовой сетки и раскраска.

Сетка заранее получена из assets/World-Full-Final.STL инструментом
tools/convert_stl.py и лежит в assets/earth_relief.npz. Это одна замкнутая
поверхность: океан — грани на уровне моря, суша поднята над ним (высоты
преувеличены, как в исходной модели). Цветов в исходнике нет — суша
раскрашивается по высоте, пустыни песочные, полярные области ледяные.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ASSETS_DIR, DATA_DIR
from .geo import point_in_polygon, vec_to_latlon

MODEL_PATH = ASSETS_DIR / "earth_relief.npz"
COLORING_PATH = DATA_DIR / "coloring.json"

WATER_COLOR = (85, 177, 241)
ICE_COLOR = (245, 247, 250)
SAND_COLOR = (236, 204, 140)
# Полосы высоты суши: высота над уровнем моря в единицах исходной модели (радиус Земли ≈ 52).
# В модели вся суша приподнята: равнины лежат на высоте ≈ 2–2,6, Альпы — до 3,5,
# Кавказ — до 6, Анды — до 9, Тибет — до 16.
HEIGHT_BANDS = [
    (2.7, (166, 204, 52)),  # низменности и равнины
    (3.3, (196, 206, 90)),  # возвышенности
    (5.0, (206, 172, 102)),  # невысокие горы и плоскогорья
    (9.0, (170, 125, 80)),  # горы
    (12.5, (196, 176, 156)),  # высокогорья
    (float("inf"), (240, 240, 240)),  # снежные вершины
]
DESERT_MAX_HEIGHT = 6.0  # выше пустыня уже горы
ANTARCTIC_LAT = -60.0  # южнее суша — лёд Антарктиды
ARCTIC_CAP_LAT = 78.0  # севернее суша — полярная ледяная шапка

SURFACE_WATER = 0
SURFACE_LAND = 1
SURFACE_ICE = 2
SURFACE_NAMES = {"water": SURFACE_WATER, "land": SURFACE_LAND, "ice": SURFACE_ICE}


@dataclass
class EarthModel:
    tris: np.ndarray  # (F, 3, 3) вершины граней; самая высокая точка на радиусе 1
    normals: np.ndarray  # (F, 3) внешние нормали граней
    centers: np.ndarray  # (F, 3) центры граней
    colors: np.ndarray  # (F, 3) uint8
    surface: np.ndarray  # (F,) int8: SURFACE_WATER / SURFACE_LAND / SURFACE_ICE
    latlon: np.ndarray  # (F, 2) широта и долгота центра грани, градусы
    height: np.ndarray  # (F,) средняя высота грани над уровнем моря (единицы исходной модели)
    sea_radius: float = 1.0  # радиус уровня моря

    @property
    def is_water(self) -> np.ndarray:
        return self.surface == SURFACE_WATER

    def __len__(self) -> int:
        return len(self.tris)


def height_color(h: np.ndarray) -> np.ndarray:
    out = np.empty((len(h), 3), np.uint8)
    lower = -np.inf
    for top, color in HEIGHT_BANDS:
        out[(h >= lower) & (h < top)] = color
        lower = top
    return out


def _in_any(latlon: np.ndarray, polygons) -> np.ndarray:
    inside = np.zeros(len(latlon), bool)
    for poly in polygons:
        poly = np.asarray(poly, float)
        for shift in (-360.0, 0.0, 360.0):
            inside |= point_in_polygon(latlon[:, 0], latlon[:, 1] + shift, poly)
    return inside


def classify(latlon: np.ndarray, vertex_heights: np.ndarray, coloring: dict) -> tuple[np.ndarray, np.ndarray]:
    """Тип поверхности и цвет граней по высотам их вершин (F, 3) и широте/долготе центров."""
    h = vertex_heights.mean(axis=1)
    water = (vertex_heights <= 0).all(axis=1)
    lat = latlon[:, 0]
    ice = ~water & ((lat < ANTARCTIC_LAT) | (lat > ARCTIC_CAP_LAT) | _in_any(latlon, [coloring["greenland"]]))
    # Песок — ниже порога высоты: общего DESERT_MAX_HEIGHT или своего у контура
    # (Гоби и Калахари на модели высокие — иначе они вышли бы горами).
    desert = np.zeros(len(h), bool)
    for d in coloring["deserts"]:
        desert |= (h < d.get("max_height", DESERT_MAX_HEIGHT)) & _in_any(latlon, [d["polygon"]])
    desert &= ~water & ~ice

    surface = np.full(len(h), SURFACE_LAND, np.int8)
    surface[water] = SURFACE_WATER
    surface[ice] = SURFACE_ICE
    colors = height_color(h)
    colors[desert] = SAND_COLOR
    colors[ice] = ICE_COLOR
    colors[water] = WATER_COLOR
    return surface, colors


def load_earth(path: Path = MODEL_PATH, coloring_path: Path = COLORING_PATH) -> EarthModel:
    data = np.load(path)
    verts = data["vertices"].astype(np.float64)
    faces = data["faces"]
    vh = data["heights"].astype(np.float64)[faces]  # (F, 3)
    tris = verts[faces]
    centers = tris.mean(axis=1)
    # Грани обходятся против часовой стрелки снаружи — нормаль по обходу, без разворотов.
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    latlon = vec_to_latlon(centers)
    coloring = json.loads(Path(coloring_path).read_text(encoding="utf-8"))
    surface, colors = classify(latlon, vh, coloring)
    # Снежные шапки знаменитых вершин: грани, сходящиеся в макушке пика, — белые.
    # По средней высоте они могли бы выйти коричневыми: соседние вершины ниже.
    if "peak_vertices" in data:
        cap = np.isin(faces, data["peak_vertices"]).any(axis=1) & (surface == SURFACE_LAND)
        colors[cap] = HEIGHT_BANDS[-1][1]
    return EarthModel(
        tris=tris,
        normals=normals,
        centers=centers,
        colors=colors,
        surface=surface,
        latlon=latlon,
        height=vh.mean(axis=1),
        sea_radius=float(data["sea_radius"]),
    )

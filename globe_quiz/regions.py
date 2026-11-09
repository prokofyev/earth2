"""Географические объекты (регионы) и разметка граней модели.

Регион задаётся в regions.json набором частей. Каждая часть — многоугольник
в градусах [широта, долгота] или прямоугольник по широте/долготе, плюс типы
поверхности (вода, суша, лёд), которые она размечает.

Интересные факты об объектах лежат отдельно, в facts.json: по пять у каждого.

Базовые регионы (материки, озёра и океаны) делят грани между собой: грань получает
первый подходящий базовый регион в порядке файла. Остальные регионы
(острова, моря, полушария…) накладываются поверх независимо.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import DATA_DIR
from .geo import latlon_to_vec, point_in_polygon, vec_to_latlon
from .model import SURFACE_NAMES, EarthModel

BASE_KINDS = {"continent", "ocean", "lake"}
FACE_KINDS = {
    "continent", "ocean", "hemisphere", "islands", "island", "peninsula",
    "sea", "gulf", "lake", "mountains", "desert", "plain", "ice",
}
POINT_KINDS = {"point", "peak"}  # полюса и вершины: точка с допуском по углу
TARGET_ONLY_KINDS = POINT_KINDS | {"line"}

# Чем больше число, тем «конкретнее» объект для фразы «Это …».
KIND_PRIORITY = {
    "hemisphere": 1,
    "continent": 2, "ocean": 2, "islands": 2,
    "island": 3, "peninsula": 3, "sea": 3, "gulf": 3, "lake": 3,
    "mountains": 4, "desert": 4, "plain": 4,
    "ice": 2,  # лёд Антарктиды называем материком: «Это Антарктида»
}


@dataclass
class Part:
    surfaces: set[int]
    polygon: np.ndarray | None = None  # (N, 2) широта, долгота
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float | None = None
    lon_max: float | None = None
    colors: np.ndarray | None = None  # (K, 3) — только грани этих цветов модели
    min_height: float | None = None  # только грани не ниже (высота над морем, единицы модели)
    max_height: float | None = None  # только грани не выше

    def contains(self, latlon: np.ndarray) -> np.ndarray:
        lat, lon = latlon[:, 0], latlon[:, 1]
        inside = (lat >= self.lat_min) & (lat <= self.lat_max)
        if self.lon_min is not None and self.lon_max is not None:
            if self.lon_min <= self.lon_max:
                inside &= (lon >= self.lon_min) & (lon <= self.lon_max)
            else:  # диапазон через 180°
                inside &= (lon >= self.lon_min) | (lon <= self.lon_max)
        if self.polygon is not None:
            in_poly = np.zeros(len(latlon), dtype=bool)
            for shift in (-360.0, 0.0, 360.0):
                in_poly |= point_in_polygon(lat, lon + shift, self.polygon)
            inside &= in_poly
        return inside


FACTS_PER_REGION = 5


@dataclass(frozen=True)
class Fact:
    """Интересный факт: короткий заголовок и пояснение в 2–3 предложения."""

    title: str
    text: str


@dataclass
class Region:
    id: str
    name: str
    kind: str
    facts: tuple[Fact, ...] = ()
    parts: list[Part] = field(default_factory=list)
    at: tuple[float, float] | None = None  # для точек
    lat: float | None = None  # для линий-параллелей
    faces: np.ndarray | None = None  # bool-маска граней после разметки
    center: tuple[float, float] = (0.0, 0.0)
    radius_deg: float = 0.0  # угловой размер (до самой дальней грани)

    @property
    def is_base(self) -> bool:
        return self.kind in BASE_KINDS

    @property
    def priority(self) -> int:
        return KIND_PRIORITY.get(self.kind, 0)


def _parse_surfaces(value, default: set[int]) -> set[int]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [value]
    if "any" in value:
        return set(SURFACE_NAMES.values())
    return {SURFACE_NAMES[v] for v in value}


def _parse_part(raw: dict, default_surfaces: set[int]) -> Part:
    poly = raw.get("polygon")
    return Part(
        surfaces=_parse_surfaces(raw.get("surface"), default_surfaces),
        polygon=np.array(poly, float) if poly else None,
        lat_min=raw.get("lat_min", -90.0),
        lat_max=raw.get("lat_max", 90.0),
        lon_min=raw.get("lon_min"),
        lon_max=raw.get("lon_max"),
        colors=np.array(raw["colors"], np.uint8) if raw.get("colors") else None,
        min_height=raw.get("min_height"),
        max_height=raw.get("max_height"),
    )


def parse_region(raw: dict) -> Region:
    kind = raw["kind"]
    if kind not in FACE_KINDS | TARGET_ONLY_KINDS:
        raise ValueError(f"неизвестный вид региона {kind!r} у {raw['id']}")
    region = Region(id=raw["id"], name=raw["name"], kind=kind)
    if kind in POINT_KINDS:
        region.at = tuple(raw["at"])
        region.center = region.at
        return region
    if kind == "line":
        region.lat = float(raw["lat"])
        region.center = (region.lat, 0.0)
        return region
    default_surfaces = _parse_surfaces(raw.get("surface", "any"), set())
    raw_parts = raw.get("parts")
    if raw_parts is None:
        raw_parts = [{k: v for k, v in raw.items() if k in
                      ("polygon", "lat_min", "lat_max", "lon_min", "lon_max", "surface", "colors",
                       "min_height", "max_height")}]
    region.parts = [_parse_part(p, default_surfaces) for p in raw_parts]
    return region


class RegionSet:
    def __init__(self, regions: list[Region]):
        ids = [r.id for r in regions]
        if len(ids) != len(set(ids)):
            raise ValueError("повторяющиеся id регионов")
        self.regions = regions
        self.by_id = {r.id: r for r in regions}
        self.membership: np.ndarray | None = None  # (F, R) bool

    @classmethod
    def load(cls, path: Path = DATA_DIR / "regions.json",
             facts_path: Path | None = DATA_DIR / "facts.json") -> "RegionSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        regions = cls([parse_region(r) for r in data["regions"]])
        if facts_path is not None:
            regions.attach_facts(json.loads(Path(facts_path).read_text(encoding="utf-8")))
        return regions

    def attach_facts(self, data: dict) -> None:
        """Раздаёт объектам факты {id: [{title, text}, …]}: ровно пять у каждого, без чужих id."""
        extra = sorted(set(data) - set(self.by_id))
        if extra:
            raise ValueError(f"факты для неизвестных объектов: {', '.join(extra)}")
        for region in self.regions:
            raw = data.get(region.id, [])
            if len(raw) != FACTS_PER_REGION:
                raise ValueError(f"у объекта «{region.name}» ({region.id}) {len(raw)} фактов "
                                 f"вместо {FACTS_PER_REGION}")
            region.facts = tuple(Fact(str(f["title"]), str(f["text"])) for f in raw)

    def __getitem__(self, region_id: str) -> Region:
        return self.by_id[region_id]

    def assign(self, model: EarthModel) -> "RegionSet":
        """Размечает грани модели регионами."""
        n_faces = len(model)
        face_regions = [r for r in self.regions if r.kind in FACE_KINDS]
        membership = np.zeros((n_faces, len(self.regions)), dtype=bool)
        base_taken = np.zeros(n_faces, dtype=bool)
        col = {r.id: i for i, r in enumerate(self.regions)}
        for region in face_regions:
            mask = np.zeros(n_faces, dtype=bool)
            for part in region.parts:
                ok = np.isin(model.surface, list(part.surfaces))
                if part.colors is not None:
                    ok &= (model.colors[:, None, :] == part.colors[None]).all(-1).any(-1)
                if part.min_height is not None:
                    ok &= model.height >= part.min_height
                if part.max_height is not None:
                    ok &= model.height <= part.max_height
                mask |= ok & part.contains(model.latlon)
            if region.is_base:
                mask &= ~base_taken
                base_taken |= mask
            membership[:, col[region.id]] = mask
            region.faces = mask
            if mask.any():
                dirs = model.centers[mask]
                dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
                c = dirs.mean(axis=0)
                c = c / np.linalg.norm(c)
                region.center = tuple(float(v) for v in vec_to_latlon(c))
                region.radius_deg = float(np.degrees(np.arccos(np.clip(dirs @ c, -1, 1)).max()))
        self.membership = membership
        self._columns = col
        self._model_latlon = model.latlon
        self._center_cache: dict[str, np.ndarray] = {}
        return self

    def regions_of_face(self, face: int) -> list[Region]:
        return [self.regions[i] for i in np.nonzero(self.membership[face])[0]]

    def most_specific(self, face: int) -> Region | None:
        regs = [r for r in self.regions_of_face(face) if r.priority > 1]
        if not regs:
            return None
        return max(regs, key=lambda r: r.priority)

    def face_in(self, face: int, region_id: str) -> bool:
        return bool(self.membership[face, self._columns[region_id]])

    def unassigned(self) -> np.ndarray:
        """Грани, у которых нет ни материка/океана, ни более конкретного объекта."""
        cols = [i for i, r in enumerate(self.regions) if r.priority >= 2]
        return ~self.membership[:, cols].any(axis=1)

    def nearest_target_point(self, region: Region, from_latlon: tuple[float, float]) -> tuple[float, float]:
        """Ближайшая к точке часть объекта — для подсказки направления."""
        if region.kind in POINT_KINDS:
            return region.at
        if region.kind == "line":
            return (region.lat, from_latlon[1])
        if region.faces is None or not region.faces.any():
            return region.center
        return _nearest_center(self._centers_cache(region), from_latlon)

    def _centers_cache(self, region: Region) -> np.ndarray:
        if region.id not in self._center_cache:
            self._center_cache[region.id] = self._model_latlon[region.faces]
        return self._center_cache[region.id]


def _nearest_center(latlons: np.ndarray, p: tuple[float, float]) -> tuple[float, float]:
    vecs = latlon_to_vec(latlons[:, 0], latlons[:, 1])
    d = vecs @ latlon_to_vec(*p)
    best = latlons[int(np.argmax(d))]
    return float(best[0]), float(best[1])

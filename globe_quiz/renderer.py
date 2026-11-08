"""Программная отрисовка глобуса средствами pygame.

Ортографическая проекция, отсечение задних граней, плоская заливка граней
с простым освещением. Сначала рисуется слой воды, потом суша, каждый слой —
от дальних граней к ближним (алгоритм художника).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pygame

from .geo import latlon_to_vec, lon_delta, wrap_lon
from .model import EarthModel

MAX_PITCH = 75.0
MIN_ZOOM = 1.0
MAX_ZOOM = 5.0
GLOBE_FILL = 0.85  # доля меньшей стороны области, которую занимает глобус при зуме 1

_LIGHT = np.array([-0.45, 0.55, 0.70])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
_AMBIENT = 0.55
_LIMB_DEPTH = 0.35  # насколько за край заглядываем в поисках выступающей суши
_UNDERLAY_RADIUS = 1.0
_UNDERLAY = (50, 108, 150)  # цвет воды у края диска (с учётом освещения)


def view_matrix(center_lat: float, center_lon: float) -> np.ndarray:
    """Матрица поворота: точка (center_lat, center_lon) смотрит на зрителя, север сверху.

    Координаты вида: x — вправо, y — вверх, z — на зрителя.
    """
    a = math.radians(-90.0 - center_lon)
    b = math.radians(center_lat)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rx @ ry


@dataclass
class View:
    """Положение глобуса: какая точка в центре и насколько он приближен."""

    center_lat: float = 20.0
    center_lon: float = 15.0
    zoom: float = 1.0

    def clamp(self) -> None:
        self.center_lat = max(-MAX_PITCH, min(MAX_PITCH, self.center_lat))
        self.center_lon = wrap_lon(self.center_lon)
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))

    def key(self) -> tuple:
        return (round(self.center_lat, 4), round(self.center_lon, 4), round(self.zoom, 4))


@dataclass
class Frame:
    """Результат последней отрисовки — нужен для выбора грани под курсором."""

    order: np.ndarray  # индексы граней в порядке отрисовки
    pts: np.ndarray  # (N, 3, 2) экранные координаты вершин в том же порядке
    rot: np.ndarray  # матрица вида
    scale: float
    center: tuple[float, float]


@dataclass
class Highlight:
    faces: np.ndarray  # bool-маска граней
    color: tuple[int, int, int] = (255, 230, 60)
    strength: float = 0.6  # 0..1, доля смешивания


@dataclass
class GlobeRenderer:
    model: EarthModel
    view: View = field(default_factory=View)
    frame: Frame | None = None
    face_colors: np.ndarray | None = None  # замена цветов граней (отладочная раскраска)

    def __post_init__(self) -> None:
        c = self.model.centers
        radial = (self.model.normals * c).sum(axis=1) / np.linalg.norm(c, axis=1)
        self._lying = radial > 0.6

    def scale_for(self, rect: pygame.Rect) -> float:
        return min(rect.width, rect.height) / 2 * GLOBE_FILL * self.view.zoom

    def rotation(self) -> np.ndarray:
        return view_matrix(self.view.center_lat, self.view.center_lon)

    def draw(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        highlights: list[Highlight] | None = None,
        outside: tuple[int, int, int] | None = None,
    ) -> None:
        """Рисует глобус в области rect.

        outside — цвет фона: если задан, всё, что вылезло за круг глобуса
        (приподнятая суша у края), закрашивается им. Так у силуэта не остаётся
        щелей, сквозь которые виден космос.
        """
        m = self.model
        rot = self.rotation()
        scale = self.scale_for(rect)
        cx, cy = rect.center

        n = m.normals @ rot.T
        # Рисуем лицевые грани ближнего полушария, а за краем — только те, что
        # выступают за силуэт водной сферы (приподнятая суша у края планеты).
        # Остальные грани обратной стороны отбрасываем: иначе стенки суши
        # рисовались бы поверх ближней воды (суша идёт вторым слоем).
        cz = m.centers @ rot[2]
        facing = n[:, 2] > 0.0
        cand = np.nonzero(facing & (cz > -_LIMB_DEPTH))[0]
        verts = m.tris[cand] @ rot.T  # (N,3,3)
        beyond_limb = (verts[:, :, 0] ** 2 + verts[:, :, 1] ** 2).max(axis=1) > 1.0
        # За краем оставляем только «лежачие» грани (верх суши), без стенок
        # берегов: иначе у края торчат зубцы с просветами космоса между ними.
        lying = self._lying[cand]
        keep = (cz[cand] > 0.0) | (beyond_limb & lying)
        idx, verts = cand[keep], verts[keep]
        depth = verts[:, :, 2].mean(axis=1)

        # Подложка — круг водного цвета: закрывает щели между гранями у края диска.
        pygame.draw.circle(surf, _UNDERLAY, (cx, cy), scale * _UNDERLAY_RADIUS)

        # Сначала вода (слой 0), потом суша (слой 1); внутри слоя — от дальних к ближним.
        layer = (~m.is_water[idx]).astype(np.int8)
        order_local = np.lexsort((depth, layer))
        idx = idx[order_local]
        verts = verts[order_local]

        pts = np.empty((len(idx), 3, 2))
        pts[:, :, 0] = cx + verts[:, :, 0] * scale
        pts[:, :, 1] = cy - verts[:, :, 1] * scale

        light = _AMBIENT + (1 - _AMBIENT) * np.clip(n[idx] @ _LIGHT, 0, 1)
        base = m.colors if self.face_colors is None else self.face_colors
        colors = base[idx].astype(np.float32)
        if highlights:
            for hl in highlights:
                mask = hl.faces[idx]
                if mask.any():
                    k = float(np.clip(hl.strength, 0, 1))
                    colors[mask] = colors[mask] * (1 - k) + np.array(hl.color, np.float32) * k
        colors = np.clip(colors * light[:, None], 0, 255).astype(np.uint8)

        # Отбрасываем грани, которые на экране целиком за пределами области.
        on_screen = (
            (pts[:, :, 0].max(1) >= rect.left)
            & (pts[:, :, 0].min(1) <= rect.right)
            & (pts[:, :, 1].max(1) >= rect.top)
            & (pts[:, :, 1].min(1) <= rect.bottom)
        )
        idx, pts, colors = idx[on_screen], pts[on_screen], colors[on_screen]

        polygon = pygame.draw.polygon
        for poly, col in zip(pts.round().astype(np.int32).tolist(), colors.tolist()):
            polygon(surf, col, poly)

        if outside is not None:
            self._clip_to_disk(surf, rect, (cx, cy), scale, outside)
        self.frame = Frame(order=idx, pts=pts, rot=rot, scale=scale, center=(cx, cy))

    def _clip_to_disk(self, surf, rect, center, scale, outside) -> None:
        """Закрашивает всё за кругом глобуса цветом фона (атмосферу рисует GlobeWidget)."""
        key = (rect.size, center[0] - rect.left, center[1] - rect.top, round(scale, 2), tuple(outside))
        if getattr(self, "_mask_key", None) != key:
            hole = (255, 0, 255) if tuple(outside) != (255, 0, 255) else (0, 255, 0)
            mask = pygame.Surface(rect.size)
            mask.fill(outside)
            pygame.draw.circle(mask, hole, (center[0] - rect.left, center[1] - rect.top),
                               scale * _UNDERLAY_RADIUS)
            mask.set_colorkey(hole)
            self._mask, self._mask_key = mask, key
        surf.blit(self._mask, rect.topleft)

    # --- Геометрия экрана -------------------------------------------------

    def project_latlon(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Экранные координаты точек (lat, lon) и маска видимости (лицевая сторона)."""
        fr = self.frame
        v = latlon_to_vec(np.asarray(lat, float), np.asarray(lon, float)) @ fr.rot.T
        xy = np.stack([fr.center[0] + v[..., 0] * fr.scale, fr.center[1] - v[..., 1] * fr.scale], -1)
        return xy, v[..., 2] > 0

    def pick_face(self, pos: tuple[float, float]) -> int | None:
        """Индекс грани, нарисованной сверху в точке экрана, или None."""
        fr = self.frame
        if fr is None or len(fr.order) == 0:
            return None
        if math.dist(pos, fr.center) > fr.scale * _UNDERLAY_RADIUS:
            return None  # за кругом глобуса — космос
        inside = _points_in_triangles(np.asarray(pos, float), fr.pts)
        hits = np.nonzero(inside)[0]
        if len(hits) == 0:
            return None
        return int(fr.order[hits[-1]])

    def screen_to_latlon(self, pos: tuple[float, float]) -> tuple[float, float] | None:
        """Точка на сфере под курсором (по пересечению с единичной сферой)."""
        fr = self.frame
        if fr is None:
            return None
        x = (pos[0] - fr.center[0]) / fr.scale
        y = -(pos[1] - fr.center[1]) / fr.scale
        r2 = x * x + y * y
        if r2 > 1.0:
            x, y = x / math.sqrt(r2), y / math.sqrt(r2)
            r2 = 1.0
        v_view = np.array([x, y, math.sqrt(max(0.0, 1.0 - r2))])
        v = fr.rot.T @ v_view
        lat = math.degrees(math.asin(max(-1.0, min(1.0, v[1]))))
        lon = -math.degrees(math.atan2(v[2], v[0]))
        return lat, lon

    def distance_to_faces(self, pos: tuple[float, float], faces: np.ndarray) -> float:
        """Экранное расстояние от точки до ближайшей видимой грани из маски faces."""
        fr = self.frame
        if fr is None:
            return math.inf
        sel = faces[fr.order]
        if not sel.any():
            return math.inf
        return float(_distance_to_triangles(np.asarray(pos, float), fr.pts[sel]).min())

    def distance_to_points(self, pos, lat, lon) -> float:
        xy, vis = self.project_latlon(lat, lon)
        xy = np.atleast_2d(xy)[np.atleast_1d(vis)]
        if len(xy) == 0:
            return math.inf
        return float(np.linalg.norm(xy - np.asarray(pos, float), axis=1).min())

    def distance_to_parallel(self, pos, lat: float) -> float:
        """Расстояние до видимой части параллели (например, экватора)."""
        lons = np.arange(-180.0, 180.0, 1.0)
        xy, vis = self.project_latlon(np.full_like(lons, lat), lons)
        best = math.inf
        p = np.asarray(pos, float)
        a, b = xy[:-1], xy[1:]
        seg_vis = vis[:-1] & vis[1:]
        if seg_vis.any():
            best = float(_distance_to_segments(p, a[seg_vis], b[seg_vis]).min())
        return best

    def visible_parallel(self, lat: float) -> list[list[tuple[float, float]]]:
        """Видимые куски параллели как ломаные для рисования."""
        lons = np.arange(-180.0, 181.0, 2.0)
        xy, vis = self.project_latlon(np.full_like(lons, lat), lons)
        runs, cur = [], []
        for p, v in zip(xy.tolist(), vis.tolist()):
            if v:
                cur.append(tuple(p))
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        return [r for r in runs if len(r) > 1]

    def globe_radius_px(self) -> float:
        return self.frame.scale if self.frame else 0.0


def ease_view_towards(view: View, lat: float, lon: float, zoom: float, k: float) -> bool:
    """Плавно двигает вид к цели. Возвращает True, когда цель достигнута."""
    lat = max(-MAX_PITCH, min(MAX_PITCH, lat))
    dlat = lat - view.center_lat
    dlon = lon_delta(view.center_lon, lon)
    dz = zoom - view.zoom
    view.center_lat += dlat * k
    view.center_lon = wrap_lon(view.center_lon + dlon * k)
    view.zoom += dz * k
    done = abs(dlat) < 0.05 and abs(dlon) < 0.05 and abs(dz) < 0.005
    if done:
        view.center_lat, view.center_lon, view.zoom = lat, wrap_lon(lon), zoom
    return done


# --- Векторная геометрия на плоскости -------------------------------------


def _points_in_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    def cross(o, u, w):
        return (u[:, 0] - o[:, 0]) * (w[1] - o[:, 1]) - (u[:, 1] - o[:, 1]) * (w[0] - o[:, 0])

    d1 = cross(a, b, p)
    d2 = cross(b, c, p)
    d3 = cross(c, a, p)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _distance_to_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = (ab**2).sum(-1)
    t = np.clip(((p - a) * ab).sum(-1) / np.where(denom > 0, denom, 1), 0, 1)
    proj = a + ab * t[..., None]
    return np.linalg.norm(proj - p, axis=-1)


def _distance_to_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    inside = _points_in_triangles(p, tri)
    d = np.minimum.reduce(
        [
            _distance_to_segments(p, tri[:, 0], tri[:, 1]),
            _distance_to_segments(p, tri[:, 1], tri[:, 2]),
            _distance_to_segments(p, tri[:, 2], tri[:, 0]),
        ]
    )
    return np.where(inside, 0.0, d)

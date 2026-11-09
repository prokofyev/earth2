"""Программная отрисовка глобуса средствами pygame.

Ортографическая проекция, отсечение задних граней, плоская заливка граней
с простым освещением. Модель — одна замкнутая поверхность (океан на уровне
моря, суша поднята над ним), грани рисуются от дальних к ближним (алгоритм
художника). Высокие горы обратной стороны выглядывают за край океана.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pygame

from .geo import latlon_to_vec, lon_delta, vec_to_latlon, wrap_lon
from .model import WATER_COLOR, EarthModel

MAX_PITCH = 75.0
MIN_ZOOM = 1.0
MAX_ZOOM = 5.0
GLOBE_FILL = 0.85  # доля меньшей стороны области, которую занимает глобус при зуме 1

_LIGHT = np.array([-0.45, 0.55, 0.70])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
_AMBIENT = 0.55
_SEAM_PX = 0.6  # на сколько пикселей раздвигаем грани, чтобы закрыть швы
_SEAM_PX_RIM = 1.0  # то же за кругом океана, где под гранями космос
_OUTER_RADIUS = 1.0  # самая высокая точка модели — всё, что видно, внутри этого круга
_DISK_KEY = (255, 0, 255)
_DISK_HOLE = (0, 255, 0)  # цвет круга в маске края океана: он становится прозрачным
_OCEAN_TEMPLATE_PX = 1024  # шаблон освещённого круга океана — один на любой масштаб


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
        # Насколько за край океана может выглядывать гора высотой до радиуса 1:
        # точка с глубиной −d видна за кругом океана, пока √(1 − d²) > радиус моря.
        sea = self.model.sea_radius
        self._limb_depth = math.sqrt(max(0.0, 1.0 - sea * sea)) + 0.02
        self._radius_grid: np.ndarray | None = None

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
        # Берём лицевые грани ближней стороны и сушу обратной стороны не глубже
        # _limb_depth: горы там выглядывают за край океана.
        cz = m.centers @ rot[2]
        idx = np.nonzero((n[:, 2] > 0.0) & (cz > -self._limb_depth))[0]
        verts = m.tris[idx] @ rot.T  # (N,3,3)
        front = cz[idx] >= 0.0
        water = m.is_water[idx]
        reach = (verts[:, :, 0] ** 2 + verts[:, :, 1] ** 2).max(axis=1)
        # Вода и суша обратной стороны, целиком лежащие внутри круга океана, не видны.
        keep = front | (~water & (reach > m.sea_radius**2))
        idx, verts, front, water = idx[keep], verts[keep], front[keep], water[keep]
        # Порядок: суша обратной стороны → (круг океана) → вода → суша ближней стороны,
        # внутри группы — от дальних граней к ближним.
        group = np.where(~front, 0, np.where(water, 1, 2))
        order_local = np.lexsort((verts[:, :, 2].mean(axis=1), group))
        idx, verts, group = idx[order_local], verts[order_local], group[order_local]

        pts = np.empty((len(idx), 3, 2))
        pts[:, :, 0] = cx + verts[:, :, 0] * scale
        pts[:, :, 1] = cy - verts[:, :, 1] * scale
        # Заливка соседних треугольников в pygame оставляет одиночные пиксели на
        # швах. На фоне космоса (горы за краем океана) они видны как точки —
        # поэтому раздвигаем вершины каждой грани от её центра на полпикселя.
        # За кругом океана подложки нет — там раздвигаем сильнее.
        mid = pts.mean(axis=1, keepdims=True)
        out = pts - mid
        reach_px = np.hypot(pts[:, :, 0] - cx, pts[:, :, 1] - cy).max(axis=1)
        seam = np.where(reach_px > scale * m.sea_radius - 1, _SEAM_PX_RIM, _SEAM_PX)
        pts_fill = pts + out / np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-6) * seam[:, None, None]

        light = _AMBIENT + (1 - _AMBIENT) * np.clip(n[idx] @ _LIGHT, 0, 1)
        base = m.colors if self.face_colors is None else self.face_colors
        colors = base[idx].astype(np.float32)
        lit = np.zeros(len(idx), bool)  # подсвеченные грани
        if highlights:
            for hl in highlights:
                mask = hl.faces[idx]
                if mask.any():
                    k = float(np.clip(hl.strength, 0, 1))
                    colors[mask] = colors[mask] * (1 - k) + np.array(hl.color, np.float32) * k
                    lit |= mask
        colors = np.clip(colors * light[:, None], 0, 255).astype(np.uint8)

        # Отбрасываем грани, которые на экране целиком за пределами области.
        on_screen = (
            (pts[:, :, 0].max(1) >= rect.left)
            & (pts[:, :, 0].min(1) <= rect.right)
            & (pts[:, :, 1].max(1) >= rect.top)
            & (pts[:, :, 1].min(1) <= rect.bottom)
        )
        idx, pts, colors, group, lit = idx[on_screen], pts[on_screen], colors[on_screen], group[on_screen], lit[on_screen]
        pts_fill, reach_px = pts_fill[on_screen], reach_px[on_screen]

        polygon = pygame.draw.polygon
        # Океан — гладкая сфера: рисуем его одним готовым кругом, а не тысячами
        # треугольников. Грани воды остаются в кадре для тыка и рисуются, только
        # когда их подсвечивают (или в отладочной раскраске).
        draw_face = (group != 1) | lit | (self.face_colors is not None)
        sel = np.nonzero(draw_face)[0]
        polys = pts_fill[sel].round().astype(np.int32).tolist()
        cols = colors[sel].tolist()
        n_back = int(np.searchsorted(group[sel], 1))
        for poly, col in zip(polys[:n_back], cols[:n_back]):
            polygon(surf, col, poly)
        self._blit_ocean(surf, (cx, cy), scale * m.sea_radius)
        for poly, col in zip(polys[n_back:], cols[n_back:]):
            polygon(surf, col, poly)
        # За кругом океана подложки нет: швы между гранями там видны точками
        # космоса, поэтому такие грани ещё и обводим. Стенки обрывов, видные
        # строго с ребра (тоньше 0,1 пикселя), не обводим: иначе у края торчит
        # «забор» из линий.
        rim = np.nonzero(reach_px[sel] > scale * m.sea_radius - 1)[0]
        if len(rim):
            rp = pts[sel[rim]]
            e = rp[:, [1, 2, 0]] - rp
            longest = np.sqrt((e**2).sum(axis=2)).max(axis=1)
            area2 = np.abs(e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0])
            for k in rim[area2 > 0.1 * np.maximum(longest, 1e-6)].tolist():
                polygon(surf, cols[k], polys[k], 1)

        if outside is not None:
            self._clip_to_disk(surf, rect, (cx, cy), scale, outside)
        self.frame = Frame(order=idx, pts=pts, rot=rot, scale=scale, center=(cx, cy))

    def _ocean_template(self) -> pygame.Surface:
        """Освещённая гладкая сфера в квадрате _OCEAN_TEMPLATE_PX: её край — край квадрата.

        Освещение зависит только от точки в долях радиуса, поэтому один шаблон годится
        для любого масштаба. За кругом цвет края продолжен: при масштабировании на краю
        нет каймы, а точный край отрезает маска в _blit_ocean.
        """
        template = getattr(self, "_template", None)
        if template is None:
            n = _OCEAN_TEMPLATE_PX
            c = (n - 1) / 2
            x, y = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
            x, y = x / (n / 2), y / (n / 2)
            rho = np.maximum(1.0, np.hypot(x, y))  # за кругом — точка края в том же направлении
            x, y = x / rho, y / rho
            nz = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
            dot = x * _LIGHT[0] - y * _LIGHT[1] + nz * _LIGHT[2]
            light = _AMBIENT + (1 - _AMBIENT) * np.clip(dot, 0, 1)
            rgb = np.clip(np.array(WATER_COLOR, float) * light[..., None], 0, 255).astype(np.uint8)
            template = pygame.surfarray.make_surface(rgb)
            if pygame.display.get_surface() is not None:
                template = template.convert()
            self._template = template
        return template

    def _blit_ocean(self, surf: pygame.Surface, center: tuple[float, float], radius: float) -> None:
        """Круг океана: из шаблона масштабируется только та часть, что видна в surf.

        Круг на 2 пикселя шире моря — стыки обрывов с краем океана без щелей.
        Цена зависит от размера видимой части, а не от радиуса, поэтому сильный зум
        не тормозит.
        """
        key = radius + 2
        cx, cy = center
        left, top = cx - key, cy - key
        box = pygame.Rect(math.floor(left), math.floor(top), math.ceil(2 * key) + 2, math.ceil(2 * key) + 2)
        win = box.clip(surf.get_rect())
        if win.width <= 0 or win.height <= 0:
            return
        template = self._ocean_template()
        k = _OCEAN_TEMPLATE_PX / (2 * key)  # пикселей шаблона на пиксель экрана
        tx0 = max(0, math.floor((win.left - left) * k))
        ty0 = max(0, math.floor((win.top - top) * k))
        tx1 = min(_OCEAN_TEMPLATE_PX, max(tx0 + 1, math.ceil((win.right - left) * k)))
        ty1 = min(_OCEAN_TEMPLATE_PX, max(ty0 + 1, math.ceil((win.bottom - top) * k)))
        piece = pygame.transform.scale(template.subsurface((tx0, ty0, tx1 - tx0, ty1 - ty0)), win.size)
        corners = ((win.left, win.top), (win.right, win.top), (win.left, win.bottom), (win.right, win.bottom))
        if max(math.hypot(x - cx, y - cy) for x, y in corners) > key:
            # Край круга виден — отрезаем всё за ним маской с точной окружностью.
            mask = getattr(self, "_ocean_mask", None)
            if mask is None or mask.get_size() != win.size:
                mask = pygame.Surface(win.size)
                mask.set_colorkey(_DISK_HOLE)
                self._ocean_mask = mask
            mask.fill(_DISK_KEY)
            pygame.draw.circle(mask, _DISK_HOLE, (cx - win.left, cy - win.top), key)
            piece.blit(mask, (0, 0))
            piece.set_colorkey(_DISK_KEY)
        surf.blit(piece, win.topleft)

    def _clip_to_disk(self, surf, rect, center, scale, outside) -> None:
        """Закрашивает всё за кругом радиуса 1 цветом фона — страховка от осколков за силуэтом."""
        key = (rect.size, center[0] - rect.left, center[1] - rect.top, round(scale, 2), tuple(outside))
        if getattr(self, "_mask_key", None) != key:
            hole = (255, 0, 255) if tuple(outside) != (255, 0, 255) else (0, 255, 0)
            mask = pygame.Surface(rect.size)
            mask.fill(outside)
            pygame.draw.circle(mask, hole, (center[0] - rect.left, center[1] - rect.top),
                               scale * _OUTER_RADIUS + 1)
            mask.set_colorkey(hole)
            self._mask, self._mask_key = mask, key
        surf.blit(self._mask, rect.topleft)

    # --- Геометрия экрана -------------------------------------------------

    def surface_radius(self, lat, lon) -> np.ndarray:
        """Радиус поверхности модели над точкой (lat, lon) — для меток на рельефе."""
        if self._radius_grid is None:
            self._radius_grid = _radius_grid(self.model)
        g = self._radius_grid
        rows, cols = g.shape
        y = np.clip((90.0 - np.asarray(lat, float)) / _GRID_DEG, 0, rows - 1.000001)
        x = (np.asarray(lon, float) + 180.0) / _GRID_DEG
        i0, j0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = y - i0, x - j0
        i1, ja, jb = np.minimum(i0 + 1, rows - 1), j0 % cols, (j0 + 1) % cols
        return ((g[i0, ja] * (1 - fx) + g[i0, jb] * fx) * (1 - fy)
                + (g[i1, ja] * (1 - fx) + g[i1, jb] * fx) * fy)

    def project_latlon(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Экранные координаты точек (lat, lon) на поверхности рельефа и маска видимости."""
        fr = self.frame
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        v = latlon_to_vec(lat, lon) * self.surface_radius(lat, lon)[..., None] @ fr.rot.T
        xy = np.stack([fr.center[0] + v[..., 0] * fr.scale, fr.center[1] - v[..., 1] * fr.scale], -1)
        return xy, v[..., 2] > 0

    def pick_face(self, pos: tuple[float, float]) -> int | None:
        """Индекс грани, нарисованной сверху в точке экрана, или None."""
        fr = self.frame
        if fr is None or len(fr.order) == 0:
            return None
        if math.dist(pos, fr.center) > fr.scale * _OUTER_RADIUS:
            return None  # за кругом глобуса — космос
        inside = _points_in_triangles(np.asarray(pos, float), fr.pts)
        hits = np.nonzero(inside)[0]
        if len(hits) == 0:
            return None
        return int(fr.order[hits[-1]])

    def screen_to_latlon(self, pos: tuple[float, float]) -> tuple[float, float] | None:
        """Точка поверхности под курсором: пересечение луча взгляда с гранью, нарисованной сверху."""
        fr = self.frame
        if fr is None:
            return None
        x = (pos[0] - fr.center[0]) / fr.scale
        y = -(pos[1] - fr.center[1]) / fr.scale
        face = self.pick_face(pos)
        if face is not None:
            a, b, c = self.model.tris[face] @ fr.rot.T
            n = np.cross(b - a, c - a)
            if abs(n[2]) > 1e-12:  # плоскость грани: n·(p − a) = 0, p = (x, y, z)
                z = a[2] - (n[0] * (x - a[0]) + n[1] * (y - a[1])) / n[2]
                v = fr.rot.T @ np.array([x, y, z])
                ll = vec_to_latlon(v)
                return float(ll[0]), float(ll[1])
        # Мимо граней — точка на сфере уровня моря (или на её краю).
        sea = self.model.sea_radius
        r2 = (x * x + y * y) / (sea * sea)
        if r2 > 1.0:
            x, y, r2 = x / math.sqrt(r2), y / math.sqrt(r2), 1.0
        v = fr.rot.T @ np.array([x, y, sea * math.sqrt(max(0.0, 1.0 - r2))])
        ll = vec_to_latlon(v)
        return float(ll[0]), float(ll[1])

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


_GRID_DEG = 0.5


def _radius_grid(model: EarthModel) -> np.ndarray:
    """Наибольший радиус поверхности в клетках 0,5° — по точкам внутри граней."""
    rows, cols = int(180 / _GRID_DEG), int(360 / _GRID_DEG)
    grid = np.full(rows * cols, model.sea_radius)
    k = 6
    w = np.array([(i / k, j / k) for i in range(k + 1) for j in range(k + 1 - i)])
    w = np.c_[1 - w.sum(axis=1), w]
    pts = np.einsum("pk,fkd->fpd", w, model.tris).reshape(-1, 3)
    ll = vec_to_latlon(pts)
    i = np.clip(((90.0 - ll[:, 0]) / _GRID_DEG).astype(int), 0, rows - 1)
    j = ((ll[:, 1] + 180.0) / _GRID_DEG).astype(int) % cols
    np.maximum.at(grid, i * cols + j, np.linalg.norm(pts, axis=1))
    return grid.reshape(rows, cols)


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

"""Глобус на экране: вращение перетаскиванием, зум, тык, подсветка, полёт к объекту."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from .renderer import MAX_ZOOM, MIN_ZOOM, GlobeRenderer, Highlight, ease_view_towards

TAP_THRESHOLD_PX = 8
# Атмосфера: полупрозрачный градиент, чуть заходит на край планеты и тает в космосе.
ATMO_INNER = 0.06  # доля радиуса внутрь от края
ATMO_OUTER = 0.20  # доля радиуса наружу от края
ATMO_PEAK_ALPHA = 190
ATMO_EDGE_COLOR = (175, 220, 255)  # у самого края — почти белый голубой
ATMO_OUTER_COLOR = (60, 120, 235)  # снаружи — густо-синий
ZOOM_STEP = 1.35
_COLORKEY = (1, 2, 3)


def make_atmosphere(size: tuple[int, int], center: tuple[float, float], radius: float) -> pygame.Surface:
    """Прозрачная поверхность размера size с кольцом-градиентом атмосферы вокруг круга."""
    glow = pygame.Surface(size, pygame.SRCALPHA)
    r_in = radius * (1 - ATMO_INNER)
    r_out = radius * (1 + ATMO_OUTER)
    thickness = r_out - r_in
    steps = max(1, min(140, int(thickness)))
    step = thickness / steps
    width = max(1, math.ceil(step) + 1)
    for i in range(steps):  # от внешнего края к внутреннему, внутренние кольца поверх
        r = r_out - i * step
        if r <= radius:  # внутри планеты: дымка нарастает к краю
            t = (r - r_in) / (radius - r_in)
            alpha = ATMO_PEAK_ALPHA * t**1.6
            k = 1.0
        else:  # снаружи: плавно тает
            t = (r - radius) / (r_out - radius)
            alpha = ATMO_PEAK_ALPHA * (1 - t) ** 1.5
            k = 1 - t
        color = tuple(int(o + (e - o) * k) for e, o in zip(ATMO_EDGE_COLOR, ATMO_OUTER_COLOR))
        pygame.draw.circle(glow, (*color, int(alpha)), center, r, width)
    return glow


def is_tap(max_shift_px: float, threshold: float = TAP_THRESHOLD_PX) -> bool:
    """Нажатие считается тыком, если курсор за всё время сдвинулся не дальше порога."""
    return max_shift_px <= threshold


@dataclass
class Marker:
    """Точка или параллель, которую нужно нарисовать поверх глобуса."""

    kind: str  # "point" | "line"
    lat: float
    lon: float = 0.0
    color: tuple[int, int, int] = (255, 80, 80)


@dataclass
class _Glow:
    faces: np.ndarray
    color: tuple[int, int, int]
    low: float
    high: float
    period: float


class GlobeWidget:
    def __init__(self, renderer: GlobeRenderer):
        self.renderer = renderer
        self.rect = pygame.Rect(0, 0, 1, 1)
        self.target_zoom = renderer.view.zoom
        self.auto_spin = 0.0  # градусов в секунду, для заставки
        self.render_count = 0
        self._cache: pygame.Surface | None = None
        self._cache_key = None
        self._last_key = None
        self._press: tuple[int, int] | None = None
        self._last: tuple[int, int] | None = None
        self._max_shift = 0.0
        self._flight: tuple[float, float, float] | None = None
        self._glows: dict[str, _Glow] = {}
        self._markers: dict[str, Marker] = {}
        self._time = 0.0
        self._atmo: pygame.Surface | None = None
        self._atmo_key = None

    # --- Управление ------------------------------------------------------

    @property
    def view(self):
        return self.renderer.view

    @property
    def dragging(self) -> bool:
        return self._press is not None and not is_tap(self._max_shift)

    @property
    def busy(self) -> bool:
        """Идёт полёт к объекту — ввод временно не принимается."""
        return self._flight is not None

    def set_rect(self, rect: pygame.Rect) -> None:
        if rect != self.rect:
            self.rect = pygame.Rect(rect)
            self._cache = None

    def zoom_by(self, factor: float) -> None:
        self.target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.target_zoom * factor))

    def fly_to(self, lat: float, lon: float, zoom: float | None = None, screen_dy: float = 0.0) -> None:
        """Плавно повернуть (и приблизить) глобус к точке.

        screen_dy — на сколько пикселей выше центра области глобуса окажется точка:
        центр вида смещается к югу на угол, под которым точка поверхности видна
        на такой высоте. Точка лежит на центральном меридиане вида, поэтому сдвиг точный.
        """
        z = self.target_zoom if zoom is None else max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if screen_dy:
            scale = self.renderer.scale_for(self.rect) / self.view.zoom * z
            radius = float(self.renderer.surface_radius(lat, lon))
            k = max(-0.9, min(0.9, screen_dy / (scale * radius)))
            lat -= math.degrees(math.asin(k))
        self._flight = (lat, lon, z)

    def set_glow(self, name: str, faces: np.ndarray | None, color=(255, 230, 60),
                 low: float = 0.25, high: float = 0.7, period: float = 1.0) -> None:
        """Мерцающая подсветка граней. faces=None убирает подсветку с этим именем."""
        if faces is None or not np.any(faces):
            self._glows.pop(name, None)
        else:
            self._glows[name] = _Glow(np.asarray(faces, bool), color, low, high, period)

    def set_marker(self, name: str, marker: Marker | None) -> None:
        if marker is None:
            self._markers.pop(name, None)
        else:
            self._markers[name] = marker

    def clear_effects(self) -> None:
        self._glows.clear()
        self._markers.clear()

    def handle_event(self, event: pygame.event.Event) -> tuple[int, int] | None:
        """Обрабатывает событие.

        Возвращает точку тыка по глобусу в координатах области глобуса
        (как у renderer.frame) или None.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and not self.busy:
                self._press = self._last = event.pos
                self._max_shift = 0.0
        elif event.type == pygame.MOUSEMOTION and self._press is not None:
            self._max_shift = max(self._max_shift, math.dist(self._press, event.pos))
            if not is_tap(self._max_shift):
                self._rotate_by(event.pos[0] - self._last[0], event.pos[1] - self._last[1])
            self._last = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._press is not None:
            self._max_shift = max(self._max_shift, math.dist(self._press, event.pos))
            tap = is_tap(self._max_shift)
            self._press = None
            local = self.to_local(event.pos)
            if tap and self.renderer.frame is not None and self.renderer.pick_face(local) is not None:
                return local
        elif event.type == pygame.MOUSEWHEEL and not self.busy:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.zoom_by(1.12 ** event.y)
        return None

    def _rotate_by(self, dx: float, dy: float) -> None:
        scale = self.renderer.scale_for(self.rect)
        deg_per_px = math.degrees(1.0 / scale)  # точка под пальцем идёт вместе с пальцем
        self.view.center_lon -= dx * deg_per_px
        self.view.center_lat += dy * deg_per_px
        self.view.clamp()

    # --- Время и рисование ------------------------------------------------

    def update(self, dt: float) -> None:
        self._time += dt
        k = 1.0 - math.exp(-dt * 6.0)
        if self._flight is not None:
            lat, lon, zoom = self._flight
            self.target_zoom = zoom
            if ease_view_towards(self.view, lat, lon, zoom, k):
                self._flight = None
        else:
            if abs(self.target_zoom - self.view.zoom) > 1e-3:
                self.view.zoom += (self.target_zoom - self.view.zoom) * k
            else:
                self.view.zoom = self.target_zoom
            if self.auto_spin and self._press is None:
                self.view.center_lon += self.auto_spin * dt
        self.view.clamp()

    def _highlights(self) -> list[Highlight]:
        out = []
        for g in self._glows.values():
            phase = 0.5 + 0.5 * math.sin(self._time * 2 * math.pi / g.period)
            out.append(Highlight(g.faces, g.color, g.low + (g.high - g.low) * phase))
        return out

    def draw(self, surf: pygame.Surface) -> None:
        key = (self.view.key(), self.rect.size, tuple(sorted(self._glows)))
        moving = key != self._last_key or bool(self._glows)
        self._last_key = key
        if moving and surf.get_rect().contains(self.rect):
            # Глобус вращается или мерцает: рисуем сразу на экран (в области глобуса),
            # без промежуточного кадра — так быстрее, а кадр всё равно новый.
            self.renderer.draw(surf.subsurface(self.rect), pygame.Rect((0, 0), self.rect.size), self._highlights())
            self._cache_key = None
            self.render_count += 1
        else:
            # Глобус стоит: один раз рисуем в запасной кадр и дальше только копируем.
            if self._cache is None or self._cache.get_size() != self.rect.size or key != self._cache_key:
                if self._cache is None or self._cache.get_size() != self.rect.size:
                    self._cache = pygame.Surface(self.rect.size).convert()
                    self._cache.set_colorkey(_COLORKEY)
                self._cache.fill(_COLORKEY)
                self.renderer.draw(self._cache, self._cache.get_rect(), self._highlights(), outside=_COLORKEY)
                self._cache_key = key
                self.render_count += 1
            surf.blit(self._cache, self.rect.topleft)
        self._draw_atmosphere(surf)
        self._draw_markers(surf)

    def _draw_atmosphere(self, surf: pygame.Surface) -> None:
        fr = self.renderer.frame
        if fr is None:
            return
        radius = fr.scale * self.renderer.model.sea_radius  # атмосфера — вокруг океана, горы торчат сквозь неё
        w, h = self.rect.size
        cx, cy = fr.center
        # Кольцо целиком за пределами экрана (сильный зум) — рисовать нечего.
        far = max(math.dist((cx, cy), corner) for corner in ((0, 0), (w, 0), (0, h), (w, h)))
        if radius * (1 - ATMO_INNER) > far:
            return
        # Слой рисуем только в квадрате вокруг кольца (в пределах области глобуса):
        # полупрозрачный слой на всё окно заметно тормозит на большом экране.
        r_out = radius * (1 + ATMO_OUTER) + 2
        box = pygame.Rect(int(cx - r_out), int(cy - r_out), int(2 * r_out) + 1, int(2 * r_out) + 1)
        box = box.clip(pygame.Rect(0, 0, w, h))
        if box.width <= 0 or box.height <= 0:
            return
        key = (box.size, round(cx - box.left, 1), round(cy - box.top, 1), round(radius, 1))
        if key != self._atmo_key:
            glow = make_atmosphere(box.size, (cx - box.left, cy - box.top), radius)
            if pygame.display.get_surface() is not None:
                glow = glow.convert_alpha()  # формат экрана — быстрее накладывается
            self._atmo = glow
            self._atmo_key = key
        # Внутри кольца есть пустой квадрат — его не накладываем, только 4 полосы вокруг.
        ox, oy = self.rect.left + box.left, self.rect.top + box.top
        a = int(radius * (1 - ATMO_INNER) / math.sqrt(2)) - 1
        inner = pygame.Rect(int(cx - box.left) - a, int(cy - box.top) - a, 2 * a, 2 * a).clip(self._atmo.get_rect())
        if a <= 0 or inner.width <= 0 or inner.height <= 0:
            surf.blit(self._atmo, (ox, oy))
            return
        bw, bh = self._atmo.get_size()
        for part in (
            pygame.Rect(0, 0, bw, inner.top),  # сверху
            pygame.Rect(0, inner.bottom, bw, bh - inner.bottom),  # снизу
            pygame.Rect(0, inner.top, inner.left, inner.height),  # слева
            pygame.Rect(inner.right, inner.top, bw - inner.right, inner.height),  # справа
        ):
            if part.width > 0 and part.height > 0:
                surf.blit(self._atmo, (ox + part.left, oy + part.top), part)

    def _draw_markers(self, surf: pygame.Surface) -> None:
        if not self._markers or self.renderer.frame is None:
            return
        ox, oy = self.rect.topleft
        pulse = 0.5 + 0.5 * math.sin(self._time * 2 * math.pi)
        for m in self._markers.values():
            if m.kind == "point":
                xy, vis = self.renderer.project_latlon(m.lat, m.lon)
                if bool(vis):
                    p = (int(xy[0] + ox), int(xy[1] + oy))
                    r = int(10 + 6 * pulse)
                    pygame.draw.circle(surf, m.color, p, r, 4)
                    pygame.draw.circle(surf, m.color, p, 4)
            elif m.kind == "line":
                for run in self.renderer.visible_parallel(m.lat):
                    pts = [(x + ox, y + oy) for x, y in run]
                    pygame.draw.lines(surf, m.color, False, pts, 4)

    # --- Перевод координат для внешнего кода -------------------------------

    def to_local(self, pos: tuple[int, int]) -> tuple[int, int]:
        return pos[0] - self.rect.left, pos[1] - self.rect.top

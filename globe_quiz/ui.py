"""Общие элементы интерфейса: шрифты, цвета, кнопки, текст, звёзды, фон."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pygame

from .renderer import View, view_matrix

BG_TOP = (8, 12, 38)
BG_BOTTOM = (22, 30, 78)
TEXT = (245, 246, 255)
TEXT_DIM = (170, 180, 220)
PANEL = (24, 32, 80)
PANEL_EDGE = (90, 110, 200)
ACCENT = (255, 200, 60)
GOOD = (90, 220, 120)
BAD = (255, 120, 110)
BUTTON = (60, 90, 200)
BUTTON_HOVER = (80, 115, 235)
BUTTON_DISABLED = (60, 66, 100)
STAR = (255, 210, 60)
STAR_EMPTY = (80, 86, 130)


@lru_cache(maxsize=64)
def font(size: int) -> pygame.font.Font:
    """Шрифт с кириллицей (FreeSans Bold, входит в pygame)."""
    return pygame.font.Font(None, max(8, int(size)))


def ui_scale(surface: pygame.Surface) -> float:
    """Единица размера интерфейса — относительно высоты окна (1.0 при высоте 900)."""
    return max(0.55, surface.get_height() / 900)


def wrap_text(text: str, fnt: pygame.font.Font, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        line = ""
        for w in words:
            test = f"{line} {w}".strip()
            if fnt.size(test)[0] <= width or not line:
                line = test
            else:
                lines.append(line)
                line = w
        lines.append(line)
    return lines


def draw_text(surf, text, size, color, pos, anchor="center", width=None, line_gap=1.15) -> pygame.Rect:
    """Рисует текст (с переносом по ширине). Возвращает занятый прямоугольник."""
    fnt = font(size)
    lines = wrap_text(text, fnt, width) if width else [text]
    images = [fnt.render(line, True, color) for line in lines]
    h_line = int(fnt.get_linesize() * line_gap)
    total_h = h_line * (len(images) - 1) + (images[-1].get_height() if images else 0)
    total_w = max((im.get_width() for im in images), default=0)
    box = pygame.Rect(0, 0, total_w, total_h)
    setattr(box, anchor, pos)
    y = box.top
    for im in images:
        r = im.get_rect()
        if anchor in ("center", "midtop", "midbottom"):
            r.midtop = (box.centerx, y)
        elif anchor in ("topright", "midright", "bottomright"):
            r.topright = (box.right, y)
        else:
            r.topleft = (box.left, y)
        surf.blit(im, r)
        y += h_line
    return box


def text_height(text: str, size: int, width: int, line_gap=1.15) -> int:
    fnt = font(size)
    n = len(wrap_text(text, fnt, width))
    return int(fnt.get_linesize() * line_gap) * (n - 1) + fnt.get_height()


def star_points(center, r_outer, r_inner=None, rotation=-90.0):
    r_inner = r_inner if r_inner is not None else r_outer * 0.45
    cx, cy = center
    pts = []
    for i in range(10):
        r = r_outer if i % 2 == 0 else r_inner
        a = math.radians(rotation + i * 36)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def draw_star(surf, center, r, color=STAR, outline=(150, 100, 0)):
    pts = star_points(center, r)
    pygame.draw.polygon(surf, color, pts)
    if outline:
        pygame.draw.polygon(surf, outline, pts, max(1, int(r / 8)))


# Замочек закрытого уровня — по форме образца assets/lock.jpg. Размеры в единицах
# образца: контур 684 × 879, толщина линии 31.
LOCK_LINE = (196, 204, 240)  # контур — светлее TEXT_DIM, чтобы тонкие линии читались
LOCK_FILL = (150, 160, 215, 70)  # корпус слегка залит, карточка видна сквозь него
LOCK_HOLE = (18, 24, 60)  # скважина — темнее карточки, как отверстие
_LOCK_W, _LOCK_H, _LOCK_T = 684, 879, 31
_LOCK_SS = 4  # рисуем крупнее и сглаженно уменьшаем
_lock_cache: dict[tuple, pygame.Surface] = {}


def lock_image(height: int, color=LOCK_LINE) -> pygame.Surface:
    """Готовая картинка замочка заданной высоты (прозрачный фон), из кеша."""
    height = max(8, int(height))
    key = (height, tuple(color))
    img = _lock_cache.get(key)
    if img is not None:
        return img
    u = height * _LOCK_SS / _LOCK_H  # пикселей на единицу образца
    t = max(_LOCK_T, 1.2 * _LOCK_H / height)  # на маленьком замке линия не тоньше ~1,2 пикселя

    def r(x, y, w, h):
        return pygame.Rect(round(x * u), round(y * u), max(1, round(w * u)), max(1, round(h * u)))

    big = pygame.Surface((round(_LOCK_W * u), round(_LOCK_H * u)), pygame.SRCALPHA)
    big.fill((*color, 0))
    clear = (*color, 0)  # прозрачное, но цвета линии: при уменьшении край без тёмной каймы
    line = (*color, 255)
    body_top = 358
    # Дужка: петля, обведённая контуром снаружи и изнутри.
    ox, ow, otop = 98, 487, 0  # внешний край петли
    ix, iw, itop = 195, 293, 97  # внутренний просвет
    leg = body_top + _LOCK_T - otop  # ножки заходят в корпус
    for (x, y, w, h), fill in (
        ((ox, otop, ow, leg), line),
        ((ox + t, otop + t, ow - 2 * t, leg), clear),
        ((ix - t, itop - t, iw + 2 * t, leg), line),
        ((ix, itop, iw, leg), clear),
    ):
        rect = r(x, y, w, h)
        pygame.draw.rect(big, fill, rect, border_top_left_radius=rect.width // 2,
                         border_top_right_radius=rect.width // 2)
    # Корпус со скруглёнными углами, слегка залитый.
    pygame.draw.rect(big, clear, r(0, body_top, _LOCK_W, _LOCK_H - body_top))
    pygame.draw.rect(big, line, r(0, body_top, _LOCK_W, _LOCK_H - body_top), border_radius=round(90 * u))
    pygame.draw.rect(big, LOCK_FILL, r(t, body_top + t, _LOCK_W - 2 * t, _LOCK_H - body_top - 2 * t),
                     border_radius=round(max(1, 90 - t) * u))
    # Скважина: кружок с прорезью, внутри тёмная.
    cx, cy, rad = 342, 569, 81
    sx, sw, sbottom = 293, 97, 748
    for inset, fill in ((0, line), (t, (*LOCK_HOLE, 255))):
        pygame.draw.circle(big, fill, (round(cx * u), round(cy * u)), round((rad - inset) * u))
        slot = r(sx + inset, cy, sw - 2 * inset, sbottom - inset - cy)
        pygame.draw.rect(big, fill, slot, border_bottom_left_radius=slot.width // 2,
                         border_bottom_right_radius=slot.width // 2)
    img = pygame.transform.smoothscale(big, (max(1, round(_LOCK_W * height / _LOCK_H)), height))
    _lock_cache[key] = img
    return img


def draw_lock(surf, center, size, color=LOCK_LINE):
    """Замочек высотой size с центром в center."""
    img = lock_image(size, color)
    surf.blit(img, img.get_rect(center=(round(center[0]), round(center[1]))))


def draw_panel(surf, rect, color=PANEL, edge=PANEL_EDGE, alpha=235, radius=18):
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color, alpha), panel.get_rect(), border_radius=radius)
    pygame.draw.rect(panel, (*edge, 255), panel.get_rect(), 3, border_radius=radius)
    surf.blit(panel, rect.topleft)


@dataclass
class Button:
    text: str
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 10, 10))
    color: tuple = BUTTON
    hover_color: tuple = BUTTON_HOVER
    enabled: bool = True
    visible: bool = True
    _pressed: bool = False

    def handle(self, event) -> bool:
        """True, если по кнопке кликнули (нажали и отпустили на ней)."""
        if not (self.visible and self.enabled):
            self._pressed = False
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._pressed = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            clicked = self._pressed and self.rect.collidepoint(event.pos)
            self._pressed = False
            return clicked
        return False

    def contains(self, pos) -> bool:
        return self.visible and self.rect.collidepoint(pos)

    def draw(self, surf, text_size=None):
        if not self.visible:
            return
        hover = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        color = self.hover_color if hover else self.color
        if not self.enabled:
            color = BUTTON_DISABLED
        radius = min(self.rect.height // 2, 22)
        shadow = self.rect.move(0, 4)
        pygame.draw.rect(surf, (10, 14, 40), shadow, border_radius=radius)
        pygame.draw.rect(surf, color, self.rect, border_radius=radius)
        pygame.draw.rect(surf, (255, 255, 255), self.rect, 2, border_radius=radius)
        size = text_size or int(self.rect.height * 0.5)
        draw_text(surf, self.text, size, TEXT, self.rect.center)


# Кнопка «назад» в левом верхнем углу — одна на всех экранах; Esc делает то же самое.
def back_button(text: str) -> Button:
    return Button(f"‹ {text}")


def place_back_button(btn: Button, s: float) -> None:
    btn.rect = pygame.Rect(int(20 * s), int(20 * s), int(190 * s), int(56 * s))


def draw_back_button(btn: Button, surf, s: float) -> None:
    btn.draw(surf, text_size=int(30 * s))


# Звёздное небо за Землёй: поворачивается тем же видом, что и глобус, и поэтому
# плывёт навстречу материкам, как при облёте планеты.
SKY_THETA_MAX = 120.0  # до какого угла от центра вида показываем небо, градусы
SKY_EDGE_FACTOR = 1.05  # край неба — чуть дальше углов окна (в половинах диагонали)
STAR_AREA_PX = 2500  # в среднем одна звезда на столько пикселей экрана


class Starfield:
    """Фон: тёмно-синий градиент и звёздное небо, которое вращается вместе с глобусом.

    Звёзды — постоянные направления на сфере. Они поворачиваются матрицей вида
    глобуса, а на экран берётся задняя полусфера — небо за Землёй. Поэтому звёзды
    плывут навстречу материкам с той же угловой скоростью, как при облёте Земли.
    На экран небо ложится равновеликой азимутальной проекцией Ламберта с центром
    в середине окна: густота звёзд везде одинаковая, звёзды появляются и исчезают
    только за краями окна. Зум неба не касается.
    """

    def __init__(self, seed: int = 7, view=None):
        self._seed = seed
        self.view = view  # renderer.View глобуса; None — вид по умолчанию
        self._size: tuple[int, int] | None = None
        self._bg: pygame.Surface | None = None
        self._sky: pygame.Surface | None = None
        self._sky_key = None
        self._last_key = None
        self._sprites: dict[tuple[int, int], pygame.Surface] = {}
        self.render_count = 0

    # --- Геометрия -------------------------------------------------------------

    def _prepare(self, size: tuple[int, int]) -> None:
        w, h = size
        self._size = size
        # Проекция Ламберта: точка под углом θ от центра — на расстоянии 2R·sin(θ/2).
        # R — сколько пикселей на радиан у центра экрана.
        edge = SKY_EDGE_FACTOR * math.hypot(w, h) / 2
        self.radius = edge / (2 * math.sin(math.radians(SKY_THETA_MAX) / 2))
        # Вся сфера занимает 4πR² пикселей — столько звёзд, чтобы густота была прежней.
        n = int(4 * math.pi * self.radius**2 / STAR_AREA_PX)
        rng = np.random.default_rng(self._seed)
        dirs = rng.normal(size=(n, 3))
        self._dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
        self._bright = rng.integers(120, 256, n)
        self._big = rng.random(n) < 0.25
        self._bg = None
        self._sky_key = None
        self._last_key = None

    def _view_angles(self) -> tuple[float, float]:
        v = self.view if self.view is not None else View()
        return v.center_lat, v.center_lon

    def star_positions(self, size: tuple[int, int], lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
        """Экранные координаты видимых звёзд (K, 2) и их номера (K,) для вида (lat, lon)."""
        if size != self._size:
            self._prepare(size)
        w, h = size
        v = self._dirs @ view_matrix(lat, lon).T
        # Небо за Землёй: θ отсчитывается от направления «от зрителя» (−z).
        idx = np.nonzero(-v[:, 2] > math.cos(math.radians(SKY_THETA_MAX)))[0]
        k = self.radius * np.sqrt(2.0 / (1.0 - v[idx, 2]))  # 2R·sin(θ/2) / sin θ
        xy = np.stack([w / 2 + v[idx, 0] * k, h / 2 - v[idx, 1] * k], axis=1)
        on = (xy[:, 0] > -3) & (xy[:, 0] < w + 3) & (xy[:, 1] > -3) & (xy[:, 1] < h + 3)
        return xy[on], idx[on]

    # --- Рисование -------------------------------------------------------------

    def _gradient(self, size: tuple[int, int]) -> pygame.Surface:
        if self._bg is None:
            w, h = size
            bg = pygame.Surface((w, h))
            if pygame.display.get_surface() is not None:
                bg = bg.convert()
            for y in range(h):
                t = y / max(1, h - 1)
                c = [int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)]
                pygame.draw.line(bg, c, (0, y), (w, y))
            self._bg = bg
        return self._bg

    def _sprite(self, bright: int, big: bool) -> pygame.Surface:
        key = (bright, big)
        spr = self._sprites.get(key)
        if spr is None:
            r = 2 if big else 1
            spr = pygame.Surface((2 * r + 1, 2 * r + 1))
            spr.fill((0, 0, 0))
            spr.set_colorkey((0, 0, 0))
            pygame.draw.circle(spr, (bright, bright, min(255, bright + 20)), (r, r), r)
            self._sprites[key] = spr
        return spr

    def _paint(self, target: pygame.Surface, size, lat: float, lon: float) -> None:
        target.blit(self._gradient(size), (0, 0))
        xy, idx = self.star_positions(size, lat, lon)
        seq = []
        for (x, y), i in zip(xy.astype(int).tolist(), idx.tolist()):
            big = bool(self._big[i])
            r = 2 if big else 1
            seq.append((self._sprite(int(self._bright[i]), big), (x - r, y - r)))
        target.blits(seq, doreturn=False)
        self.render_count += 1

    def draw(self, surf: pygame.Surface) -> None:
        size = surf.get_size()
        if size != self._size:
            self._prepare(size)
        lat, lon = self._view_angles()
        key = (size, round(lat, 3), round(lon, 3))
        if key == self._sky_key:  # глобус стоит — готовое небо из кеша
            surf.blit(self._sky, (0, 0))
            return
        if key == self._last_key:
            # Второй кадр без поворота: сохраняем небо в кеш, дальше — только копия.
            if self._sky is None or self._sky.get_size() != size:
                self._sky = pygame.Surface(size)
                if pygame.display.get_surface() is not None:
                    self._sky = self._sky.convert()
            self._paint(self._sky, size, lat, lon)
            self._sky_key = key
            surf.blit(self._sky, (0, 0))
            return
        # Глобус вращается: рисуем сразу на экран, без лишней копии через кеш.
        self._paint(surf, size, lat, lon)
        self._last_key = key

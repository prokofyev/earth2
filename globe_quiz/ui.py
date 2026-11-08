"""Общие элементы интерфейса: шрифты, цвета, кнопки, текст, звёзды, фон."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import pygame

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


def draw_lock(surf, center, size, color=TEXT_DIM):
    cx, cy = center
    w, h = size, size * 0.75
    body = pygame.Rect(0, 0, w, h)
    body.midtop = (cx, cy - h * 0.15)
    pygame.draw.arc(surf, color, pygame.Rect(cx - w * 0.33, cy - h * 0.95, w * 0.66, h * 1.4),
                    0, math.pi, max(2, int(size / 8)))
    pygame.draw.rect(surf, color, body, border_radius=int(size / 8))


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


class Starfield:
    """Фон: тёмно-синий градиент со звёздами."""

    def __init__(self, seed: int = 7):
        self._seed = seed
        self._cache: pygame.Surface | None = None

    def draw(self, surf: pygame.Surface) -> None:
        if self._cache is None or self._cache.get_size() != surf.get_size():
            w, h = surf.get_size()
            bg = pygame.Surface((w, h)).convert()
            for y in range(h):
                t = y / max(1, h - 1)
                c = [int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)]
                pygame.draw.line(bg, c, (0, y), (w, y))
            rng = random.Random(self._seed)
            for _ in range(w * h // 2500):
                x, y = rng.randrange(w), rng.randrange(h)
                b = rng.randint(120, 255)
                r = rng.choice([1, 1, 1, 2])
                pygame.draw.circle(bg, (b, b, min(255, b + 20)), (x, y), r)
            self._cache = bg
        surf.blit(self._cache, (0, 0))

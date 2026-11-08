"""Отладочный режим --debug-regions: раскраска граней по регионам и координаты по клику."""

from __future__ import annotations

import colorsys

import numpy as np
import pygame

from ..answers import inspect_tap
from ..app import Scene
from ..ui import TEXT, TEXT_DIM, draw_text, ui_scale


def _color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.6, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


class DebugScene(Scene):
    def enter(self) -> None:
        app = self.app
        colors = np.full((len(app.model), 3), (255, 0, 255), dtype=np.uint8)  # неразмеченное — пурпурным
        for i, r in enumerate(app.regions.regions):
            if r.is_base and r.faces is not None:
                colors[r.faces] = _color(i)
        app.globe.renderer.face_colors = colors
        self.info = "Кликни по глобусу — в консоли и здесь появятся координаты и регионы"

    def layout(self, size) -> None:
        w, h = size
        self.s = ui_scale(pygame.display.get_surface())
        top = int(110 * self.s)
        self.app.globe.set_rect(pygame.Rect(0, top, w, h - top))

    def handle(self, event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.running = False
            return
        pos = self.app.globe.handle_event(event)
        if pos is None:
            return
        tap = inspect_tap(self.app.globe.renderer, self.app.regions, pos)
        if tap is None:
            return
        lat, lon = tap.latlon
        names = ", ".join(r.id for r in tap.regions) or "—"
        self.info = f"{lat:.1f}, {lon:.1f}: {names}"
        print(f"[{lat:.2f}, {lon:.2f}] грань {tap.face}: {names}")

    def update(self, dt) -> None:
        self.app.globe.update(dt)

    def draw(self, surf) -> None:
        w, _ = surf.get_size()
        self.app.starfield.draw(surf)
        self.app.globe.draw(surf)
        unassigned = int(self.app.regions.unassigned().sum())
        draw_text(surf, f"Отладка регионов · неразмеченных граней: {unassigned}", int(34 * self.s),
                  TEXT, (w // 2, int(35 * self.s)))
        draw_text(surf, self.info, int(26 * self.s), TEXT_DIM, (w // 2, int(80 * self.s)), width=w - 40)

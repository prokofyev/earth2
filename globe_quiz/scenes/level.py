"""Сцена уровня: вопрос, глобус, ответы, подсказки, звёзды."""

from __future__ import annotations

import pygame

from ..answers import cap, check_answer, correct_message, inspect_tap, miss_message, nearest_target, reveal_message
from ..app import Scene
from ..fx import StarFlight
from ..globe_widget import ZOOM_STEP, Marker
from ..quiz import LevelRun, Result
from ..regions import Region
from ..ui import (ACCENT, BAD, GOOD, PANEL, TEXT, TEXT_DIM, Button, draw_panel, draw_star, draw_text,
                  text_height, ui_scale)

ASK, MISS, CORRECT, REVEAL = "ask", "miss", "correct", "reveal"
FOUND_COLOR = (90, 230, 110)
HINT_COLOR = (255, 230, 60)
REVEAL_COLOR = (255, 150, 60)


def zoom_for(region: Region) -> float:
    if region.kind == "peak":
        return 2.5  # вершину показываем поближе
    if region.kind in ("point", "line", "hemisphere"):
        return 1.0
    r = region.radius_deg
    if r < 6:
        return 3.5
    if r < 15:
        return 2.2
    if r < 35:
        return 1.4
    return 1.0


class LevelScene(Scene):
    def __init__(self, app, level):
        super().__init__(app)
        self.level = level

    def enter(self) -> None:
        self.run = LevelRun(self.level)
        self.state = ASK
        self.title = ""
        self.body = ""
        self.flights: list[StarFlight] = []
        self.shown_stars = 0
        self.ui_press = False
        self.btn_menu = Button("‹ В меню")
        self.btn_plus = Button("+")
        self.btn_minus = Button("−")
        self.btn_next = Button("Дальше ›", visible=False)
        g = self.app.globe
        g.view.center_lat, g.view.center_lon = 20.0, 20.0
        g.view.zoom = g.target_zoom = 1.0

    @property
    def buttons(self) -> list[Button]:
        return [self.btn_menu, self.btn_plus, self.btn_minus, self.btn_next]

    def layout(self, size) -> None:
        w, h = size
        self.s = s = ui_scale(pygame.display.get_surface())
        self.top_h = int(130 * s)
        self.btn_menu.rect = pygame.Rect(int(20 * s), int(20 * s), int(190 * s), int(56 * s))
        bsz = int(70 * s)
        self.btn_plus.rect = pygame.Rect(w - bsz - int(24 * s), h - 2 * bsz - int(44 * s), bsz, bsz)
        self.btn_minus.rect = pygame.Rect(w - bsz - int(24 * s), h - bsz - int(24 * s), bsz, bsz)
        self.btn_next.rect = pygame.Rect(0, 0, int(240 * s), int(64 * s))
        self.counter_pos = (w - int(150 * s), int(48 * s))
        self.app.globe.set_rect(pygame.Rect(0, self.top_h, w, h - self.top_h))

    # --- Ввод ------------------------------------------------------------

    def handle(self, event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._to_menu()
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_KP_ENTER) and self.btn_next.visible:
                self._next()
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.app.globe.zoom_by(ZOOM_STEP)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.app.globe.zoom_by(1 / ZOOM_STEP)
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.ui_press = any(b.contains(event.pos) for b in self.buttons) or self._panel_contains(event.pos)
        if self.btn_menu.handle(event):
            self._to_menu()
            return
        if self.btn_plus.handle(event):
            self.app.globe.zoom_by(ZOOM_STEP)
            return
        if self.btn_minus.handle(event):
            self.app.globe.zoom_by(1 / ZOOM_STEP)
            return
        if self.btn_next.handle(event):
            self._next()
            return
        if self.ui_press and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
            if event.type == pygame.MOUSEBUTTONUP:
                self.ui_press = False
            return
        tap = self.app.globe.handle_event(event)
        if tap is not None and self.state in (ASK, MISS):
            self._answer(tap)

    def _panel_contains(self, pos) -> bool:
        return self.state in (CORRECT, REVEAL) and getattr(self, "panel_rect", pygame.Rect(0, 0, 0, 0)).collidepoint(pos)

    def _to_menu(self) -> None:
        from .menu import MenuScene

        self.app.switch(MenuScene(self.app))  # незавершённый проход не сохраняется

    # --- Ответы -----------------------------------------------------------

    def _answer(self, pos) -> None:
        app = self.app
        renderer = app.globe.renderer
        q = self.run.current
        tap = inspect_tap(renderer, app.regions, pos)
        if tap is None:
            return
        found = check_answer(renderer, app.regions, q, tap, pos)
        if found is not None:
            out = self.run.submit(True)
            self.state = CORRECT
            self.title, self.body = correct_message(found, q)
            app.globe.clear_effects()  # убираем подсветку-подсказку
            self._show(found, "found", FOUND_COLOR, fly=False)
            start = (pos[0] + app.globe.rect.left, pos[1] + app.globe.rect.top)
            for i in range(out.stars):
                self.flights.append(StarFlight(start, self.counter_pos, delay=0.15 * i, size=26 * self.s))
            self.btn_next.visible = True
            return
        out = self.run.submit(False)
        if out.result is Result.FAILED:
            self.state = REVEAL
            target, _ = nearest_target(app.regions, q, tap.latlon)
            self.title, self.body = reveal_message(target, q)
            app.globe.clear_effects()
            self._show(target, "reveal", REVEAL_COLOR, fly=True)
            self.btn_next.visible = True
            return
        self.state = MISS
        self.title = miss_message(app.regions, q, tap, riddle=self.level.riddles)
        tries = "Осталась одна попытка." if out.attempts_left == 1 else f"Осталось попыток: {out.attempts_left}."
        self.body = tries
        if out.show_hint:
            for rid in q.accept:
                self._show(app.regions[rid], "hint", HINT_COLOR, fly=False, soft=True, key_suffix=rid)

    def _show(self, region: Region, key: str, color, fly: bool, soft=False, key_suffix="") -> None:
        g = self.app.globe
        name = f"{key}:{key_suffix}" if key_suffix else key
        if region.kind in ("point", "peak", "line"):
            at = (region.lat, 0.0) if region.kind == "line" else region.at
            g.set_marker(name, Marker("line" if region.kind == "line" else "point", *at, color=color))
        else:
            low, high = (0.3, 0.65) if soft else (0.35, 0.8)
            g.set_glow(name, region.faces, color=color, low=low, high=high)
        if fly:
            lat, lon = region.center
            if region.kind == "line":
                lon = g.view.center_lon
            g.fly_to(lat, lon, zoom_for(region))

    def _next(self) -> None:
        if self.state not in (CORRECT, REVEAL) or self.app.globe.busy:
            return
        # Звёзды, которые ещё летят, засчитываем сразу.
        self.shown_stars = self.run.total_stars
        self.flights.clear()
        if self.run.is_last:
            self.run.next_question()
            from .result import ResultScene

            self.app.switch(ResultScene(self.app, self.level, self.run))
            return
        self.run.next_question()
        self.state = ASK
        self.title = self.body = ""
        self.btn_next.visible = False
        self.app.globe.clear_effects()

    # --- Время и рисование -----------------------------------------------

    def update(self, dt) -> None:
        self.app.globe.update(dt)
        for f in self.flights:
            if f.update(dt):
                self.shown_stars += 1
        self.flights = [f for f in self.flights if not f.arrived]

    def draw(self, surf) -> None:
        s = self.s
        w, h = surf.get_size()
        app = self.app
        app.starfield.draw(surf)
        app.globe.draw(surf)

        # Верхняя панель: вопрос, номер вопроса, звёзды.
        bar = pygame.Surface((w, self.top_h), pygame.SRCALPHA)
        bar.fill((10, 14, 40, 215))
        surf.blit(bar, (0, 0))
        q_width = w - int(2 * 260 * s)
        if not self.run.finished:
            draw_text(surf, self.run.current.text, int(44 * s), TEXT,
                      (w // 2, int(self.top_h * 0.45)), width=q_width)
            draw_text(surf, f"{self.level.title} · вопрос {self.run.index + 1} из {len(self.run.questions)}",
                      int(24 * s), TEXT_DIM, (w // 2, self.top_h - int(18 * s)))
        draw_star(surf, (self.counter_pos[0] - int(40 * s), self.counter_pos[1]), int(26 * s))
        draw_text(surf, f"{self.shown_stars}", int(48 * s), ACCENT,
                  (self.counter_pos[0] - int(5 * s), self.counter_pos[1]), anchor="midleft")
        for b in (self.btn_menu, self.btn_plus, self.btn_minus):
            b.draw(surf, text_size=int(30 * s) if b is self.btn_menu else int(50 * s))

        if self.state != ASK and self.title:
            self._draw_message(surf)
        for f in self.flights:
            f.draw(surf)

    def _draw_message(self, surf) -> None:
        s = self.s
        w, h = surf.get_size()
        width = min(int(900 * s), w - int(220 * s))
        pad = int(22 * s)
        title_size, body_size = int(36 * s), int(28 * s)
        inner = width - 2 * pad
        th = text_height(self.title, title_size, inner)
        bh = text_height(self.body, body_size, inner) if self.body else 0
        btn_h = self.btn_next.rect.height + pad if self.btn_next.visible else 0
        height = pad * 2 + th + (int(12 * s) + bh if bh else 0) + btn_h
        rect = pygame.Rect(0, 0, width, height)
        rect.midbottom = (w // 2, h - int(24 * s))
        edge = {MISS: BAD, CORRECT: GOOD, REVEAL: (255, 170, 80)}.get(self.state, ACCENT)
        draw_panel(surf, rect, color=PANEL, edge=edge, alpha=235)
        self.panel_rect = rect
        y = rect.top + pad
        r = draw_text(surf, cap(self.title), title_size, edge if self.state != MISS else TEXT,
                      (rect.centerx, y), anchor="midtop", width=inner)
        y = r.bottom + int(12 * s)
        if self.body:
            r = draw_text(surf, self.body, body_size, TEXT_DIM if self.state == MISS else TEXT,
                          (rect.centerx, y), anchor="midtop", width=inner)
            y = r.bottom
        if self.btn_next.visible:
            self.btn_next.rect.midbottom = (rect.centerx, rect.bottom - pad)
            self.btn_next.enabled = not self.app.globe.busy
            self.btn_next.text = "Итоги ›" if self.run.is_last else "Дальше ›"
            self.btn_next.draw(surf, text_size=int(32 * s))

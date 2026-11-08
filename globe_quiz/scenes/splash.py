"""Заставка: название, вращающийся глобус, автор модели и лицензия со ссылками."""

from __future__ import annotations

import math
import webbrowser

import pygame

from ..app import Scene
from ..ui import ACCENT, TEXT, TEXT_DIM, draw_text, font, ui_scale

LINK = (130, 190, 255)
LINK_HOVER = (190, 225, 255)


def set_cursor(hand: bool) -> None:
    try:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hand else pygame.SYSTEM_CURSOR_ARROW)
    except pygame.error:  # курсоры могут быть недоступны (например, без окна)
        pass


class SplashScene(Scene):
    def enter(self) -> None:
        self.t = 0.0
        self.links: list[tuple[pygame.Rect, str]] = []
        self._link_down: str | None = None
        g = self.app.globe
        g.view.center_lat, g.view.center_lon, g.view.zoom = 20.0, 20.0, 1.0
        g.target_zoom = 1.0
        g.auto_spin = 14.0
        c = self.app.credits
        # Строки об авторе: куски текста, у ссылок — адрес.
        self.credit_lines = [
            [("3D-модель «", None), (c["title"], c["source"]), ("», автор ", None),
             (c["author"], c["author_url"])],
            [("Лицензия ", None), (c["license"], c["license_url"])],
        ]

    def layout(self, size) -> None:
        w, h = size
        self.s = ui_scale(pygame.display.get_surface())
        top = int(170 * self.s)
        bottom = int(160 * self.s)
        self.app.globe.set_rect(pygame.Rect(0, top, w, max(100, h - top - bottom)))

    def _link_at(self, pos) -> str | None:
        return next((url for rect, url in self.links if rect.collidepoint(pos)), None)

    def handle(self, event) -> None:
        if event.type == pygame.MOUSEMOTION:
            set_cursor(self._link_at(event.pos) is not None)
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._link_down = self._link_at(event.pos)
            return
        if event.type == pygame.MOUSEBUTTONUP:
            url = self._link_at(event.pos)
            down, self._link_down = self._link_down, None
            if url is not None or down is not None:
                if url is not None and url == down:
                    webbrowser.open(url)  # ссылка открывается в браузере, игра остаётся здесь
                return
        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONUP):
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.app.running = False
                return
            set_cursor(False)
            self.app.switch(self.app.after_splash())

    def update(self, dt) -> None:
        self.t += dt
        self.app.globe.update(dt)

    def _draw_credit_line(self, surf, segments, center_y) -> None:
        fnt = font(int(26 * self.s))
        mouse = pygame.mouse.get_pos()
        images = [fnt.render(text, True, TEXT_DIM) for text, _ in segments]
        x = surf.get_width() // 2 - sum(im.get_width() for im in images) // 2
        for (text, url), im in zip(segments, images):
            rect = im.get_rect(midleft=(x, center_y))
            if url:
                hover = rect.collidepoint(mouse)
                im = fnt.render(text, True, LINK_HOVER if hover else LINK)
                pygame.draw.line(surf, LINK_HOVER if hover else LINK,
                                 (rect.left, rect.bottom - 2), (rect.right, rect.bottom - 2), 2)
                self.links.append((rect, url))
            surf.blit(im, rect)
            x += rect.width

    def draw(self, surf) -> None:
        s = self.s
        w, h = surf.get_size()
        self.app.starfield.draw(surf)
        self.app.globe.draw(surf)
        draw_text(surf, "Планета Земля", int(96 * s), ACCENT, (w // 2, int(70 * s)))
        draw_text(surf, "Викторина по географии", int(44 * s), TEXT, (w // 2, int(135 * s)))

        self.links = []
        y = h - int(135 * s)
        for segments in self.credit_lines:
            self._draw_credit_line(surf, segments, y)
            y += int(34 * s)
        alpha = 0.5 + 0.5 * math.sin(self.t * 3)
        col = tuple(int(TEXT_DIM[i] + (TEXT[i] - TEXT_DIM[i]) * alpha) for i in range(3))
        draw_text(surf, "Нажми любую клавишу или кнопку мыши", int(34 * s), col,
                  (w // 2, h - int(45 * s)))

"""Меню уровней на фоне вращающейся Земли: карточки, имя игрока, смена игрока."""

from __future__ import annotations

import pygame

from ..app import Scene
from ..quiz import MAX_QUESTIONS, previous_level
from ..ui import (ACCENT, BAD, PANEL, PANEL_EDGE, STAR_EMPTY, TEXT, TEXT_DIM, Button, draw_lock,
                  draw_panel, draw_star, draw_text, ui_scale)


def lock_place(card: pygame.Rect, title: pygame.Rect, s: float) -> tuple[tuple[int, int], int]:
    """Центр и высота замочка: посередине свободного места под названием карточки.

    Высота — 46·s, но в тесной карточке (маленькое окно) замок уменьшается,
    чтобы не наезжать на название и край.
    """
    free = card.bottom - title.bottom
    height = max(12, min(int(46 * s), free - int(16 * s)))
    return (card.centerx, title.bottom + free // 2), height


class MenuScene(Scene):
    def enter(self) -> None:
        self.cards: list[tuple[pygame.Rect, object]] = []
        self.message = ""
        self.message_t = 0.0
        self._pressed = None
        self.btn_player = Button("Сменить игрока")
        g = self.app.globe
        g.view.zoom = g.target_zoom = 1.0
        g.view.center_lat = 20.0
        g.auto_spin = 6.0

    def layout(self, size) -> None:
        w, h = size
        self.s = s = ui_scale(pygame.display.get_surface())
        levels = self.app.levels
        cols = 4 if w >= 1000 else 3  # 11 карточек: 4+4+3 или 3+3+3+2
        rows = (len(levels) + cols - 1) // cols
        gap = int(24 * s)
        top = int(150 * s)
        bottom = int(90 * s)
        card_w = min(int(330 * s), (w - gap * (cols + 1)) // cols)
        card_h = min(int(220 * s), (h - top - bottom - gap * (rows - 1)) // rows)
        self.cards = []
        for i, level in enumerate(levels):
            r, c = divmod(i, cols)
            in_row = min(cols, len(levels) - r * cols)
            row_w = in_row * card_w + (in_row - 1) * gap
            x0 = (w - row_w) // 2
            rect = pygame.Rect(x0 + c * (card_w + gap), top + r * (card_h + gap), card_w, card_h)
            self.cards.append((rect, level))
        globe_top = int(120 * s)  # глобус ниже заголовка, чтобы тот читался
        self.app.globe.set_rect(pygame.Rect(0, globe_top, w, h - globe_top))
        self.btn_player.rect = pygame.Rect(0, 0, int(270 * s), int(56 * s))
        self.btn_player.rect.topright = (w - int(20 * s), int(20 * s))

    def _card_at(self, pos):
        for rect, level in self.cards:
            if rect.collidepoint(pos):
                return level
        return None

    def handle(self, event) -> None:
        if self.btn_player.handle(event):
            from .players import PlayersScene

            self.app.switch(PlayersScene(self.app))
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._pressed = self._card_at(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            level = self._card_at(event.pos)
            if level is not None and level is self._pressed:
                self._open(level)
            self._pressed = None

    def _open(self, level) -> None:
        if not self.app.progress.is_unlocked(level):
            prev = previous_level(self.app.levels, level)
            self.message = f"Чтобы открыть этот уровень, пройди уровень «{prev.title}»"
            self.message_t = 3.0
            return
        from .level import LevelScene

        self.app.switch(LevelScene(self.app, level))

    def update(self, dt) -> None:
        self.message_t = max(0.0, self.message_t - dt)
        self.app.globe.update(dt)

    def draw(self, surf) -> None:
        s = self.s
        w, h = surf.get_size()
        self.app.starfield.draw(surf)
        self.app.globe.draw(surf)
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((8, 12, 38, 70))
        surf.blit(shade, (0, 0))
        draw_text(surf, "Выбери уровень", int(72 * s), ACCENT, (w // 2, int(75 * s)))
        name = self.app.players.current or ""
        draw_text(surf, f"Играет: {name}", int(34 * s), TEXT, (int(24 * s), int(48 * s)), anchor="midleft")
        self.btn_player.draw(surf, text_size=int(28 * s))
        progress = self.app.progress
        mouse = pygame.mouse.get_pos()
        for rect, level in self.cards:
            unlocked = progress.is_unlocked(level)
            hover = unlocked and rect.collidepoint(mouse)
            color = (40, 56, 130) if hover else PANEL
            edge = ACCENT if hover else PANEL_EDGE
            # Карточки полупрозрачные — за ними видно вращающуюся Землю.
            draw_panel(surf, rect, color=color, edge=edge, alpha=200 if unlocked else 140)
            title_col = TEXT if unlocked else TEXT_DIM
            draw_text(surf, f"{level.id}", int(46 * s), ACCENT if unlocked else TEXT_DIM,
                      (rect.left + int(30 * s), rect.top + int(32 * s)))
            title_rect = draw_text(surf, level.title, int(36 * s), title_col,
                                   (rect.centerx, rect.top + int(rect.height * 0.42)), width=rect.width - int(30 * s))
            if unlocked:
                best = progress.best_for(level.key)
                max_stars = 3 * min(MAX_QUESTIONS, len(level.questions))
                filled = 3 if best >= max_stars else (2 if best >= max_stars * 2 / 3 else (1 if best > 0 else 0))
                y = rect.bottom - int(48 * s)
                for k in range(3):
                    draw_star(surf, (rect.centerx + (k - 1) * int(42 * s), y), int(18 * s),
                              color=ACCENT if k < filled else STAR_EMPTY,
                              outline=(150, 100, 0) if k < filled else None)
                draw_text(surf, f"{best} из {max_stars}", int(24 * s), TEXT_DIM,
                          (rect.centerx, rect.bottom - int(18 * s)))
            else:
                draw_lock(surf, *lock_place(rect, title_rect, s))
        if self.message_t > 0:
            draw_text(surf, self.message, int(32 * s), BAD, (w // 2, h - int(45 * s)), width=w - 40)
        else:
            draw_text(surf, "Esc — выход", int(22 * s), TEXT_DIM, (w // 2, h - int(30 * s)))

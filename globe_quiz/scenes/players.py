"""Экран игроков: выбор игрока и создание нового с вводом имени."""

from __future__ import annotations

import math

import pygame

from ..app import Scene
from ..progress import MAX_NAME_LENGTH
from ..ui import (ACCENT, BAD, BUTTON, PANEL, PANEL_EDGE, TEXT, TEXT_DIM, Button, draw_panel, draw_text,
                  font, ui_scale)

NAME_CHARS_EXTRA = set(" -")


def allowed(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit() or ch in NAME_CHARS_EXTRA


class PlayersScene(Scene):
    def enter(self) -> None:
        self.entering = not self.app.players.names  # игроков нет — сразу ввод имени
        self.name = ""
        self.error = ""
        self.t = 0.0
        self.player_buttons: list[Button] = []
        self.btn_new = Button("Новый игрок")
        self.btn_ok = Button("Готово")
        self.btn_cancel = Button("Отмена", color=(90, 96, 140), hover_color=(110, 118, 170))
        self.app.globe.auto_spin = 8.0
        if self.entering:
            pygame.key.start_text_input()

    @property
    def title(self) -> str:
        if self.entering:
            return "Как тебя зовут?" if not self.app.players.names else "Новый игрок"
        return "Кто играет?"

    def layout(self, size) -> None:
        w, h = size
        self.s = s = ui_scale(pygame.display.get_surface())
        self.app.globe.set_rect(pygame.Rect(0, 0, w, h))
        names = self.app.players.names
        bw, bh, gap = int(320 * s), int(80 * s), int(20 * s)
        cols = max(1, min(3, (w - gap) // (bw + gap)))
        top = int(170 * s)
        self.player_buttons = []
        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            in_row = min(cols, len(names) - r * cols)
            x0 = (w - (in_row * bw + (in_row - 1) * gap)) // 2
            current = name == self.app.players.current
            btn = Button(name, color=(70, 110, 220) if current else BUTTON)
            btn.rect = pygame.Rect(x0 + c * (bw + gap), top + r * (bh + gap), bw, bh)
            self.player_buttons.append(btn)
        rows = (len(names) + cols - 1) // cols
        y_new = top + rows * (bh + gap) + int(20 * s)
        self.btn_new.rect = pygame.Rect(0, 0, int(320 * s), int(70 * s))
        self.btn_new.rect.midtop = (w // 2, min(y_new, h - int(100 * s)))
        # Поле ввода имени и кнопки под ним.
        self.field = pygame.Rect(0, 0, min(int(640 * s), w - 40), int(90 * s))
        self.field.center = (w // 2, int(h * 0.45))
        bw2 = int(220 * s)
        self.btn_ok.rect = pygame.Rect(0, 0, bw2, int(70 * s))
        self.btn_cancel.rect = pygame.Rect(0, 0, bw2, int(70 * s))
        if self.app.players.names:
            self.btn_ok.rect.topright = (w // 2 - int(15 * s), self.field.bottom + int(40 * s))
            self.btn_cancel.rect.topleft = (w // 2 + int(15 * s), self.field.bottom + int(40 * s))
        else:
            self.btn_ok.rect.midtop = (w // 2, self.field.bottom + int(40 * s))
        pygame.key.set_text_input_rect(self.field)

    # --- Ввод ------------------------------------------------------------

    def handle(self, event) -> None:
        if self.entering:
            self._handle_entering(event)
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._to_menu()
            return
        for btn in self.player_buttons:
            if btn.handle(event):
                self.app.players.select(btn.text)
                self.app.save_progress()
                self._to_menu()
                return
        if self.btn_new.handle(event):
            self.entering = True
            self.name = ""
            self.error = ""
            pygame.key.start_text_input()

    def _handle_entering(self, event) -> None:
        if event.type == pygame.TEXTINPUT:
            for ch in event.text:
                if allowed(ch) and len(self.name) < MAX_NAME_LENGTH:
                    self.name += ch
            self.error = ""
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.name = self.name[:-1]
                self.error = ""
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._create()
            elif event.key == pygame.K_ESCAPE:
                self._cancel()
        elif self.btn_ok.handle(event):
            self._create()
        elif self.app.players.names and self.btn_cancel.handle(event):
            self._cancel()

    def _create(self) -> None:
        try:
            self.app.players.add(self.name)
        except ValueError as e:
            self.error = str(e)
            return
        self.app.save_progress()
        pygame.key.stop_text_input()
        self._to_menu()

    def _cancel(self) -> None:
        if not self.app.players.names:
            return  # без игрока дальше нельзя
        self.entering = False
        self.error = ""
        pygame.key.stop_text_input()

    def _to_menu(self) -> None:
        if self.app.players.current is None:
            return
        pygame.key.stop_text_input()
        from .menu import MenuScene

        self.app.switch(MenuScene(self.app))

    # --- Рисование ---------------------------------------------------------

    def update(self, dt) -> None:
        self.t += dt
        self.app.globe.update(dt)

    def draw(self, surf) -> None:
        s = self.s
        w, h = surf.get_size()
        self.app.starfield.draw(surf)
        self.app.globe.draw(surf)
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((8, 12, 38, 120))
        surf.blit(shade, (0, 0))
        draw_text(surf, self.title, int(72 * s), ACCENT, (w // 2, int(85 * s)))
        if self.entering:
            draw_panel(surf, self.field, color=PANEL, edge=ACCENT, alpha=245)
            fnt = font(int(54 * s))
            img = fnt.render(self.name, True, TEXT)
            r = img.get_rect(midleft=(self.field.left + int(24 * s), self.field.centery))
            surf.blit(img, r)
            if math.sin(self.t * 6) > 0:
                x = r.right + 4 if self.name else r.left
                pygame.draw.line(surf, TEXT, (x, self.field.top + int(18 * s)),
                                 (x, self.field.bottom - int(18 * s)), 3)
            hint = self.error or "Напиши имя на клавиатуре и нажми Enter"
            draw_text(surf, hint, int(30 * s), BAD if self.error else TEXT_DIM,
                      (w // 2, self.field.top - int(35 * s)))
            self.btn_ok.draw(surf, text_size=int(34 * s))
            if self.app.players.names:
                self.btn_cancel.draw(surf, text_size=int(34 * s))
        else:
            for btn in self.player_buttons:
                btn.draw(surf, text_size=int(38 * s))
            self.btn_new.draw(surf, text_size=int(34 * s))
            draw_text(surf, "Esc — назад", int(22 * s), TEXT_DIM, (w // 2, h - int(30 * s)))

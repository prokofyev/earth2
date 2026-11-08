"""Итог уровня: звёзды, салют при прохождении, «Ещё раз» и «В меню»."""

from __future__ import annotations

import pygame

from ..app import Scene
from ..fx import Fireworks
from ..quiz import PASS_RATIO
from ..ui import ACCENT, GOOD, TEXT, TEXT_DIM, Button, draw_star, draw_text, ui_scale


class ResultScene(Scene):
    def __init__(self, app, level, run):
        super().__init__(app)
        self.level = level
        self.stars = run.total_stars
        self.max_stars = run.max_stars
        self.passed = run.passed
        level_count = len(app.levels)
        was_unlocked = app.progress.unlocked
        app.progress.record(level.id, self.stars, self.passed, level_count)
        app.save_progress()
        self.unlocked_new = app.progress.unlocked > was_unlocked

    @property
    def next_level(self):
        """Следующий уровень, если этот пройден и он не последний."""
        if self.passed and self.level.id < len(self.app.levels):
            return self.app.levels[self.level.id]
        return None

    def enter(self) -> None:
        self.btn_main = Button("Следующий уровень" if self.next_level else "Ещё раз")
        self.btn_menu = Button("В меню")
        self.fireworks: Fireworks | None = None

    def layout(self, size) -> None:
        w, h = size
        self.s = s = ui_scale(pygame.display.get_surface())
        bh = int(72 * s)
        main_w, menu_w = int(360 * s), int(240 * s)
        gap = int(40 * s)
        left = w // 2 - (main_w + gap + menu_w) // 2
        self.btn_main.rect = pygame.Rect(left, int(h * 0.72), main_w, bh)
        self.btn_menu.rect = pygame.Rect(left + main_w + gap, int(h * 0.72), menu_w, bh)
        if self.passed:
            self.fireworks = Fireworks(pygame.Rect(0, 0, w, h))
        self.app.globe.set_rect(pygame.Rect(0, 0, 1, 1))

    def handle(self, event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._menu()
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._main()
        elif self.btn_main.handle(event):
            self._main()
        elif self.btn_menu.handle(event):
            self._menu()

    def _main(self) -> None:
        from .level import LevelScene

        self.app.switch(LevelScene(self.app, self.next_level or self.level))

    def _menu(self) -> None:
        from .menu import MenuScene

        self.app.switch(MenuScene(self.app))

    def update(self, dt) -> None:
        if self.fireworks is not None:
            self.fireworks.update(dt)

    def draw(self, surf) -> None:
        s = self.s
        w, h = surf.get_size()
        self.app.starfield.draw(surf)
        if self.fireworks is not None:
            self.fireworks.draw(surf)
        if self.passed:
            title = f"Уровень «{self.level.title}» пройден!"
            sub = "Ура! Открыт следующий уровень!" if self.unlocked_new else "Отличная работа!"
            color = GOOD
        else:
            need = int(-(-self.max_stars * PASS_RATIO // 1))
            title = "Почти получилось!"
            sub = f"Чтобы пройти уровень, нужно {need} звёзд. Попробуй ещё раз — у тебя получится!"
            color = ACCENT
        draw_text(surf, title, int(64 * s), color, (w // 2, int(h * 0.2)), width=w - 80)
        draw_star(surf, (w // 2 - int(110 * s), int(h * 0.42)), int(60 * s))
        draw_text(surf, f"{self.stars} из {self.max_stars}", int(80 * s), TEXT,
                  (w // 2 - int(30 * s), int(h * 0.42)), anchor="midleft")
        draw_text(surf, sub, int(34 * s), TEXT_DIM, (w // 2, int(h * 0.57)), width=w - 120)
        self.btn_main.draw(surf, text_size=int(34 * s))
        self.btn_menu.draw(surf, text_size=int(34 * s))

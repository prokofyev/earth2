"""Окно приложения, общий глобус и смена сцен."""

from __future__ import annotations

import json
import sys

import pygame

from . import DATA_DIR
from .globe_widget import GlobeWidget
from .model import load_earth
from .progress import Players, Progress, load_players, save_players
from .quiz import load_levels
from .regions import RegionSet
from .renderer import GlobeRenderer
from .ui import TEXT, Starfield, draw_text

TITLE = "Планета Земля — викторина"
FPS = 60


def open_window() -> pygame.Surface:
    """Обычное окно, развёрнутое на весь экран (не полноэкранный режим)."""
    desk_w, desk_h = pygame.display.get_desktop_sizes()[0]
    screen = pygame.display.set_mode((min(1280, desk_w), min(800, desk_h)), pygame.RESIZABLE)
    try:
        from pygame._sdl2.video import Window

        Window.from_display_module().maximize()
    except Exception:  # приватный API может пропасть — тогда просто большое окно
        screen = pygame.display.set_mode((desk_w, max(480, desk_h - 80)), pygame.RESIZABLE)
    return screen


def model_credits() -> dict:
    """Сведения об авторе модели Земли (globe_quiz/data/model.json)."""
    data = json.loads((DATA_DIR / "model.json").read_text(encoding="utf-8"))
    keys = ("title", "author", "author_url", "source", "license", "license_url", "changes")
    return {k: data.get(k, "") for k in keys}


class Scene:
    def __init__(self, app: "App"):
        self.app = app

    def enter(self) -> None: ...

    def layout(self, size: tuple[int, int]) -> None: ...

    def handle(self, event: pygame.event.Event) -> None: ...

    def update(self, dt: float) -> None: ...

    def draw(self, surf: pygame.Surface) -> None: ...


class App:
    def __init__(self, debug_regions: bool = False):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = open_window()
        self.starfield = Starfield()
        self._loading_screen()

        self.model = load_earth()
        self.regions = RegionSet.load().assign(self.model)
        self.levels = load_levels()
        self.players: Players = load_players()
        self.credits = model_credits()
        self.globe = GlobeWidget(GlobeRenderer(self.model))
        self.starfield.view = self.globe.view  # небо вращается вместе с глобусом
        self.clock = pygame.time.Clock()
        self.running = True
        self._size = None

        from .scenes.debug import DebugScene
        from .scenes.splash import SplashScene

        self.scene: Scene = DebugScene(self) if debug_regions else SplashScene(self)
        self.scene.enter()

    def _loading_screen(self) -> None:
        surf = pygame.display.get_surface()
        self.starfield.draw(surf)
        draw_text(surf, "Загружаем глобус…", 48, TEXT, surf.get_rect().center)
        pygame.display.flip()

    def switch(self, scene: Scene) -> None:
        self.scene = scene
        self.globe.clear_effects()
        self.globe.auto_spin = 0.0
        self.scene.enter()
        self.scene.layout(pygame.display.get_surface().get_size())

    @property
    def progress(self) -> Progress:
        """Прогресс текущего игрока (пустой, пока игрок не выбран)."""
        return self.players.progress or Progress()

    def after_splash(self) -> Scene:
        """Куда идти после заставки: в меню или, если игроков нет, к выбору игрока."""
        from .scenes.menu import MenuScene
        from .scenes.players import PlayersScene

        return MenuScene(self) if self.players.current else PlayersScene(self)

    def save_progress(self) -> None:
        try:
            save_players(self.players)
        except OSError as e:  # не мешаем игре, если диск недоступен
            print(f"Не удалось сохранить прогресс: {e}", file=sys.stderr)

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.1)
            surf = pygame.display.get_surface()
            if surf.get_size() != self._size:
                self._size = surf.get_size()
                self.scene.layout(self._size)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    self.scene.handle(event)
            self.scene.update(dt)
            self.scene.draw(surf)
            pygame.display.flip()
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    App(debug_regions="--debug-regions" in argv).run()

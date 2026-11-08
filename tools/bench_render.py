"""Замер скорости отрисовки глобуса: крутит глобус и печатает FPS.

Запуск: uv run python tools/bench_render.py [секунды] [ширина высота]
"""

import sys
import time

import pygame

from globe_quiz.model import load_earth
from globe_quiz.renderer import GlobeRenderer


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    pygame.init()
    if len(sys.argv) > 3:
        size = (int(sys.argv[2]), int(sys.argv[3]))
    else:
        w, h = pygame.display.get_desktop_sizes()[0]
        size = (w, h - 80)
    screen = pygame.display.set_mode(size)
    model = load_earth()
    renderer = GlobeRenderer(model)
    clock = pygame.time.Clock()
    frames = 0
    draw_time = 0.0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        renderer.view.center_lon += 2.0
        renderer.view.clamp()
        t = time.perf_counter()
        screen.fill((10, 14, 40))
        renderer.draw(screen, screen.get_rect())
        draw_time += time.perf_counter() - t
        pygame.display.flip()
        frames += 1
        clock.tick()
    total = time.perf_counter() - start
    print(f"window {size}, faces drawn last frame: {len(renderer.frame.order)}")
    print(f"FPS: {frames / total:.1f}, draw only: {frames / draw_time:.1f}")


if __name__ == "__main__":
    main()

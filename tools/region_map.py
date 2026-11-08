"""Плоская карта модели с сеткой и разметкой регионов — для правки regions.json.

Запуск:
  uv run python -m tools.region_map out.png                    # вся Земля, цвета модели
  uv run python -m tools.region_map out.png --regions          # раскраска базовыми регионами
  uv run python -m tools.region_map out.png --only madagascar  # подсветить один регион
  uv run python -m tools.region_map out.png --box 30 -15 75 50 --px 20
      (--box lat_min lon_min lat_max lon_max, --px пикселей на градус)
"""

from __future__ import annotations

import argparse
import colorsys
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from globe_quiz.geo import vec_to_latlon  # noqa: E402
from globe_quiz.model import load_earth  # noqa: E402
from globe_quiz.regions import RegionSet  # noqa: E402


def _region_color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--box", nargs=4, type=float, default=[-90, -180, 90, 180])
    ap.add_argument("--px", type=float, default=5.0)
    ap.add_argument("--regions", action="store_true", help="раскрасить базовыми регионами")
    ap.add_argument("--only", nargs="*", default=[], help="подсветить эти регионы")
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1))
    font = pygame.font.Font(None, 18)
    model = load_earth()
    regions = RegionSet.load().assign(model)

    lat0, lon0, lat1, lon1 = args.box
    w, h = int((lon1 - lon0) * args.px), int((lat1 - lat0) * args.px)
    surf = pygame.Surface((w, h))
    surf.fill((0, 0, 0))

    colors = model.colors.astype(int).copy()
    if args.regions:
        colors[:] = (40, 40, 40)
        for i, r in enumerate(regions.regions):
            if r.is_base and r.faces is not None:
                colors[r.faces] = _region_color(i)
        colors[regions.unassigned()] = (255, 0, 255)
    for rid in args.only:
        r = regions[rid]
        colors[r.faces] = (colors[r.faces] * 0.3 + np.array([255, 0, 0]) * 0.7).astype(int)

    radius = np.linalg.norm(model.centers, axis=1)
    order = np.argsort(radius)  # сверху — самые внешние грани
    ll = vec_to_latlon(model.tris)  # (F, 3, 2)
    for f in order:
        lat, lon = ll[f, :, 0], ll[f, :, 1].copy()
        if lon.max() - lon.min() > 180:  # грань через линию перемены дат
            lon[lon < 0] += 360
        for shift in (-360, 0, 360):
            xs = (lon + shift - lon0) * args.px
            ys = (lat1 - lat) * args.px
            if xs.max() < 0 or xs.min() > w:
                continue
            pygame.draw.polygon(surf, [int(c) for c in colors[f]], list(zip(xs, ys)))

    grid = 10 if args.px < 12 else 5
    for lat in np.arange(np.ceil(lat0 / grid) * grid, lat1 + 0.1, grid):
        y = (lat1 - lat) * args.px
        pygame.draw.line(surf, (60, 60, 60), (0, y), (w, y))
        surf.blit(font.render(f"{lat:g}", True, (255, 255, 255)), (2, y + 1))
    for lon in np.arange(np.ceil(lon0 / grid) * grid, lon1 + 0.1, grid):
        x = (lon - lon0) * args.px
        pygame.draw.line(surf, (60, 60, 60), (x, 0), (x, h))
        surf.blit(font.render(f"{lon:g}", True, (255, 255, 255)), (x + 2, 2))

    if args.regions:
        for i, r in enumerate(regions.regions):
            if r.is_base and r.faces is not None and r.faces.any():
                lat, lon = r.center
                x, y = (lon - lon0) * args.px, (lat1 - lat) * args.px
                surf.blit(font.render(r.id, True, (0, 0, 0), (255, 255, 255)), (x, y))
        print("неразмеченных граней:", int(regions.unassigned().sum()))
    pygame.image.save(surf, args.out)


if __name__ == "__main__":
    main()

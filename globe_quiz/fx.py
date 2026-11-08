"""Эффекты: звёзды, летящие в счётчик, и салют."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pygame

from .ui import draw_star

FIREWORK_COLORS = [
    (255, 90, 90), (255, 200, 60), (120, 230, 120), (90, 180, 255),
    (220, 120, 255), (255, 150, 220), (255, 255, 255),
]


@dataclass
class StarFlight:
    """Звезда летит по дуге от start к end. Не мешает вводу."""

    start: tuple[float, float]
    end: tuple[float, float]
    delay: float = 0.0
    duration: float = 0.9
    size: float = 26.0
    t: float = 0.0
    arrived: bool = False

    def update(self, dt: float) -> bool:
        """Возвращает True в тот момент, когда звезда долетела."""
        self.t += dt
        if not self.arrived and self.t >= self.delay + self.duration:
            self.arrived = True
            return True
        return False

    def draw(self, surf) -> None:
        if self.arrived or self.t < self.delay:
            return
        k = min(1.0, (self.t - self.delay) / self.duration)
        e = 1 - (1 - k) ** 3
        (x0, y0), (x1, y1) = self.start, self.end
        cx, cy = (x0 + x1) / 2, min(y0, y1) - 160  # контрольная точка — дуга вверх
        x = (1 - e) ** 2 * x0 + 2 * (1 - e) * e * cx + e**2 * x1
        y = (1 - e) ** 2 * y0 + 2 * (1 - e) * e * cy + e**2 * y1
        scale = 1.4 - 0.6 * e
        draw_star(surf, (x, y), self.size * scale)


@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: tuple
    life: float
    max_life: float


@dataclass
class _Rocket:
    x: float
    y: float
    vy: float
    target_y: float
    color: tuple


@dataclass
class Fireworks:
    """Салют: ракеты взлетают и разлетаются искрами."""

    area: pygame.Rect
    duration: float = 5.0
    rng: random.Random = field(default_factory=random.Random)
    t: float = 0.0
    _next_launch: float = 0.0
    rockets: list = field(default_factory=list)
    particles: list = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.t > self.duration and not self.rockets and not self.particles

    def update(self, dt: float) -> None:
        self.t += dt
        a = self.area
        if self.t < self.duration and self.t >= self._next_launch:
            self._next_launch = self.t + self.rng.uniform(0.25, 0.55)
            x = self.rng.uniform(a.left + a.width * 0.1, a.right - a.width * 0.1)
            self.rockets.append(_Rocket(x, a.bottom, -a.height * self.rng.uniform(0.9, 1.2),
                                        self.rng.uniform(a.top + a.height * 0.1, a.top + a.height * 0.45),
                                        self.rng.choice(FIREWORK_COLORS)))
        g = a.height * 0.35
        for r in self.rockets[:]:
            r.y += r.vy * dt
            if r.y <= r.target_y:
                self.rockets.remove(r)
                self._explode(r)
        for p in self.particles[:]:
            p.life -= dt
            if p.life <= 0:
                self.particles.remove(p)
                continue
            p.vy += g * dt
            p.vx *= 0.985
            p.x += p.vx * dt
            p.y += p.vy * dt

    def _explode(self, r: _Rocket) -> None:
        n = self.rng.randint(60, 100)
        speed = self.area.height * self.rng.uniform(0.25, 0.4)
        second = self.rng.choice(FIREWORK_COLORS)
        for i in range(n):
            ang = 2 * math.pi * i / n + self.rng.uniform(-0.05, 0.05)
            s = speed * self.rng.uniform(0.6, 1.0)
            life = self.rng.uniform(1.0, 1.8)
            color = r.color if i % 3 else second
            self.particles.append(_Particle(r.x, r.y, math.cos(ang) * s, math.sin(ang) * s,
                                            color, life, life))

    def draw(self, surf) -> None:
        for r in self.rockets:
            pygame.draw.circle(surf, (255, 240, 200), (int(r.x), int(r.y)), 4)
            pygame.draw.line(surf, (255, 180, 80), (r.x, r.y), (r.x, r.y + 18), 2)
        for p in self.particles:
            k = p.life / p.max_life
            c = tuple(int(v * (0.35 + 0.65 * k)) for v in p.color)
            pygame.draw.circle(surf, c, (int(p.x), int(p.y)), max(1, int(3.5 * k + 1)))

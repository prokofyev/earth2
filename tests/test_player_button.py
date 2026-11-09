"""Кнопка игрока: имя, звезда и сумма звёзд."""

import pygame

from globe_quiz.ui import BUTTON, Button, font


def drawn_columns(btn, size):
    """Столбцы, где на кнопке нарисовано что-то, кроме её фона (надпись и звезда)."""
    surf = pygame.Surface((btn.rect.right + 40, btn.rect.bottom + 40))
    surf.fill((0, 0, 0))
    btn.draw(surf, text_size=size)
    y0, y1 = btn.rect.centery - 12, btn.rect.centery + 12  # средняя полоса: скругления рамки не мешают
    cols = []
    for x in range(btn.rect.left + 3, btn.rect.right - 3):
        if any(surf.get_at((x, y))[:3] != BUTTON for y in range(y0, y1)):
            cols.append(x)
    return cols


def test_plain_button_unchanged():
    btn = Button("Маша", rect=pygame.Rect(20, 20, 320, 80))
    assert btn.content_width(38) == font(38).size("Маша")[0]
    assert btn.fit_size(38) == 38


def test_stars_widen_content():
    plain = Button("Маша")
    with_stars = Button("Маша", stars=45)
    assert with_stars.content_width(38) > plain.content_width(38) + font(38).size("45")[0]


def test_long_name_with_stars_fits_button():
    btn = Button("Очень-очень-длин", rect=pygame.Rect(20, 20, 320, 80), stars=120)
    assert btn.content_width(38) > 320  # без подгонки не влезло бы
    size = btn.fit_size(38)
    assert size < 38 and btn.content_width(size) <= 320 - int(38 * 0.65)
    cols = drawn_columns(btn, 38)
    assert cols and cols[0] > btn.rect.left + 8 and cols[-1] < btn.rect.right - 8  # поля по краям


def test_star_is_drawn():
    btn = Button("Оля", rect=pygame.Rect(20, 20, 320, 80), stars=0)
    surf = pygame.Surface((400, 140))
    btn.draw(surf, text_size=38)
    yellow = [(x, y) for x in range(20, 340) for y in range(20, 100)
              if (lambda c: c.r > 200 and c.g > 150 and c.b < 120)(surf.get_at((x, y)))]
    assert len(yellow) > 30  # жёлтая звезда рядом с именем

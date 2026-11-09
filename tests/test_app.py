"""Сквозная проверка сцен без окна: заставка → меню → уровень → итог."""

import numpy as np
import pygame
import pytest

import globe_quiz.progress as progress_mod
from globe_quiz.app import App, model_credits
from globe_quiz.scenes.level import CORRECT, LevelScene, MISS
from globe_quiz.scenes.menu import MenuScene
from globe_quiz.scenes.result import ResultScene
from globe_quiz.scenes.splash import SplashScene
from globe_quiz.ui import font


@pytest.fixture
def fresh_app(tmp_path, monkeypatch):
    """Первый запуск: игроков ещё нет."""
    monkeypatch.setattr(progress_mod, "PROGRESS_PATH", tmp_path / "progress.json")
    a = App()
    a._size = None
    yield a
    pygame.key.stop_text_input()


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Запуск с уже созданным игроком «Тест»."""
    monkeypatch.setattr(progress_mod, "PROGRESS_PATH", tmp_path / "progress.json")
    players = progress_mod.Players()
    players.add("Тест")
    progress_mod.save_players(players)
    a = App()
    a._size = None
    yield a


def current_progress():
    return progress_mod.load_players().progress


def frame(app, events=(), dt=1 / 60):
    surf = pygame.display.get_surface()
    if surf.get_size() != app._size:
        app._size = surf.get_size()
        app.scene.layout(app._size)
    for e in events:
        app.scene.handle(e)
    app.scene.update(dt)
    app.scene.draw(surf)


def click(app, pos):
    frame(app, [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos),
                pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos)])


def target_screen_pos(app, region_id):
    g = app.globe
    region = app.regions[region_id]
    lat, lon = region.center
    small = region.faces is not None and region.radius_deg < 12
    g.fly_to(lat, lon, 2.5 if small else 1.0)
    for _ in range(150):
        g.update(0.05)
    frame(app)
    if region.kind in ("point", "peak"):
        (x, y), _ = g.renderer.project_latlon(*region.at)
        return int(x + g.rect.left), int(y + g.rect.top)
    if region.kind == "line":
        (x, y), _ = g.renderer.project_latlon(region.lat, g.view.center_lon)
        return int(x + g.rect.left), int(y + g.rect.top)
    fr = g.renderer.frame
    for k in np.nonzero(region.faces[fr.order])[0][::-1]:
        c = fr.pts[k].mean(axis=0)
        if g.renderer.pick_face(c) == fr.order[k]:
            return int(c[0] + g.rect.left), int(c[1] + g.rect.top)
    raise AssertionError(region_id)


def _missing_glyphs(text, f, missing):
    bad = set()
    for ch in set(text) - {" ", "\n"}:
        img = f.render(ch, True, (255, 255, 255))
        if img.get_width() == 0 or pygame.image.tobytes(img, "RGBA") == missing:
            bad.add(ch)
    return bad


def test_all_game_texts_have_glyphs():
    """Кириллица и все символы из вопросов, фактов и интерфейса есть в шрифте."""
    from globe_quiz.quiz import load_levels
    from globe_quiz.regions import RegionSet

    f = font(40)
    missing = pygame.image.tobytes(f.render("\ue000", True, (255, 255, 255)), "RGBA")
    assert _missing_glyphs("Ёж Щука ЭЮЯ ёъ", f, missing) == set()
    texts = ["‹ В меню", "Дальше ›", "Итоги ›", "+", "−", "«»—·…"]
    for level in load_levels():
        texts.append(level.title)
        texts += [q.text + (q.fact or "") for q in level.questions]
    texts += [r.name + r.fact for r in RegionSet.load().regions]
    texts += list(model_credits().values())
    bad = set()
    for t in texts:
        bad |= _missing_glyphs(t, f, missing)
    assert bad == set()


def test_credits_from_model():
    c = model_credits()
    assert c["title"] == "The Whole World"
    assert c["author"] == "Davision3d"
    assert c["author_url"] == "https://www.thingiverse.com/Davision3d/designs"
    assert c["license"] == "CC BY-NC 4.0"
    assert c["license_url"] == "https://creativecommons.org/licenses/by-nc/4.0/"
    assert c["source"] == "https://www.thingiverse.com/thing:671647"
    assert c["changes"] == "Модель упрощена и раскрашена для игры"


def test_full_flow_level_one(app, tmp_path):
    assert isinstance(app.scene, SplashScene)
    frame(app)
    click(app, (10, 10))
    assert isinstance(app.scene, MenuScene)
    frame(app)

    # Закрытый уровень не запускается.
    rect2, _ = app.scene.cards[1]
    click(app, rect2.center)
    assert isinstance(app.scene, MenuScene) and "Материки" in app.scene.message

    rect1, _ = app.scene.cards[0]
    click(app, rect1.center)
    scene = app.scene
    assert isinstance(scene, LevelScene)
    frame(app)

    # Первый вопрос: сначала промах, потом правильный ответ.
    first = scene.run.current
    wrong = next(r for r in ("antarctica", "australia") if r not in first.accept)
    click(app, target_screen_pos(app, wrong))
    assert scene.state == MISS and scene.title.startswith("Это ")
    while True:
        q = scene.run.current
        click(app, target_screen_pos(app, q.accept[0]))
        assert scene.state == CORRECT, q.text
        for _ in range(90):
            frame(app)
        last = scene.run.is_last
        frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
        if last:
            break
    assert isinstance(app.scene, ResultScene)
    result = app.scene
    assert result.stars == 17 and result.max_stars == 18 and result.passed
    assert result.fireworks is not None
    for _ in range(120):
        frame(app)
    saved = current_progress()
    assert saved.best_for(1) == 17 and saved.is_unlocked(2)


def test_escape_from_level_keeps_progress(app):
    app.switch(LevelScene(app, app.levels[0]))
    frame(app)
    frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
    assert isinstance(app.scene, MenuScene)
    assert current_progress().best == {}


def test_resize_relayouts(app):
    app.switch(LevelScene(app, app.levels[0]))
    frame(app)
    pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    frame(app)
    assert app.globe.rect.width == 640 and app.globe.rect.bottom == 480
    assert app.scene.btn_minus.rect.bottom <= 480


def test_all_eight_levels_in_a_row_from_fresh_start(app):
    """С чистого прогресса проходим все уровни подряд; каждый открывает следующий."""
    for level in app.levels:
        app.switch(MenuScene(app))
        frame(app)
        rect, card_level = app.scene.cards[level.id - 1]
        assert card_level is level and app.progress.is_unlocked(level.id)
        click(app, rect.center)
        scene = app.scene
        assert isinstance(scene, LevelScene) and scene.level is level
        frame(app)
        while True:
            q = scene.run.current
            click(app, target_screen_pos(app, q.accept[0]))
            assert scene.state == CORRECT, (level.title, q.text)
            last = scene.run.is_last
            frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
            if last:
                break
        assert isinstance(app.scene, ResultScene) and app.scene.passed
    saved = current_progress()
    assert saved.unlocked == 8
    assert all(saved.best_for(lv.id) == 3 * min(8, len(lv.questions)) for lv in app.levels)


def _finish_level(app, level, perfect=True):
    app.switch(LevelScene(app, level))
    scene = app.scene
    frame(app)
    while True:
        q = scene.run.current
        if perfect:
            scene.run.submit(True)
        else:
            for _ in range(3):
                scene.run.submit(False)
        scene.state = CORRECT
        scene.btn_next.visible = True
        last = scene.run.is_last
        frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
        if last:
            return app.scene


def test_result_next_level_button(app):
    result = _finish_level(app, app.levels[1])
    assert result.passed and result.btn_main.text == "Следующий уровень"
    frame(app)
    click(app, result.btn_main.rect.center)
    assert isinstance(app.scene, LevelScene) and app.scene.level is app.levels[2]


def test_result_failed_offers_retry(app):
    result = _finish_level(app, app.levels[0], perfect=False)
    assert not result.passed and result.btn_main.text == "Ещё раз"
    frame(app)
    click(app, result.btn_main.rect.center)
    assert isinstance(app.scene, LevelScene) and app.scene.level is app.levels[0]


def test_result_last_level_offers_retry(app):
    result = _finish_level(app, app.levels[7])
    assert result.passed and result.btn_main.text == "Ещё раз"


def test_after_peaks_next_is_riddles(app):
    result = _finish_level(app, app.levels[6])
    assert result.passed and result.btn_main.text == "Следующий уровень"
    click(app, result.btn_main.rect.center)
    assert isinstance(app.scene, LevelScene) and app.scene.level.title == "Загадки"


def type_text(app, text):
    frame(app, [pygame.event.Event(pygame.TEXTINPUT, text=ch) for ch in text])


def press(app, key):
    frame(app, [pygame.event.Event(pygame.KEYDOWN, key=key)])


def test_old_progress_file_starts_clean(tmp_path, monkeypatch):
    """Прогресс прежней версии (семь уровней) не переносится: игра спрашивает имя."""
    import json

    from globe_quiz.scenes.players import PlayersScene

    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"version": 2, "current": "Саша", "players": {
        "Саша": {"best": {str(i): 20 for i in range(1, 8)}, "unlocked": 7}}}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(progress_mod, "PROGRESS_PATH", path)
    app = App()
    app._size = None
    frame(app)
    click(app, (10, 10))
    assert isinstance(app.scene, PlayersScene) and app.scene.title == "Как тебя зовут?"
    type_text(app, "Саша")
    press(app, pygame.K_RETURN)
    assert isinstance(app.scene, MenuScene)
    assert app.progress.unlocked == 1 and app.progress.best == {}
    pygame.key.stop_text_input()


def test_first_run_asks_for_name(fresh_app):
    from globe_quiz.scenes.players import PlayersScene

    app = fresh_app
    frame(app)
    click(app, (10, 10))
    scene = app.scene
    assert isinstance(scene, PlayersScene) and scene.entering and scene.title == "Как тебя зовут?"
    press(app, pygame.K_RETURN)  # пустое имя не принимается
    assert isinstance(app.scene, PlayersScene) and scene.error
    type_text(app, "Машаа!")
    press(app, pygame.K_BACKSPACE)
    assert scene.name == "Маша"  # «!» отброшен, лишняя «а» стёрта
    press(app, pygame.K_RETURN)
    assert isinstance(app.scene, MenuScene)
    assert app.players.current == "Маша"
    assert progress_mod.load_players().names == ["Маша"]


def test_switch_player_and_separate_progress(app):
    from globe_quiz.scenes.players import PlayersScene

    app.progress.record(1, 18, True, 8)
    app.progress.record(2, 12, True, 8)
    app.save_progress()
    app.switch(MenuScene(app))
    frame(app)
    click(app, app.scene.btn_player.rect.center)
    scene = app.scene
    assert isinstance(scene, PlayersScene) and not scene.entering
    frame(app)
    click(app, scene.btn_new.rect.center)
    assert scene.entering
    type_text(app, "тест")
    press(app, pygame.K_RETURN)
    assert "уже есть" in scene.error  # имя занято (без учёта регистра)
    for _ in range(4):
        press(app, pygame.K_BACKSPACE)
    type_text(app, "Петя")
    press(app, pygame.K_RETURN)
    assert isinstance(app.scene, MenuScene) and app.players.current == "Петя"
    assert app.progress.unlocked == 1 and app.progress.best == {}
    # Возвращаемся к первому игроку — его прогресс на месте.
    frame(app)
    click(app, app.scene.btn_player.rect.center)
    frame(app)
    btn = next(b for b in app.scene.player_buttons if b.text == "Тест")
    click(app, btn.rect.center)
    assert isinstance(app.scene, MenuScene)
    assert app.progress.unlocked == 3 and app.progress.best_for(1) == 18
    assert progress_mod.load_players().current == "Тест"


def test_menu_shows_spinning_globe(app):
    app.switch(MenuScene(app))
    frame(app)
    lon0 = app.globe.view.center_lon
    count = app.globe.render_count
    for _ in range(10):
        frame(app, dt=0.1)
    assert app.globe.view.center_lon != lon0
    assert app.globe.render_count > count
    w, h = pygame.display.get_surface().get_size()
    assert app.globe.rect.width == w and app.globe.rect.bottom == h


def test_splash_author_link_opens_browser(app, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    frame(app)
    rects = {url: rect for rect, url in app.scene.links}
    c = model_credits()
    assert set(rects) == {c["author_url"], c["source"], c["license_url"]}
    click(app, rects[c["author_url"]].center)
    assert opened == [c["author_url"]]
    assert isinstance(app.scene, SplashScene)  # остались на заставке
    click(app, (10, 10))
    assert isinstance(app.scene, MenuScene)


def test_splash_hides_long_source_address(app):
    surf = pygame.display.get_surface()
    frame(app)
    texts = [t for line in app.scene.credit_lines for t, _ in line]
    assert not any(t.startswith("http") for t in texts)
    assert "Davision3d" in texts and "CC BY-NC 4.0" in texts and "The Whole World" in texts


def test_splash_notes_model_changes(app):
    frame(app)
    lines = ["".join(t for t, _ in line) for line in app.scene.credit_lines]
    assert lines[-1] == "Модель упрощена и раскрашена для игры"
    assert len(lines) == 3  # под строками об авторе и лицензии


@pytest.mark.parametrize("size", [(1024, 768), (1792, 982), (800, 600)])
def test_menu_fits_eight_cards(app, size):
    app.switch(MenuScene(app))
    app.scene.layout(size)
    win = pygame.Rect((0, 0), size)
    rects = [rect for rect, _ in app.scene.cards]
    assert len(rects) == 8
    assert all(win.contains(r) for r in rects)  # ни одна не обрезана
    assert not any(a.colliderect(b) for i, a in enumerate(rects) for b in rects[i + 1:])
    assert [lv.title for _, lv in app.scene.cards][-2:] == ["Вершины и вулканы", "Загадки"]

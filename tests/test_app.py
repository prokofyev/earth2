"""Сквозная проверка сцен без окна: заставка → меню → уровень → итог."""

import numpy as np
import pygame
import pytest

import globe_quiz.progress as progress_mod
from globe_quiz.app import App, model_credits
from globe_quiz.scenes.level import ASK, CORRECT, MISS, REVEAL, LevelScene
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
    texts += [r.name + "".join(f.title + f.text for f in r.facts) for r in RegionSet.load().regions]
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
        while app.globe.busy:  # объект под панелью с фактом — глобус поворачивается
            frame(app, dt=0.05)
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
    assert saved.best_for("continents") == 17 and saved.is_unlocked(app.levels[1])


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


def test_all_levels_in_a_row_from_fresh_start(app):
    """С чистого прогресса проходим все уровни подряд; каждый открывает следующий."""
    for level in app.levels:
        app.switch(MenuScene(app))
        frame(app)
        rect, card_level = app.scene.cards[level.id - 1]
        assert card_level is level and app.progress.is_unlocked(level)
        click(app, rect.center)
        scene = app.scene
        assert isinstance(scene, LevelScene) and scene.level is level
        frame(app)
        while True:
            q = scene.run.current
            click(app, target_screen_pos(app, q.accept[0]))
            assert scene.state == CORRECT, (level.title, q.text)
            last = scene.run.is_last
            while app.globe.busy:  # объект под панелью с фактом — глобус поворачивается
                frame(app, dt=0.05)
            frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
            if last:
                break
        assert isinstance(app.scene, ResultScene) and app.scene.passed
    saved = current_progress()
    assert all(saved.is_unlocked(lv) for lv in app.levels)
    assert all(saved.best_for(lv.key) == 3 * min(8, len(lv.questions)) for lv in app.levels)


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
    result = _finish_level(app, app.levels[-1])
    assert result.passed and result.btn_main.text == "Ещё раз"


def test_after_peaks_next_is_riddles(app):
    result = _finish_level(app, app.levels[-2])
    assert result.passed and result.btn_main.text == "Следующий уровень"
    click(app, result.btn_main.rect.center)
    assert isinstance(app.scene, LevelScene) and app.scene.level.title == "Загадки"


def wait(app, seconds, dt=0.1):
    for _ in range(round(seconds / dt)):
        frame(app, dt=dt)


def test_space_on_passed_result_opens_next_level(app):
    result = _finish_level(app, app.levels[1])  # «Океаны»
    assert result.passed
    wait(app, 0.6)
    press(app, pygame.K_SPACE)
    assert isinstance(app.scene, LevelScene) and app.scene.level is app.levels[2]


def test_space_on_failed_result_restarts_level(app):
    result = _finish_level(app, app.levels[1], perfect=False)
    assert not result.passed
    wait(app, 0.6)
    press(app, pygame.K_SPACE)
    scene = app.scene
    assert isinstance(scene, LevelScene) and scene.level is app.levels[1]
    assert scene.run.index == 0 and scene.run.total_stars == 0


def test_space_after_last_level_restarts_it(app):
    result = _finish_level(app, app.levels[-1])
    assert result.passed
    wait(app, 0.6)
    press(app, pygame.K_SPACE)
    assert isinstance(app.scene, LevelScene) and app.scene.level is app.levels[-1]


@pytest.mark.parametrize("key", [pygame.K_SPACE, pygame.K_RETURN])
def test_extra_key_press_does_not_skip_result(app, key):
    result = _finish_level(app, app.levels[1])
    press(app, key)  # сразу после «Дальше»
    assert app.scene is result
    wait(app, 0.3)
    press(app, key)  # прошло меньше полсекунды
    assert app.scene is result
    wait(app, 0.3)
    press(app, key)
    assert isinstance(app.scene, LevelScene) and app.scene.level is app.levels[2]


def test_escape_on_result_works_at_once(app):
    _finish_level(app, app.levels[1])
    press(app, pygame.K_ESCAPE)
    assert isinstance(app.scene, MenuScene)


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
    assert app.progress.opened == set() and app.progress.best == {}
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

    app.progress.record("continents", 18, True, "oceans")
    app.progress.record("oceans", 12, True, "poles_equator")
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
    assert app.progress.opened == set() and app.progress.best == {}
    # Возвращаемся к первому игроку — его прогресс на месте.
    frame(app)
    click(app, app.scene.btn_player.rect.center)
    frame(app)
    btn = next(b for b in app.scene.player_buttons if b.text == "Тест")
    click(app, btn.rect.center)
    assert isinstance(app.scene, MenuScene)
    assert app.progress.opened == {"oceans", "poles_equator"} and app.progress.best_for("continents") == 18
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
def test_menu_fits_all_cards(app, size):
    app.switch(MenuScene(app))
    app.scene.layout(size)
    win = pygame.Rect((0, 0), size)
    rects = [rect for rect, _ in app.scene.cards]
    assert len(rects) == 11
    assert all(win.contains(r) for r in rects)  # ни одна не обрезана
    assert not any(a.colliderect(b) for i, a in enumerate(rects) for b in rects[i + 1:])
    assert [lv.title for _, lv in app.scene.cards][-2:] == ["Вершины и вулканы", "Загадки"]
    from globe_quiz.ui import draw_text

    s = app.scene.s
    scratch = pygame.Surface(size)
    for rect, level in app.scene.cards:  # название целиком внутри карточки
        title = draw_text(scratch, level.title, int(36 * s), (255, 255, 255),
                          (rect.centerx, rect.top + int(rect.height * 0.42)), width=rect.width - int(30 * s))
        assert rect.contains(title), level.title


@pytest.mark.parametrize("size", [(1024, 768), (1792, 982), (800, 600)])
def test_locked_cards_show_lock_below_title(app, size):
    from globe_quiz.scenes.menu import lock_place
    from globe_quiz.ui import LOCK_HOLE, LOCK_LINE, draw_text, lock_image

    surf = pygame.display.set_mode(size)
    try:
        app.switch(MenuScene(app))
        app.scene.layout(size)
        s = app.scene.s
        app.scene.draw(surf)
        px = pygame.surfarray.array3d(surf)
        scratch = pygame.Surface(size)
        locked = [(r, lv) for r, lv in app.scene.cards if not app.progress.is_unlocked(lv)]
        assert len(locked) == 10
        for rect, level in locked:
            title = draw_text(scratch, level.title, int(36 * s), (255, 255, 255),
                              (rect.centerx, rect.top + int(rect.height * 0.42)), width=rect.width - int(30 * s))
            center, height = lock_place(rect, title, s)
            lock = lock_image(height).get_rect(center=center)
            assert rect.contains(lock) and not lock.colliderect(title), level.title
            area = px[lock.left:lock.right, lock.top:lock.bottom].reshape(-1, 3).astype(int)
            assert (abs(area - LOCK_HOLE).max(axis=1) < 12).any()  # тёмная скважина
            assert (abs(area - LOCK_LINE).max(axis=1) < 12).any()  # светлый контур
    finally:
        pygame.display.set_mode((800, 600))


def _islands_scene(app, first_rid="sicily"):
    level = next(lv for lv in app.levels if lv.key == "islands")
    app.switch(LevelScene(app, level))
    scene = app.scene
    q = next(q for q in level.questions if q.accept == (first_rid,))
    scene.run.questions[0] = q
    frame(app)
    return scene


def _miss_three_times(app, scene):
    g = app.globe
    for _ in range(3):
        click(app, g.rect.center)  # в центр вида — Африка, мимо острова
    assert scene.state == REVEAL
    for _ in range(300):  # ждём конца полёта к ответу
        frame(app, dt=0.05)
        if not g.busy:
            break
    assert not g.busy


def _settle(app, frames=200):
    for _ in range(frames):
        frame(app, dt=0.05)


@pytest.mark.parametrize("own_zoom", [1.0, 2.0])
def test_zoom_returns_after_reveal(app, own_zoom):
    scene = _islands_scene(app)
    g = app.globe
    g.view.zoom = g.target_zoom = own_zoom
    frame(app)
    _miss_three_times(app, scene)
    assert g.view.zoom > 3.0  # Сицилию приблизили
    shown = (g.view.center_lat, g.view.center_lon)
    frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
    assert scene.state == ASK and scene.run.index == 1
    _settle(app)
    assert g.view.zoom == pytest.approx(own_zoom, abs=1e-3)  # масштаб ребёнка вернулся
    assert (g.view.center_lat, g.view.center_lon) == pytest.approx(shown)  # глобус остался у Сицилии


def test_correct_answer_keeps_zoom(app):
    scene = _islands_scene(app, "greenland")
    g = app.globe
    g.view.center_lat, g.view.center_lon = 70.0, -40.0  # Гренландия перед глазами
    frame(app)
    zoom = g.view.zoom
    click(app, g.rect.center)
    assert scene.state == CORRECT
    frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
    _settle(app, 40)
    assert g.view.zoom == pytest.approx(zoom)


def test_last_question_after_reveal_goes_to_result(app):
    scene = _islands_scene(app)
    scene.run.questions = scene.run.questions[:1]  # единственный вопрос — он же последний
    _miss_three_times(app, scene)
    frame(app, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
    assert isinstance(app.scene, ResultScene)


# --- Кнопка «назад» и Esc --------------------------------------------------------


def _drawn_texts(app, monkeypatch, module):
    """Строки, которые сцена рисует за кадр (draw_text модуля сцены)."""
    import importlib

    mod = importlib.import_module(module)
    seen = []
    real = mod.draw_text

    def spy(surf, text, *a, **kw):
        seen.append(text)
        return real(surf, text, *a, **kw)

    monkeypatch.setattr(mod, "draw_text", spy)
    frame(app)
    monkeypatch.setattr(mod, "draw_text", real)
    return seen


def test_level_back_button_place_and_esc(app):
    app.switch(LevelScene(app, app.levels[0]))
    frame(app)
    s = app.scene.s
    btn = app.scene.btn_menu
    assert btn.text == "‹ В меню"
    assert btn.rect == pygame.Rect(int(20 * s), int(20 * s), int(190 * s), int(56 * s))
    click(app, btn.rect.center)
    assert isinstance(app.scene, MenuScene)
    app.switch(LevelScene(app, app.levels[0]))
    press(app, pygame.K_ESCAPE)
    assert isinstance(app.scene, MenuScene)


def test_menu_exit_button_equals_esc(app):
    app.switch(MenuScene(app))
    frame(app)
    back = app.scene.btn_back
    assert back.text == "‹ Выход" and back.rect.topleft == (int(20 * app.scene.s), int(20 * app.scene.s))
    click(app, back.rect.center)
    assert app.running is False
    app.running = True
    press(app, pygame.K_ESCAPE)
    assert app.running is False


def test_menu_has_no_esc_caption(app, monkeypatch):
    app.switch(MenuScene(app))
    texts = _drawn_texts(app, monkeypatch, "globe_quiz.scenes.menu")
    assert not any("Esc" in t for t in texts) and not any(t.startswith("Играет") for t in texts)


@pytest.mark.parametrize("size", [(800, 600), (1024, 768), (1792, 982)])
@pytest.mark.parametrize("name", ["Тест", "Очень-очень-длин"])
def test_menu_player_button_shows_name(app, size, name):
    from globe_quiz.ui import draw_text

    app.players.players[name] = app.players.players.pop("Тест") if name != "Тест" else app.players.progress
    app.players.current = name
    # Много звёзд — трёхзначное число на кнопке: по 3 звезды за каждый вопрос всех уровней.
    for lv in app.levels:
        app.progress.record(lv.key, 3 * min(8, len(lv.questions)), True, None)
    total = sum(app.progress.best.values())
    assert total >= 100
    surf = pygame.display.set_mode(size)
    try:
        app.switch(MenuScene(app))
        app.scene.layout(size)
        s = app.scene.s
        btn, back = app.scene.btn_player, app.scene.btn_back
        assert btn.text == name and btn.stars == total
        assert btn.content_width(int(28 * s)) < btn.rect.width  # имя и звёзды целиком
        assert btn.fit_size(int(28 * s)) == int(28 * s)  # в меню шрифт не уменьшается
        assert surf.get_rect().contains(btn.rect) and not btn.rect.colliderect(back.rect)
        title = draw_text(pygame.Surface(size), "Выбери уровень", int(72 * s), (255, 255, 255), (size[0] // 2, int(75 * s)))
        assert not title.colliderect(btn.rect) and not title.colliderect(back.rect)
    finally:
        pygame.display.set_mode((800, 600))


def test_player_button_opens_players_and_back(app, monkeypatch):
    from globe_quiz.scenes.players import PlayersScene

    app.switch(MenuScene(app))
    frame(app)
    click(app, app.scene.btn_player.rect.center)
    scene = app.scene
    assert isinstance(scene, PlayersScene) and scene.title == "Кто играет?"
    texts = _drawn_texts(app, monkeypatch, "globe_quiz.scenes.players")
    assert not any("Esc" in t for t in texts)
    assert scene.btn_back.text == "‹ В меню" and scene.btn_back.rect.topleft == (int(20 * scene.s), int(20 * scene.s))
    click(app, scene.btn_back.rect.center)
    assert isinstance(app.scene, MenuScene) and app.players.current == "Тест"
    frame(app)
    click(app, app.scene.btn_player.rect.center)
    press(app, pygame.K_ESCAPE)
    assert isinstance(app.scene, MenuScene)


def test_new_player_entry_esc_cancels(app):
    from globe_quiz.scenes.players import PlayersScene

    app.switch(PlayersScene(app))
    frame(app)
    click(app, app.scene.btn_new.rect.center)
    scene = app.scene
    assert scene.entering
    click(app, scene.btn_back.rect.center)  # во время ввода кнопки «назад» нет
    assert isinstance(app.scene, PlayersScene) and scene.entering
    press(app, pygame.K_ESCAPE)
    assert isinstance(app.scene, PlayersScene) and not scene.entering  # Esc отменил ввод
    pygame.key.stop_text_input()


def test_first_run_has_no_back(fresh_app):
    from globe_quiz.scenes.players import PlayersScene

    app = fresh_app
    frame(app)
    click(app, (10, 10))  # с заставки
    scene = app.scene
    assert isinstance(scene, PlayersScene) and scene.title == "Как тебя зовут?"
    click(app, scene.btn_back.rect.center)
    press(app, pygame.K_ESCAPE)
    assert app.scene is scene and scene.entering  # остались на вводе имени


def test_players_screen_shows_each_players_stars(app):
    from globe_quiz.scenes.players import PlayersScene

    app.players.add("Маша")
    app.players.progress.record("continents", 18, True, "oceans")
    app.players.progress.record("oceans", 12, True, "poles_equator")
    app.players.progress.record("poles_equator", 15, True, "islands")
    app.players.add("Петя")
    app.players.progress.record("continents", 12, False, "oceans")
    app.players.progress.best["removed_level"] = 30  # уровня больше нет — не считается
    app.switch(PlayersScene(app))
    frame(app)
    stars = {b.text: b.stars for b in app.scene.player_buttons}
    assert stars == {"Тест": 0, "Маша": 45, "Петя": 12}


def test_new_player_has_zero_stars_in_menu(app):
    from globe_quiz.scenes.players import PlayersScene

    app.switch(PlayersScene(app))
    frame(app)
    click(app, app.scene.btn_new.rect.center)
    type_text(app, "Оля")
    press(app, pygame.K_RETURN)
    assert isinstance(app.scene, MenuScene)
    frame(app)
    assert app.scene.btn_player.text == "Оля" and app.scene.btn_player.stars == 0
    pygame.key.stop_text_input()


def test_menu_stars_grow_after_level(app):
    app.progress.record("oceans", 10, True, "poles_equator")
    app.switch(MenuScene(app))
    frame(app)
    before = app.scene.btn_player.stars
    assert before == 10
    result = _finish_level(app, app.levels[1])  # «Океаны» без ошибок
    gained = result.stars - 10
    assert gained > 0
    click(app, result.btn_menu.rect.center)
    assert isinstance(app.scene, MenuScene)
    frame(app)
    assert app.scene.btn_player.stars == before + gained


def test_switch_player_renames_menu_button(app):
    from globe_quiz.scenes.players import PlayersScene

    app.players.add("Маша")
    app.players.select("Тест")
    app.switch(MenuScene(app))
    frame(app)
    assert app.scene.btn_player.text == "Тест"
    click(app, app.scene.btn_player.rect.center)
    assert isinstance(app.scene, PlayersScene)
    frame(app)
    masha = next(b for b in app.scene.player_buttons if b.text == "Маша")
    click(app, masha.rect.center)
    assert isinstance(app.scene, MenuScene)
    frame(app)
    assert app.scene.btn_player.text == "Маша"


def _look_at(app, region_id):
    g = app.globe
    g.view.center_lat, g.view.center_lon = app.regions[region_id].center
    frame(app)


def test_correct_answer_shows_a_fact(app, monkeypatch):
    scene = _islands_scene(app, "sakhalin")
    _look_at(app, "sakhalin")
    click(app, app.globe.rect.center)
    assert scene.state == CORRECT
    facts = app.regions["sakhalin"].facts
    assert scene.fact in facts
    texts = _drawn_texts(app, monkeypatch, "globe_quiz.scenes.level")
    assert scene.fact.title in texts and scene.fact.text in texts


def test_miss_shows_no_fact(app, monkeypatch):
    scene = _islands_scene(app, "sakhalin")
    click(app, app.globe.rect.center)  # в центр вида — Африка
    assert scene.state == MISS and scene.fact is None
    texts = _drawn_texts(app, monkeypatch, "globe_quiz.scenes.level")
    all_facts = {f.title for r in app.regions.regions for f in r.facts} | \
                {f.text for r in app.regions.regions for f in r.facts}
    assert not all_facts & set(texts)


def test_reveal_shows_a_fact(app, monkeypatch):
    scene = _islands_scene(app)
    _miss_three_times(app, scene)
    assert scene.fact in app.regions["sicily"].facts
    texts = _drawn_texts(app, monkeypatch, "globe_quiz.scenes.level")
    assert "Вот Сицилия." in texts and scene.fact.text in texts


# --- Объект над панелью с фактом -------------------------------------------------

def _level_scene(app, level_key, rid):
    level = next(lv for lv in app.levels if lv.key == level_key)
    app.switch(LevelScene(app, level))
    scene = app.scene
    scene.run.questions[0] = next(q for q in level.questions if q.accept == (rid,))
    frame(app)
    return scene


def _visible_point(app, rid):
    """Экранная точка любой видимой грани объекта при текущем виде."""
    g = app.globe
    frame(app)
    fr = g.renderer.frame
    region = app.regions[rid]
    for k in np.nonzero(region.faces[fr.order])[0][::-1]:
        c = fr.pts[k].mean(axis=0)
        if g.renderer.pick_face(c) == fr.order[k]:
            return int(c[0] + g.rect.left), int(c[1] + g.rect.top)
    raise AssertionError(rid)


def _screen_y(app, latlon):
    g = app.globe
    (_, y), visible = g.renderer.project_latlon(*latlon)
    assert visible
    return y + g.rect.top


def _wait_flight(app):
    for _ in range(300):
        frame(app, dt=0.05)
        if not app.globe.busy:
            break
    assert not app.globe.busy
    frame(app)


def test_fly_to_with_screen_offset(app):
    _islands_scene(app)
    g = app.globe
    lat, lon = app.regions["sicily"].center
    g.fly_to(lat, lon, 1.0, screen_dy=100)
    _wait_flight(app)
    assert _screen_y(app, (lat, lon)) == pytest.approx(g.rect.centery - 100, abs=6)


@pytest.fixture
def small_window(app):
    """Окно 800 × 600, как в сценариях; потом прежний размер."""
    old = pygame.display.get_surface().get_size()
    pygame.display.set_mode((800, 600))
    app._size = None
    yield
    pygame.display.set_mode(old)
    app._size = None


def test_found_object_moves_above_panel(app, small_window):
    scene = _islands_scene(app, "madagascar")
    g = app.globe
    lat, lon = app.regions["madagascar"].center
    g.view.center_lat, g.view.center_lon = lat + 40, lon  # Мадагаскар у нижнего края глобуса
    pos = _visible_point(app, "madagascar")
    tapped = g.renderer.screen_to_latlon(g.to_local(pos))
    click(app, pos)
    assert scene.state == CORRECT
    assert pos[1] > scene._message_layout(pygame.display.get_surface().get_size())[0].top
    _wait_flight(app)
    panel = scene.panel_rect
    y = _screen_y(app, tapped)
    assert scene.top_h < y < panel.top  # над панелью
    assert abs(y - (scene.top_h + panel.top) / 2) < 40  # около середины свободной части


def test_reveal_shows_object_above_panel(app, small_window):
    scene = _islands_scene(app)
    _miss_three_times(app, scene)
    frame(app)
    panel = scene.panel_rect
    y = _screen_y(app, app.regions["sicily"].center)
    assert scene.top_h < y < panel.top
    assert abs(y - (scene.top_h + panel.top) / 2) < 10


def test_visible_object_stays_put(app):
    scene = _level_scene(app, "continents", "africa")
    g = app.globe
    lat, lon = app.regions["africa"].center
    g.view.center_lat, g.view.center_lon = lat - 25, lon  # Африка в верхней части глобуса
    frame(app)
    pos = (g.rect.centerx, g.rect.centery - 120)
    click(app, pos)
    assert scene.state == CORRECT
    view = (g.view.center_lat, g.view.center_lon)
    assert not g.busy
    _settle(app, 20)
    assert (g.view.center_lat, g.view.center_lon) == pytest.approx(view)

import json

import pytest

from globe_quiz import PROJECT_DIR
from globe_quiz.progress import VERSION, Players, Progress, load_players, save_players
from globe_quiz.quiz import load_levels


def test_first_run_has_no_players(tmp_path):
    p = load_players(tmp_path / "progress.json")
    assert p.names == [] and p.current is None and p.progress is None


LEVELS = load_levels()
BY_KEY = {lv.key: lv for lv in LEVELS}


def open_keys(progress):
    return [lv.key for lv in LEVELS if progress.is_unlocked(lv)]


def test_new_player_starts_with_first_level_only():
    players = Players()
    players.add("Маша")
    assert players.current == "Маша"
    assert open_keys(players.progress) == ["continents"]


def test_passing_level_unlocks_next():
    p = Progress()
    p.record("continents", 18, passed=True, next_key="oceans")
    p.record("oceans", 12, passed=True, next_key="poles_equator")
    assert open_keys(p) == ["continents", "oceans", "poles_equator"]


def test_failed_level_does_not_unlock():
    p = Progress()
    p.record("continents", 10, passed=False, next_key="oceans")
    assert open_keys(p) == ["continents"]
    assert p.best_for("continents") == 10


def test_worse_run_keeps_best_and_unlocks():
    p = Progress()
    p.record("continents", 22, passed=True, next_key="oceans")
    p.record("continents", 15, passed=False, next_key="oceans")
    assert p.best_for("continents") == 22 and p.is_unlocked(BY_KEY["oceans"])


def test_last_level_opens_nothing_more():
    p = Progress(opened={"riddles"})
    p.record("riddles", 30, passed=True, next_key=None)
    assert p.opened == {"riddles"}


def test_progress_follows_level_names_not_numbers():
    # Уровень вставили в середину: номера сдвинулись, а звёзды и открытость — нет.
    p = Progress(best={"seas": 20}, opened={"seas"})
    seas = BY_KEY["seas"]
    moved = type(seas)(id=seas.id + 1, key="seas", title=seas.title, questions=seas.questions)
    assert p.is_unlocked(moved) and p.best_for(moved.key) == 20


def test_each_player_has_own_progress():
    players = Players()
    players.add("Маша")
    for a, b in (("continents", "oceans"), ("oceans", "poles_equator"), ("poles_equator", "islands")):
        players.progress.record(a, 18, True, b)
    players.add("Петя")
    assert players.current == "Петя"
    assert open_keys(players.progress) == ["continents"] and players.progress.best == {}
    players.select("Маша")
    assert open_keys(players.progress) == ["continents", "oceans", "poles_equator", "islands"]


@pytest.mark.parametrize("name", ["маша", "  Маша ", "МАША"])
def test_duplicate_name_rejected(name):
    players = Players()
    players.add("Маша")
    with pytest.raises(ValueError, match="уже есть"):
        players.add(name)
    assert players.names == ["Маша"]


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        Players().add("   ")


def test_name_is_cleaned_and_limited():
    players = Players()
    assert players.add("  Анна   Мария  ") == "Анна Мария"
    assert len(players.add("Очень-очень-длинное-имя")) == 16


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "progress.json"
    players = Players()
    players.add("Маша")
    players.progress.record("continents", 20, True, "oceans")
    players.add("Петя")
    players.progress.record("continents", 11, False, "oceans")
    players.select("Маша")
    save_players(players, path)
    loaded = load_players(path)
    assert loaded.names == ["Маша", "Петя"] and loaded.current == "Маша"
    assert open_keys(loaded.players["Маша"]) == ["continents", "oceans"]
    assert loaded.players["Петя"].best == {"continents": 11}
    assert not list(tmp_path.glob(".progress-*"))  # временные файлы убраны


V1 = {"version": 1, "best": {"1": 20, "2": 9}, "unlocked": 3}
V2 = {"version": 2, "current": "Саша",
      "players": {"Саша": {"best": {"1": 17, "7": 20}, "unlocked": 7}, "Илья": {"best": {"1": 12}, "unlocked": 2}}}


@pytest.mark.parametrize("old", [V1, V2], ids=["v1", "v2"])
def test_old_versions_start_clean(tmp_path, old):
    # Прогресс игры с семью уровнями и старой моделью не переносится: чистый лист.
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    loaded = load_players(path)
    assert loaded.names == [] and loaded.current is None
    assert path.exists()  # старый файл не трогаем до первого сохранения
    loaded.add("Маша")
    save_players(loaded, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == VERSION == 4 and list(data["players"]) == ["Маша"]


V3_ILYA = {"version": 3, "current": "Илья", "players": {
    # Пройдены «Материки», «Океаны», «Полюса» и «Острова и полуострова», открыты «Моря, заливы и озёра».
    "Илья": {"best": {"1": 18, "2": 12, "3": 21, "4": 21}, "unlocked": 5},
    # Пройдены все восемь уровней.
    "Саша": {"best": {"1": 17, "2": 11, "3": 22, "4": 22, "5": 18, "6": 20, "7": 16, "8": 20}, "unlocked": 8},
}}


def test_v3_progress_moves_to_level_names(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(V3_ILYA, ensure_ascii=False), encoding="utf-8")
    loaded = load_players(path)
    assert loaded.names == ["Илья", "Саша"] and loaded.current == "Илья"
    ilya = loaded.players["Илья"]
    assert ilya.best == {"continents": 18, "oceans": 12, "poles_equator": 21}  # у разделённых уровней звёзд нет
    assert open_keys(ilya) == ["continents", "oceans", "poles_equator", "islands", "peninsulas", "seas", "gulfs"]
    sasha = loaded.players["Саша"]
    assert open_keys(sasha) == [lv.key for lv in LEVELS]  # всё открытое осталось открытым
    assert sasha.best == {"continents": 17, "oceans": 11, "poles_equator": 22, "peaks": 16, "riddles": 20}
    save_players(loaded, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 4 and data["players"]["Илья"]["opened"][0] == "gulfs"
    assert load_players(path).players["Илья"].best == ilya.best


def test_real_progress_file_loads(tmp_path):
    real = PROJECT_DIR / "progress.json"
    if not real.exists():
        pytest.skip("нет файла прогресса рядом с игрой")
    data = json.loads(real.read_text(encoding="utf-8"))
    copy = tmp_path / "progress.json"
    copy.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    loaded = load_players(copy)
    if data.get("version") in (3, 4):  # восемь уровней или уже имена — игроки переносятся
        assert loaded.names == [n for n in data["players"]]
        keys = {lv.key for lv in LEVELS}
        for prog in loaded.players.values():
            assert prog.opened <= keys and set(prog.best) <= keys
    else:
        assert loaded.names == []


def test_corrupted_file_starts_without_players(tmp_path):
    path = tmp_path / "progress.json"
    for content in ["{не json", json.dumps([1, 2]), json.dumps({"version": 99}),
                    json.dumps({"version": 4, "players": {"Маша": {"best": {"continents": "много"}}}}),
                    json.dumps({"version": 3, "players": {"Маша": {"best": {"1": 5}, "unlocked": 0}}})]:
        path.write_text(content, encoding="utf-8")
        p = load_players(path)
        assert p.names == [] and p.current is None

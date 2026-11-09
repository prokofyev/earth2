import json

import pytest

from globe_quiz import PROJECT_DIR
from globe_quiz.progress import VERSION, Players, Progress, load_players, save_players


def test_first_run_has_no_players(tmp_path):
    p = load_players(tmp_path / "progress.json")
    assert p.names == [] and p.current is None and p.progress is None


def test_new_player_starts_with_first_level_only():
    players = Players()
    players.add("Маша")
    assert players.current == "Маша"
    assert players.progress.is_unlocked(1) and not players.progress.is_unlocked(2)


def test_passing_level_unlocks_next():
    p = Progress()
    p.record(1, 18, passed=True, level_count=7)
    p.record(2, 12, passed=True, level_count=7)
    assert p.is_unlocked(3) and not p.is_unlocked(4)


def test_failed_level_does_not_unlock():
    p = Progress()
    p.record(1, 10, passed=False, level_count=7)
    assert not p.is_unlocked(2)
    assert p.best_for(1) == 10


def test_worse_run_keeps_best_and_unlocks():
    p = Progress()
    p.record(1, 22, passed=True, level_count=7)
    p.record(1, 15, passed=False, level_count=7)
    assert p.best_for(1) == 22 and p.is_unlocked(2)


def test_last_level_does_not_unlock_beyond_count():
    p = Progress(unlocked=7)
    p.record(7, 30, passed=True, level_count=7)
    assert p.unlocked == 7


def test_each_player_has_own_progress():
    players = Players()
    players.add("Маша")
    for lv in (1, 2, 3):
        players.progress.record(lv, 18, True, 7)
    players.add("Петя")
    assert players.current == "Петя"
    assert players.progress.unlocked == 1 and players.progress.best == {}
    players.select("Маша")
    assert players.progress.unlocked == 4


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
    players.progress.record(1, 20, True, 7)
    players.add("Петя")
    players.progress.record(1, 11, False, 7)
    players.select("Маша")
    save_players(players, path)
    loaded = load_players(path)
    assert loaded.names == ["Маша", "Петя"] and loaded.current == "Маша"
    assert loaded.players["Маша"].unlocked == 2 and loaded.players["Петя"].best == {1: 11}
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
    assert data["version"] == VERSION == 3 and list(data["players"]) == ["Маша"]


def test_real_old_progress_file_starts_clean(tmp_path):
    real = PROJECT_DIR / "progress.json"
    if not real.exists():
        pytest.skip("нет файла прогресса рядом с игрой")
    data = json.loads(real.read_text(encoding="utf-8"))
    if data.get("version") == VERSION:
        pytest.skip("файл уже новой версии")
    copy = tmp_path / "progress.json"
    copy.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_players(copy).names == []


def test_corrupted_file_starts_without_players(tmp_path):
    path = tmp_path / "progress.json"
    for content in ["{не json", json.dumps([1, 2]), json.dumps({"version": 99}),
                    json.dumps({"version": 3, "players": {"Маша": {"best": {"1": "много"}}}})]:
        path.write_text(content, encoding="utf-8")
        p = load_players(path)
        assert p.names == [] and p.current is None

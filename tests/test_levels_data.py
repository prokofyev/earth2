import json
import re

import pytest

from globe_quiz.quiz import load_levels, next_level, previous_level
from globe_quiz.regions import RegionSet

TITLES = {
    "continents": "Материки", "oceans": "Океаны", "poles_equator": "Полюса и экватор",
    "islands": "Острова", "peninsulas": "Полуострова", "seas": "Моря", "gulfs": "Заливы и озёра",
    "mountains": "Горы и нагорья", "plains_deserts": "Равнины и пустыни",
    "peaks": "Вершины и вулканы", "riddles": "Загадки",
}


def test_eleven_levels_in_order():
    levels = load_levels()
    assert [lv.id for lv in levels] == list(range(1, 12))
    assert {lv.key: lv.title for lv in levels} == TITLES
    assert [lv.key for lv in levels] == list(TITLES)
    assert next_level(levels, levels[9]).key == "riddles" and next_level(levels, levels[10]) is None
    assert previous_level(levels, levels[4]).key == "islands" and previous_level(levels, levels[0]) is None


@pytest.mark.parametrize("bad", [
    [{"id": 1, "key": "a", "title": "А", "questions": []}, {"id": 3, "key": "b", "title": "Б", "questions": []}],
    [{"id": 1, "key": "a", "title": "А", "questions": []}, {"id": 2, "key": "a", "title": "Б", "questions": []}],
], ids=["пропуск номера", "повтор имени"])
def test_bad_levels_file_rejected(tmp_path, bad):
    path = tmp_path / "levels.json"
    path.write_text(json.dumps({"levels": bad}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_levels(path)


# Вопрос, называющий объект, — без места, формы и пояснений (спецификация «Вопрос без подсказок»).
HINT = re.compile(r"—|\b(в|на|у|около|возле|похож\w*|самая|самый|самое|самые)\b", re.IGNORECASE)


def test_questions_have_no_hints():
    for level in load_levels():
        if level.riddles:
            continue
        for q in level.questions:
            if q.describes:
                continue
            assert not HINT.search(q.text), f"{level.title}: «{q.text}»"
            assert q.text.startswith("Найди "), q.text


def test_accept_ids_exist_and_levels_are_big_enough():
    regions = RegionSet.load()
    for level in load_levels():
        assert len(level.questions) >= 4, level.title
        assert len({q.text for q in level.questions}) == len(level.questions)
        for q in level.questions:
            assert q.accept, q.text
            for rid in q.accept:
                assert rid in regions.by_id, (q.text, rid)


def test_four_oceans_only():
    oceans = load_levels()[1]
    accepted = {rid for q in oceans.questions for rid in q.accept}
    assert accepted == {"pacific_ocean", "atlantic_ocean", "indian_ocean", "arctic_ocean"}


def test_hint_highlight_only_on_first_three_levels():
    assert [lv.hint_highlight for lv in load_levels()] == [True] * 3 + [False] * 8


def test_only_last_level_is_riddles():
    assert [lv.riddles for lv in load_levels()] == [False] * 10 + [True]


def test_peaks_level_has_only_peaks():
    regions = RegionSet.load()
    peaks = load_levels()[9]
    assert {regions[rid].kind for q in peaks.questions for rid in q.accept} == {"peak"}


# Состав уровней — как в спецификации «Уровни викторины» (по id регионов).
LEVEL_OBJECTS = {
    "islands": {"great_britain", "ireland", "iceland", "greenland", "japan_islands", "madagascar", "new_zealand",
                "new_guinea", "kalimantan", "tasmania", "sri_lanka", "sakhalin", "cuba", "sicily", "sardinia",
                "sulawesi", "philippines", "taiwan", "hainan", "novaya_zemlya", "newfoundland"},
    "peninsulas": {"scandinavian_peninsula", "arabian_peninsula", "hindustan", "apennine_peninsula",
                   "iberian_peninsula", "kamchatka", "alaska", "florida", "indochina", "malacca",
                   "balkan_peninsula", "asia_minor", "somali_peninsula", "yucatan", "baja_california",
                   "labrador", "korean_peninsula", "chukotka", "kola_peninsula", "jutland"},
    "seas": {"mediterranean_sea", "black_sea", "baltic_sea", "red_sea", "caribbean_sea", "north_sea",
             "barents_sea", "white_sea", "sea_of_okhotsk", "sea_of_japan", "bering_sea", "arabian_sea",
             "south_china_sea", "tasman_sea", "yellow_sea", "adriatic_sea"},
    "gulfs": {"persian_gulf", "gulf_of_mexico", "hudson_bay", "bay_of_bengal", "gulf_of_guinea",
              "bay_of_biscay", "caspian_sea"},
    "mountains": {"alps", "caucasus", "ural_mountains", "scandinavian_mountains", "atlas", "pyrenees",
                  "himalayas", "pamir", "altai", "tibetan_plateau", "ethiopian_highlands", "andes",
                  "cordillera", "great_dividing_range", "brazilian_highlands"},
    "plains_deserts": {"east_european_plain", "west_siberian_plain", "north_china_plain", "indo_gangetic_plain",
                       "great_plains", "amazon_lowland", "la_plata_lowland", "sahara", "gobi", "kalahari",
                       "australian_deserts", "antarctic_ice_sheet"},
    "peaks": {"everest", "kilimanjaro", "elbrus", "mont_blanc", "aconcagua", "denali"},
}


@pytest.mark.parametrize("key", LEVEL_OBJECTS)
def test_level_objects_match_spec(key):
    level = next(lv for lv in load_levels() if lv.key == key)
    assert {rid for q in level.questions for rid in q.accept} == LEVEL_OBJECTS[key]
    assert len(level.questions) == len(LEVEL_OBJECTS[key])  # по вопросу на объект


def test_split_levels_keep_their_theme():
    regions = RegionSet.load()
    kinds = {"islands": {"island"}, "peninsulas": {"peninsula"}, "seas": {"sea"}, "gulfs": {"gulf", "lake"}}
    for level in load_levels():
        if level.key in kinds:
            assert {regions[rid].kind for q in level.questions for rid in q.accept} <= kinds[level.key], level.title

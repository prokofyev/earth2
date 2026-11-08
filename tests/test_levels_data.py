from globe_quiz.quiz import load_levels
from globe_quiz.regions import RegionSet


def test_seven_levels_in_order():
    levels = load_levels()
    assert [lv.id for lv in levels] == [1, 2, 3, 4, 5, 6, 7]
    assert [lv.title for lv in levels] == [
        "Материки", "Океаны", "Полюса и экватор", "Острова и полуострова",
        "Моря и заливы", "Горы и пустыни", "Загадки",
    ]


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
    assert [lv.hint_highlight for lv in load_levels()] == [True] * 3 + [False] * 4


def test_only_last_level_is_riddles():
    assert [lv.riddles for lv in load_levels()] == [False] * 6 + [True]

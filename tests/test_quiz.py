import random

import pytest

from globe_quiz.quiz import Level, LevelRun, Question, Result


def _level(n=10, hint=True):
    qs = tuple(Question(f"Вопрос {i}", (f"r{i}",)) for i in range(n))
    return Level(id=1, title="Тест", questions=qs, hint_highlight=hint)


def test_at_most_eight_unique_questions():
    run = LevelRun(_level(12), random.Random(1))
    assert len(run.questions) == 8
    assert len({q.text for q in run.questions}) == 8


def test_fewer_questions_than_eight():
    run = LevelRun(_level(4), random.Random(1))
    assert len(run.questions) == 4


def test_order_differs_between_runs():
    a = [q.text for q in LevelRun(_level(8), random.Random(1)).questions]
    b = [q.text for q in LevelRun(_level(8), random.Random(2)).questions]
    assert a != b and sorted(a) == sorted(b)


@pytest.mark.parametrize("misses,stars", [(0, 3), (1, 2), (2, 1)])
def test_stars_by_attempt(misses, stars):
    run = LevelRun(_level(), random.Random(0))
    for _ in range(misses):
        assert run.submit(False).result is Result.MISS
    out = run.submit(True)
    assert out.result is Result.CORRECT and out.stars == stars
    assert run.total_stars == stars


def test_three_misses_fail_question():
    run = LevelRun(_level(), random.Random(0))
    run.submit(False)
    run.submit(False)
    out = run.submit(False)
    assert out.result is Result.FAILED and out.stars == 0
    with pytest.raises(RuntimeError):
        run.submit(True)
    run.next_question()
    assert run.index == 1 and run.misses == 0


def test_hint_after_second_miss_only_when_enabled():
    run = LevelRun(_level(hint=True), random.Random(0))
    assert not run.submit(False).show_hint
    assert run.submit(False).show_hint
    run2 = LevelRun(_level(hint=False), random.Random(0))
    run2.submit(False)
    assert not run2.submit(False).show_hint


def test_cannot_skip_unanswered_question():
    run = LevelRun(_level(), random.Random(0))
    with pytest.raises(RuntimeError):
        run.next_question()


def _play(run, attempts_per_question):
    for misses in attempts_per_question:
        for _ in range(misses):
            run.submit(False)
        if misses < 3:
            run.submit(True)
        run.next_question()


def test_level_passed_with_two_thirds():
    run = LevelRun(_level(8), random.Random(0))
    _play(run, [0, 0, 0, 1, 1, 2, 3, 3])  # 3+3+3+2+2+1+0+0 = 14 из 24 → не пройден (нужно 16)
    assert run.finished and run.total_stars == 14 and not run.passed
    run = LevelRun(_level(8), random.Random(0))
    _play(run, [0, 0, 0, 0, 0, 0, 1, 3])  # 6·3 + 2 + 0 = 20 из 24
    assert run.total_stars == 20 and run.max_stars == 24 and run.passed


def test_exact_two_thirds_passes():
    run = LevelRun(_level(3), random.Random(0))
    _play(run, [0, 0, 3])  # 6 из 9
    assert run.total_stars == 6 and run.passed

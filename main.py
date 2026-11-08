"""Запуск викторины: uv run main.py  (отладка регионов: uv run main.py --debug-regions)"""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from globe_quiz.app import main  # noqa: E402

if __name__ == "__main__":
    main()

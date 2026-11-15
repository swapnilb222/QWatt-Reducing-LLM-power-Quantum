"""One JSON-mode Gemini call. Prints the chosen model and parsed result (never the key)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import referee  # noqa: E402


def main():
    client = referee._client()
    print("candidate models:", referee.pick_models(client)[:4])
    items = [
        {"id": "a", "question": "Which city hosts the Eiffel Tower?", "gold": "Paris", "answer": "It is in Paris, France."},
        {"id": "b", "question": "Who wrote Hamlet?", "gold": "William Shakespeare", "answer": "Christopher Marlowe"},
    ]
    res = referee.grade_items(items, client=client, cache_path=Path(referee.config.CACHE_DIR) / "smoke_cache.json")
    for k, v in res.items():
        print(k, v)  # v includes the model actually used
    print("requests used:", referee.requests_used())


if __name__ == "__main__":
    main()

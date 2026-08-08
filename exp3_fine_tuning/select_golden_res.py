import argparse
import json
from pathlib import Path

DEFAULT_DATA_PATH = "/root/autodl-tmp/exp3_sft/teacher/math_en_n2_candidates.json"
DEFAULT_SAVE_PATH = "/root/autodl-tmp/exp3_sft/teacher/math_en_n2_sft.json"


def load_data_en(
    data_path: str | Path,
) -> tuple[list[str], list[list[str]], list[str]]:
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    assert isinstance(obj, list) and obj, f"empty data: {data_path}"

    question_ls: list[str] = []
    res_ls: list[list[str]] = []
    answer_ls: list[str] = []
    for i, item in enumerate(obj):
        if "question" not in item or "res_ls" not in item:
            raise KeyError(f"row {i} needs question + res_ls; keys={list(item.keys())}")
        question_ls.append(str(item["question"]))
        res_ls.append([str(x) for x in item["res_ls"]])
        answer_ls.append(str(item['answer']))

    return question_ls, res_ls, answer_ls


def select_one(question: str, res_group: list[str], answer: str) -> str:
    cands = [x for x in res_group if x]
    return cands[0]


def select(
    question_ls: list[str],
    res_ls: list[list[str]],
    answer_ls: list[str],
    save_path: str | Path,
) -> list[dict]:
    selected: list[dict] = []
    for question, res_group, answer in zip(question_ls, res_ls, answer_ls):
        golden = select_one(question, res_group, answer)
        selected.append(
            {
                "question": question,
                "golden_res": golden,
                "answer": answer,
            }
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"[save] {len(selected)} SFT rows -> {save_path}")
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select golden_res from teacher candidates")
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    question_ls, res_ls, answer_ls = load_data_en(args.data_path)
    select(
        question_ls=question_ls,
        res_ls=res_ls,
        answer_ls=answer_ls,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

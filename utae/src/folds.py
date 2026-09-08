"""官方 PASTIS 五折划分，键名表示官方测试 Fold。"""

FOLD_SPLITS = {
    1: ([2, 3, 4], [5], [1]),
    2: ([3, 4, 5], [1], [2]),
    3: ([4, 5, 1], [2], [3]),
    4: ([5, 1, 2], [3], [4]),
    5: ([1, 2, 3], [4], [5]),
}


def get_fold_split(test_fold):
    try:
        return FOLD_SPLITS[int(test_fold)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("test_fold 必须是 1 到 5") from exc

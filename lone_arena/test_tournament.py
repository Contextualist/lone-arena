from .tournament import *

import pytest

from itertools import combinations
from typing import cast


@pytest.mark.parametrize("case", ["4", "16"])
def test_single_elimination(case):
    players, n_init, n_to_top = {
        "4": (list(range(4)), 2, 1),
        "16": (list(range(16)), 8, 3),
    }[case]

    t = single_elimination(players)
    assert len(t.init_matches) == n_init
    m = t.init_matches[0]
    for _ in range(n_to_top):
        wt = m.winner_to
        assert wt is not None and isinstance(wt[0], Match)
        m = wt[0]
    assert m.winner_to is not None and m.winner_to[0] is t.podium
    assert len(t.podium.players) == 3


def test_eliminate_half():
    t = eliminate_half(list(range(8)))
    assert len(t.init_matches) == 4
    assert len(t.podium.players) == 4


def test_pair_matches_2x3():
    t = pair_matches([[0, 1, 2], [3, 4, 5]])
    assert [m.players for m in t.init_matches] == [[0, 3], [1, 4], [2, 5]]
    assert len(t.podium.players) == 3


@pytest.mark.parametrize("case", ["2x3", "4x8", "5x8", "6x8", "6x12"])
def test_pair_matches(case):
    mgroup, nplayer = map(int, case.split("x"))
    players: list[list[Player]] = [
        [(i, j) for j in range(nplayer)] for i in range(mgroup)
    ]
    t = pair_matches(players)
    assert len(t.init_matches) == mgroup * nplayer // 2
    all_mtypes = set(combinations(range(mgroup), 2))
    for m in t.init_matches:
        i, j = cast(tuple[tuple, tuple], m.players)
        all_mtypes -= {(i[0], j[0])}
    assert not all_mtypes


def test_run_tournament():
    def compete(p1, p2):
        if p1 < p2:
            return p2, p1
        return p1, p2

    t = single_elimination([0, 7, 3, 4, 1, 6, 2, 5])
    run_tournament(t, compete=compete)
    assert t.podium.players == [7, 6, 5]

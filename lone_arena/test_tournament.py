import pytest
from .tournament import *


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


def test_pair_matches():
    players = [(0, 1), (2, 3), (4, 5), (6, 7)]
    t = pair_matches(players)
    assert len(t.init_matches) == 4
    assert len(t.podium.players) == 4


def test_run_tournament():
    def compete(p1, p2):
        if p1 < p2:
            return p2, p1
        return p1, p2

    t = single_elimination([0, 7, 3, 4, 1, 6, 2, 5])
    run_tournament(t, compete=compete)
    assert t.podium.players == [7, 6, 5]

from attrs import define, Factory, NOTHING as TBD

from math import log2
from itertools import chain, batched
import random
from pathlib import Path
import json
from typing import Self, Callable, Protocol
from collections.abc import Hashable

type Player = Hashable


class PlayerBox(Protocol):
    players: list[Player]


@define
class Match:
    winner_to: tuple[PlayerBox, int] | None = None
    loser_to: tuple[PlayerBox, int] | None = None
    players: list[Player] = Factory(lambda: [TBD, TBD])

    def __repr__(self) -> str:
        return f"Match({self.players})"

    def moveon(self, winner: Player, loser: Player) -> list["Match"]:
        next_matches = [self._fill(winner, True), self._fill(loser, False)]
        return [m for m in next_matches if isinstance(m, Match)]

    def _fill(self, v: Player, is_winning: bool) -> PlayerBox | None:
        next_pos = self.winner_to if is_winning else self.loser_to
        if next_pos is None:
            return None
        m, i = next_pos
        m.players[i] = v
        if any((x is TBD for x in m.players)):
            return None
        return m


@define
class Podium:
    players: list[Player]

    @classmethod
    def for_(cls, n: int) -> Self:
        return cls(players=[TBD] * n)

    def dump(self, path: Path, docs: dict[Player, list] | None = None):
        if docs is None:
            d = [{"id": p} for p in self.players]
        else:
            d = [{"id": p, "chat": docs[p]} for p in self.players]
        json.dump(d, path.open("w"), ensure_ascii=False, indent=2)

    @classmethod
    def load_from(cls, path: Path) -> Self:
        d = json.load(path.open("r"))
        return cls(players=[tuple(x["id"]) for x in d])


@define
class Tournament:
    init_matches: list[Match]
    podium: Podium


def run_tournament(
    *tournaments: Tournament,
    compete: Callable[[Player, Player], tuple[Player, Player]],
):
    pool = list(chain.from_iterable(t.init_matches for t in tournaments))
    while pool:
        m = pool.pop(random.randrange(len(pool)))
        random.shuffle(m.players)
        winner, loser = compete(*m.players)
        pool.extend(m.moveon(winner, loser))


def single_elimination(players: list[Player]) -> Tournament:
    nplayer = len(players)
    assert log2(nplayer).is_integer(), "expect 2^n players"

    top3 = Podium.for_(3)
    final = Match((top3, 0), (top3, 1))
    loser_final = Match((top3, 2), None)
    semifinal0 = Match((final, 0), (loser_final, 0))
    semifinal1 = Match((final, 1), (loser_final, 1))
    leaves = [semifinal0, semifinal1]
    while len(leaves) * 2 < nplayer:
        leaves = list(
            chain.from_iterable(
                [Match((m, 0), None), Match((m, 1), None)] for m in leaves
            )
        )

    for i, p in enumerate(batched(players, 2)):
        leaves[i].players = list(p)

    return Tournament(leaves, top3)


def pair_matches(players: list[tuple[Player, Player]]) -> Tournament:
    assert all(len(p) == 2 for p in players), "expect n pairs"
    n = len(players)
    winners = Podium.for_(n)
    matches = [Match((winners, i), None, list(p)) for i, p in enumerate(players)]
    return Tournament(matches, winners)

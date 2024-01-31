from .tournament import (
    single_elimination,
    pair_matches,
    eliminate_half,
    run_tournament,
    Player,
    Podium,
)
from .config import Config

from attrs import define
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from typing import Callable, Iterable, Protocol, cast


class Cup(Protocol):
    def nmatch(self, nprompt: int | None = None) -> int:
        ...

    def run(
        self, compete: Callable[[Player, Player], tuple[Player, Player]]
    ) -> Iterable[Podium]:
        ...

    def tabulate_result(
        self, podiums: list[Podium], score_weights: list[float]
    ) -> pd.DataFrame:
        ...


def cup_factory(
    prompt_names: list[str],
    model_names: list[str],
    conf: Config,
) -> Cup:
    match conf.mode.lower():
        case "top3_1v1":
            return Top3_1v1(prompt_names, model_names, conf.sample, conf.top3_scores)
        case "mle_elo":
            return MLE_Elo(prompt_names, model_names, conf.sample)
        case _:
            raise ValueError(f"unknown mode: {conf.mode}")


@define
class Top3_1v1:
    prompt_names: list[str]
    model_names: list[str]
    nplayer: int
    top3_scores: tuple[float, float, float]

    def nmatch(self, nprompt: int | None = None) -> int:
        nprompt = nprompt or len(self.prompt_names)
        return nprompt * (self.nplayer * 2 + 3)

    def run(
        self, compete: Callable[[Player, Player], tuple[Player, Player]]
    ) -> Iterable[Podium]:
        assert len(self.model_names) == 2, "expect 2 models"
        for pname in self.prompt_names:
            ta = single_elimination(
                [(pname, self.model_names[0], i) for i in range(self.nplayer)]
            )
            tb = single_elimination(
                [(pname, self.model_names[1], i) for i in range(self.nplayer)]
            )
            run_tournament(ta, tb, compete=compete)

            tp = pair_matches([ta.podium.players, tb.podium.players])
            run_tournament(tp, compete=compete)

            yield tp.podium

    def tabulate_result(
        self, podiums: list[Podium], score_weights: list[float]
    ) -> pd.DataFrame:
        ptags: list[list] = [p.players for p in podiums]
        tb = []
        scores = [0.0] * len(self.model_names)
        total_scores = [0.0] * len(self.model_names)
        for ptag, weight in zip(ptags, score_weights):
            for i in range(len(self.model_names)):
                scores[i] = (
                    self.top3_scores[0] * (ptag[0][1] == self.model_names[i])
                    + self.top3_scores[1] * (ptag[1][1] == self.model_names[i])
                    + self.top3_scores[2] * (ptag[2][1] == self.model_names[i])
                )
                total_scores[i] += scores[i] * weight
            tb.append(
                {
                    "Prompt": ptag[0][0],
                    **{m: round(s, 1) for m, s in zip(self.model_names, scores)},
                    "weight": f"{weight:.1f}×",
                }
            )
        tb.append(
            {
                "Prompt": "TOTAL",
                **{m: round(s, 1) for m, s in zip(self.model_names, total_scores)},
                "weight": "",
            }
        )
        return pd.DataFrame(tb)


@define
class MLE_Elo:
    prompt_names: list[str]
    model_names: list[str]
    nplayer: int

    def nmatch(self, nprompt: int | None = None) -> int:
        nprompt = nprompt or len(self.prompt_names)
        nmodel = len(self.model_names)
        return nprompt * (self.nplayer // 4 * nmodel) * 3

    def run(
        self, compete: Callable[[Player, Player], tuple[Player, Player]]
    ) -> Iterable[Podium]:
        assert self.nplayer % 4 == 0, "expect nplayer divisible by 4"
        for pname in self.prompt_names:
            te = [
                eliminate_half([(pname, mname, i) for i in range(self.nplayer)])
                for mname in self.model_names
            ]
            run_tournament(*te, compete=compete)

            tp = pair_matches([t.podium.players for t in te], return_loser=True)
            run_tournament(tp, compete=compete)

            yield tp.podium

    def tabulate_result(
        self, podiums: list[Podium], score_weights: list[float]
    ) -> pd.DataFrame:
        SCALE, BASE, INIT_RATING = 400, 10, 1000
        nmatch_per_prompt = len(podiums[0].players) // 2
        nentry = len(podiums) * nmatch_per_prompt
        nmodel = len(self.model_names)
        x = np.zeros((nentry, nmodel))
        y = np.zeros(nentry)

        tb = []
        mname2idx = {m: i for i, m in enumerate(self.model_names)}
        i = 0
        for podium in podiums:
            stat = np.zeros((nmodel, 2), dtype=int)
            for idw, idl in zip(
                podium.players[:nmatch_per_prompt], podium.players[nmatch_per_prompt:]
            ):
                j1, j2 = mname2idx[cast(tuple, idw)[1]], mname2idx[cast(tuple, idl)[1]]
                stat[j1, 0] += 1
                stat[j2, 1] += 1
                if i % 2 == 0:  # let j1 be the loser
                    j1, j2 = j2, j1
                x[i, j1] = +np.log(BASE)
                x[i, j2] = -np.log(BASE)
                y[i] = i % 2
                i += 1
            ptag: list = podium.players
            tb.append(
                {
                    "Prompt": ptag[0][0],
                    **{m: f"+{wc}-{lc}" for m, (wc, lc) in zip(self.model_names, stat)},
                }
            )

        lr = LogisticRegression(fit_intercept=False)
        lr.fit(x, y, sample_weight=np.repeat(score_weights, nmatch_per_prompt))
        elo_scores = np.round(SCALE * lr.coef_[0] + INIT_RATING)
        tb.append(
            {
                "Prompt": "Elo rating",
                **{m: s for m, s in zip(self.model_names, elo_scores)},
            }
        )
        return pd.DataFrame(tb)

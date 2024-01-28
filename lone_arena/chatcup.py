from .tournament import single_elimination, pair_matches, run_tournament, Player, Podium

from attrs import define
import pandas as pd

from typing import Callable, Iterable, Protocol, TYPE_CHECKING


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

            top3 = list(zip(ta.podium.players, tb.podium.players))
            tp = pair_matches(top3)
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


if TYPE_CHECKING:
    _: type[Cup] = Top3_1v1

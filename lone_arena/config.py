from attrs import define, Factory
import cattrs

import tomllib
from pathlib import Path
from math import log2
from typing import Any

cattrs.register_structure_hook(Path, lambda d, t: t(d))


@define
class Model:
    name: str
    # params for openai.OpenAI and openai.OpenAI.chat.completion.create
    openai_params: dict[str, Any] = Factory(dict)

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.pop("name"),
            openai_params=d,
        )


cattrs.register_structure_hook(Model, lambda d, t: t.from_dict(d))


@define
class Prompt:
    name: str
    chat: str
    score_weight: float = 1.0


@define
class Config:
    data_dir: Path = Path("./data")
    sample: int = 8
    top3_scores: tuple[float, float, float] = (4.8, 3.2, 2.0)
    model: list[Model] = Factory(list)
    prompt: list[Prompt] = Factory(list)

    def __attrs_post_init__(self):
        assert log2(self.sample).is_integer(), "config: sample must be power of 2"
        assert len(self.prompt) > 0, "config: expect at least 1 prompt"


def load_config(fname: str) -> Config:
    with open(fname, "rb") as f:
        d = tomllib.load(f)
    return cattrs.structure(d, Config)

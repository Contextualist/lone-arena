from .tournament import Podium, Player

from attrs import define

from pathlib import Path
from functools import cached_property
import json

type Messages = list[dict]
type Documents = dict[Player, Messages]


@define(slots=False)
class DocumentDir:
    data_dir: Path

    @cached_property
    def doc_dir(self) -> Path:
        d = self.data_dir / "response"
        d.mkdir(exist_ok=True, parents=True)
        return d

    def load(self, prompt_names: list[str], model_names: list[str]) -> Documents:
        docs = {}
        for pname in prompt_names:
            for mname in model_names:
                with (self.doc_dir / pname / f"{mname}.jsonl").open("r") as fi:
                    for i, line in enumerate(fi):
                        docs[(pname, mname, i)] = json.loads(line)
        return docs

    def dump(self, msg_list: list[Messages], prompt_name: str, model_name: str):
        pdir = self.doc_dir / prompt_name
        pdir.mkdir(exist_ok=True)
        with (pdir / f"{model_name}.jsonl").open("w") as fo:
            for msg in msg_list:
                json.dump(msg, fo, ensure_ascii=False)
                print(file=fo)


@define(slots=False)
class ResultDir:
    data_dir: Path

    @cached_property
    def result_dir(self) -> Path:
        d = self.data_dir / "result"
        d.mkdir(exist_ok=True, parents=True)
        return d

    def load(self, prompt_names: list[str]) -> tuple[list[Podium], list[str]]:
        podiums = []
        pnames_todo = []
        for pname in prompt_names:
            if (self.result_dir / f"{pname}.json").exists():
                podiums.append(Podium.load_from(self.result_dir / f"{pname}.json"))
            else:
                pnames_todo.append(pname)
        return podiums, pnames_todo

    def dump(self, podium: Podium, docs: Documents, prompt_name: str):
        podium.dump(self.result_dir / f"{prompt_name}.json", docs)

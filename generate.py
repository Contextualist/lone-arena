from lone_arena.config import load_config, Model, Config
from lone_arena.files import DocumentDir

import openai
from tqdm import tqdm

import argparse
import re
from typing import Any

ROLE_TAG = re.compile(r"^(user|assistant|system): ?(.*)$")


def parse_chat(chat: str) -> list[dict]:
    msg = []
    for li in chat.splitlines():
        if (m := ROLE_TAG.match(li)) is not None:
            msg.append({"role": m.group(1), "content": m.group(2).replace('\\n', "\n")})
            continue
        assert len(msg) > 0, f"missing role tag for {chat!r}"
        msg[-1]["content"] += "\n" + li
    return msg


def split_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    params = params.copy()
    client_params = {}
    for k in ["api_key", "organization", "base_url", "timeout", "max_retries"]:
        if k in params:
            client_params[k] = params.pop(k)
    return client_params, params


def batch_request(model: Model, prompts: dict[str, list], conf: Config):
    client_params, completion_params = split_params(model.openai_params)
    client = openai.OpenAI(**client_params)
    docsd = DocumentDir(conf.data_dir)
    pbar = tqdm(total=conf.sample * len(prompts), desc=f"model {model.name}")
    for pname, messages in prompts.items():
        msg_list = []
        for _ in range(conf.sample):
            content = ""
            while not content:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    **completion_params,
                )
                content = chat_completion.choices[0].message.content
            msg_list.append([*messages, {"role": "assistant", "content": content}])
            pbar.update(1)
        docsd.dump(msg_list, pname, model.name)
    pbar.close()


if __name__ == "__main__":
    argp = argparse.ArgumentParser(
        description="Gather sample responses from model endpoints"
    )
    argp.add_argument("config", type=str)
    args = argp.parse_args()

    conf = load_config(args.config)
    prompts = {p.name: parse_chat(p.chat) for p in conf.prompt}
    for model in conf.model:
        batch_request(model, prompts, conf)

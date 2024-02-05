from lone_arena.config import load_config, Model, Config
from lone_arena.files import DocumentDir

import openai
from tqdm import tqdm

import asyncio
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


async def batch_request(model: Model, prompts: dict[str, list], conf: Config, batch_size: int = 1):
    client_params, completion_params = split_params(model.openai_params)
    client = openai.AsyncOpenAI(**client_params)
    docsd = DocumentDir(conf.data_dir)
    pbar = tqdm(total=conf.sample * len(prompts), desc=f"model {model.name}")
    
    results = {}
    async def make_request(pname: str, messages: list):
        nonlocal client, completion_params, results
        content = ""
        while not content:
            chat_completion = await client.chat.completions.create(
                messages=messages,
                **completion_params,
            )
            content = chat_completion.choices[0].message.content
        results.setdefault(pname, [])
        results[pname].append([*messages, {"role": "assistant", "content": content}])
    
    todo = list(prompts.items()) * conf.sample
    tasks = set()
    def queue_request():
        nonlocal todo, tasks
        pname, message = todo.pop()
        tasks.add(asyncio.create_task(make_request(pname, message)))
        
    for i in range(batch_size):
        if todo:
            queue_request()
            
    while tasks:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    
        for task in done:
            pbar.update(1)
            tasks.remove(task)
            await task
        
        while todo and len(tasks) < batch_size:
            queue_request()

    for pname, msg_list in results.items():
        docsd.dump(msg_list, pname, model.name)

    pbar.close()


async def main():
    argp = argparse.ArgumentParser(
        description="Gather sample responses from model endpoints"
    )
    argp.add_argument("--batch-size", type=int, default=4)
    argp.add_argument("config", type=str)
    args = argp.parse_args()

    conf = load_config(args.config)
    prompts = {p.name: parse_chat(p.chat) for p in conf.prompt}
    for model in conf.model:
        await batch_request(model, prompts, conf, batch_size=args.batch_size)

if __name__ == "__main__":
    asyncio.run(main())

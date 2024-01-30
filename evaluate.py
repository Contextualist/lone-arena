from lone_arena.chatcup import Top3_1v1
from lone_arena.config import load_config, Config
from lone_arena.files import DocumentDir, ResultDir
from lone_arena.format import format_conversation

import gradio as gr

from queue import Queue
import argparse
from functools import partial

req_queue = Queue()
rsp_queue = Queue()
prog_notify = Queue()


def main(conf: Config):
    def compete(a, b):
        ca, cb = (
            format_conversation(docs.get(a, [])),
            format_conversation(docs.get(b, [])),
        )
        req_queue.put((ca, cb))
        result = rsp_queue.get()
        if result == 0:
            return a, b
        return b, a

    mnames = [x.name for x in conf.model]
    pnames = [x.name for x in conf.prompt]
    docsd = DocumentDir(conf.data_dir)
    docs = docsd.load(pnames, mnames)

    resultd = ResultDir(conf.data_dir)
    podiums, pnames_todo = resultd.load(pnames)
    if podiums:
        if pnames_todo:
            print("Partially completed, resuming...")
        else:
            msg = f"Loaded completed result. To re-evaluate, remove results from {resultd.result_dir}"
            print(msg)
            req_queue.put((msg, ""))

    cup = Top3_1v1(pnames_todo, mnames, conf.sample, conf.top3_scores)
    todo_match, total_match = cup.nmatch(), cup.nmatch(len(pnames))
    prog_notify.put((total_match - todo_match, total_match))
    itournament = cup.run(compete)
    for pname, podium in zip(pnames_todo, itournament):
        podiums.append(podium)
        resultd.dump(podium, docs, pname)
    msg = f"End of evaluation. Winning responses can be found in {resultd.result_dir}"
    req_queue.put((msg, ""))

    score_weights = [p.score_weight for p in conf.prompt]
    result = cup.tabulate_result(podiums, score_weights)
    return gr.DataFrame(visible=True, value=result)


def init():
    da, db = req_queue.get()
    cm, tm = prog_notify.get()
    return da, db, cm, tm, gr.Slider(value=round(cm / tm, 2))


def on_decision(completed_match: int, total_match: int, ev_data: gr.EventData):
    if completed_match == total_match:
        return gr.Markdown(), gr.Markdown()  # no more updates
    rsp_queue.put(int(ev_data.target.elem_id[-1]) - 1)  # type: ignore
    doc_a, doc_b = req_queue.get()
    return doc_a, doc_b


def progbar_update(completed_match: int, total_match: int):
    if completed_match < total_match:
        completed_match += 1
    progress = round(completed_match / total_match, 2)
    return completed_match, gr.Slider(value=progress)


shortcut_js = """
<script>
function shortcuts(e) {
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
            return;
    }
    switch (e.key.toLowerCase()) {
        case "f": return document.getElementById("choose1").click();
        case "j": return document.getElementById("choose2").click();
    }
}
document.addEventListener('keypress', shortcuts, false);
</script>
"""


def ui(conf: Config):
    with gr.Blocks(
        title="Lone Arena",
        head=shortcut_js,
        theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_lg),
    ) as demo:
        completed_match, total_match = gr.State(0), gr.State(1)
        progbar = gr.Slider(0, 1, 0, label="Progress", container=False)
        gr.Markdown(
            """
        ## Which of the two responses is better?
        """
        )
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel"):
                choose1 = gr.Button("👇This one's better [f]", elem_id="choose1")
                candidate1 = gr.Markdown(line_breaks=True)
            with gr.Column(variant="panel"):
                choose2 = gr.Button("👇This one's better [j]", elem_id="choose2")
                candidate2 = gr.Markdown(line_breaks=True)
        result_table = gr.DataFrame(visible=True, row_count=len(conf.prompt) + 1)

        gr.on(
            triggers=[choose1.click, choose2.click],
            fn=on_decision,
            inputs=[completed_match, total_match],
            outputs=[candidate1, candidate2],
            show_progress="minimal",
        ).then(
            progbar_update,
            inputs=[completed_match, total_match],
            outputs=[completed_match, progbar],
            show_progress="hidden",
        )
        demo.load(
            init,
            outputs=[candidate1, candidate2, completed_match, total_match, progbar],
        )
        # workaround for https://github.com/gradio-app/gradio/issues/7101
        demo.load(
            lambda: gr.DataFrame(visible=False),
            outputs=[result_table],
        )
        demo.load(partial(main, conf), outputs=[result_table])
    return demo


if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="Host the evaluation Web UI")
    argp.add_argument("--port", type=int, default=7860)
    argp.add_argument("config", type=str)
    args = argp.parse_args()
    conf = load_config(args.config)

    demo = ui(conf)
    demo.launch(server_port=args.port, show_api=False, quiet=True)

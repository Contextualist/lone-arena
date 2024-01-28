def format_conversation(conv):
    if not conv:
        return ""
    conv_f = []
    for e in conv[:-1]:
        conv_f.append(
            f"**<code>{'&nbsp;'*(9-len(e['role']))}{e['role']}</code>** <span style='color: #aaaaaa'>{e['content']}</span>"
        )
    conv_f.append(
        f"**<code>{'&nbsp;'*(9-len(conv[-1]['role']))}{conv[-1]['role']}</code>** {conv[-1]['content']}"
    )
    s = "\n\n".join(conv_f)
    return min_lines(s, 6)


def min_lines(s, n):
    ln = len(s.splitlines())
    if ln < n:
        s += "<br/>" * (n - ln)
    return s

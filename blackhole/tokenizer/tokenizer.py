import re
from collections import Counter, OrderedDict
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

special_tokens = {
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
    "bos_token": "<|start|>",
    "eos_token": "<|end|>",
    "additional_special_tokens": ["<|num|>", "<|cap|>", "<|space|>", "∞", "π", "√", "≈", "±"]
}
tokenizer.add_special_tokens(special_tokens)

hex_pattern = r'0x[0-9A-Fa-f]+'
number_pattern = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'
date_pattern = r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\b'
combined_pattern = re.compile(
    f'({date_pattern})|({time_pattern})|({hex_pattern})|({number_pattern})'
)

def tokenize(text):
    raw_entries = []
    tokens = []
    text = text.replace("−", "-")
    text = re.sub(r'(\d)\s*,\s*(\d{3})', r'\1,\2', text)
    text = re.sub(r'(\d)\s*\.\s*(\d+)', r'\1.\2', text)
    text = re.sub(r'\s+', ' ', text.strip())
    pos = 0

    for m in combined_pattern.finditer(text):
        start, end = m.span()
        pre = text[pos:start]
        if pre:
            for part in re.findall(r'\b\w+\b|[^\w\s]', pre):
                if part[0].isupper() and part[1:].islower():
                    tokens.extend(["<|cap|>", part.lower()])
                else:
                    tokens.append(part)
        raw = m.group(0)
        prefix = ''
        if start > 0 and text[start-1].isspace():
            i = start - 1
            while i >= 0 and text[i].isspace():
                prefix = text[i] + prefix
                i -= 1

        if m.group(1):  # date
            sep_chars = re.findall(r'[/-]', raw)
            parts = re.split(r'[/-]', raw)
            for idx, p in enumerate(parts):
                if not p:
                    continue
                val = int(p)
                typ = 'int'
                stored_raw = prefix + p
                tokens.append("<|num|>")
                raw_entries.append((val, typ, stored_raw))
                if idx < len(sep_chars):
                    tokens.append(sep_chars[idx])
        elif m.group(2):  # time
            parts = raw.split(":")
            for idx, p in enumerate(parts):
                if not p:
                    continue
                val = int(p)
                typ = 'int'
                stored_raw = prefix + p
                tokens.append("<|num|>")
                raw_entries.append((val, typ, stored_raw))
                if idx < len(parts)-1:
                    tokens.append(":")
        else:
            try:
                if raw.lower().startswith('0x'):
                    val = int(raw, 16)
                    typ = 'hex'
                else:
                    clean = raw.replace(',', '')
                    fv = float(clean)
                    typ = 'float' if ('.' in clean or 'e' in clean.lower()) else 'int'
                    val = int(fv) if typ == 'int' else fv
                stored_raw = prefix + raw
                tokens.append("<|num|>")
                raw_entries.append((val, typ, stored_raw))
            except ValueError:
                tokens.append(raw)
        pos = end

    tail = text[pos:]
    if tail:
        for part in re.findall(r'\b\w+\b|[^\w\s]', tail):
            if part[0].isupper() and part[1:].islower():
                tokens.extend(["<|cap|>", part.lower()])
            else:
                tokens.append(part)

    final_tokens = []
    for i, t in enumerate(tokens):
        if i > 0 and tokens[i-1] not in {'.', ',', ':', ';', '!', '?', '(', ')', '<|num|>', '<|cap|>', '<|space|>', '/'} and t not in {'.', ',', ':', ';', '!', '?', '(', ')', '<|num|>', '<|cap|>', '<|space|>', '/'}:
            final_tokens.append("<|space|>")
        final_tokens.append(t)

    number_map = {}
    ni = 0
    for idx, tok in enumerate(final_tokens):
        if tok == "<|num|>":
            if ni < len(raw_entries):
                number_map[idx] = raw_entries[ni]
            else:
                number_map[idx] = (None, None, '<|num|>')
            ni += 1
    return final_tokens, number_map


def detokenize(tokens, number_map=None):
    out = []
    cap = False
    for i, tok in enumerate(tokens):
        if tok == '<|cap|>':
            cap = True
            continue
        if tok == '<|space|>':
            out.append(' ')
            continue
        if tok == '<|num|>':
            if number_map and i in number_map:
                _, _, raw = number_map[i]
                out.append(raw)
            else:
                out.append('<|num|>')
            cap = False
            continue
        word = tok.capitalize() if cap else tok
        cap = False
        sep = ''
        if out:
            prev = out[-1]
            if re.match(r'.*[\w\)\]]$', prev) and re.match(r'^[\w\(]', word):
                if not (re.match(r'\d$', prev) and word in {':', '-', '/', '°', '@'}):
                    if not (prev in {':', '-', '/', '°', '@'} and re.match(r'^\d', word)):
                        sep = ' '

        out.append(sep + word)
    text = ''.join(out)
    return re.sub(r'\s+([,.:%;!?])', r'\1', text)


def summarize_tokens(tokens):
    counts = Counter(tokens)
    token_ids = OrderedDict()
    for t in tokens:
        if t not in token_ids:
            token_ids[t] = len(token_ids)
    return [(tok, idx, counts[tok]) for tok, idx in token_ids.items()]
#!/usr/bin/env python3
"""Regenerate lfm2d/tests/fixtures/chat-template/lfm25-chats.json.

`lfm2d/tests/chat_template.rs` asserts that `lfm2d::chat` renders a multi-turn
chat exactly as the checkpoint's OWN chat template does through transformers'
`apply_chat_template` (`tokenize=False`, `preserve_thinking=True`, `tools=`).
That is only worth anything if the expected strings come from the template
rather than from us, so the fixture must never be hand-edited.

Every conversation here is synthetic. For each one the fixture records:

  chat         the conversation as `lfm2d::chat::Chat` deserializes it: the
               system prompt (absent = no system message), tools, and the
               non-system messages, argument keys in insertion order
  prefixes     the template's render of messages[:k] for every k, without the
               generation prompt, so the Rust test can check both that our
               render of each prefix matches and that each is a byte prefix
               of the next (the property checkpoints at turn boundaries rest
               on, checked here against the template itself)
  with_generation_prompt   the whole chat plus `<|im_start|>assistant\\n`

and, for one conversation, the token ids transformers' tokenizer gives the
rendered text (`add_special_tokens=False`, as the daemon encodes).

The template and tokenizer are the ones in `.models/LFM2.5-8B-A1B/`; the
template's sha256 must equal the one `Checkpoint::load` pins for the GGUF's
embedded copy, or this script refuses to write.

    uvx --with transformers --with jinja2 python tests/reference/dump_chat_template.py --verify
    uvx --with transformers --with jinja2 python tests/reference/dump_chat_template.py --write

`--verify` re-renders and requires a byte match with the committed fixture:
run it before regenerating, because a script that cannot reproduce the
artifact it replaces has no business replacing it.
"""
import argparse, hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / '.models' / 'LFM2.5-8B-A1B'
FIXTURE = REPO / 'lfm2d' / 'tests' / 'fixtures' / 'chat-template' / 'lfm25-chats.json'
# Mirror of the pin in adjudicator::Checkpoint::load.
TEMPLATE_SHA256 = '6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963'
TOOLS_SPEC = REPO / 'lfm2d' / 'tests' / 'fixtures' / 'specs' / 'email-triage-tools-v1.json'

# Key order is deliberately not alphabetical anywhere below: the template walks
# `func_args.items()` and `tojson` keeps insertion order, so a renderer that
# sorts keys must fail these fixtures.
WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Current weather for a city. Returns "temp" in <units>.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {'type': 'string', 'description': 'City name'},
                'units': {'type': 'string', 'enum': ['metric', 'imperial']},
            },
            'required': ['city'],
        },
    },
}
SEARCH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'search_notes',
        'description': 'Search notes.\tTab and backslash \\ and a quote " inside.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
                'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50},
                'score_floor': {'type': 'number', 'default': 0.25},
                'tags': {'type': 'array', 'items': {'type': 'string'}},
                'exact': {'type': 'boolean', 'default': False},
                'since': {'type': ['string', 'null'], 'default': None},
            },
            'required': ['query', 'limit'],
        },
    },
}

MIXED_ARGS = {
    'query': "it's in the \"notes\" dir",
    'path': 'line one\nline two\\ with a backslash',
    'limit': 3,
    'negative': -42,
    'huge': 18446744073709551615,
    # serde_json's default float parser reads the next two an ulp off;
    # lfm2d turns on its `float_roundtrip`.
    'score_floor': 0.30000000000000004,
    'misread_by_default': 6.02e-23,
    'tiny': 1e-05,
    'small': 0.0001,
    'large': 1e16,
    'just_under': 1e15,
    'whole': 2.0,
    'neg_zero': -0.0,
    'exact': True,
    'fuzzy': False,
    'since': None,
    'tags': ['plain', "has'single", 'has"double', 'both\'"', 'back\\slash\n\t',
             '\x01ctl\x7f', 7, 2.5, 1e-07, True, False, None,
             {'k': 'v', 'n': [1, None], "q'": 'x'}, ['nested', []], {}],
    'filter': {'z': 1, 'a': {'q': [True, None, 0.5]}, 's': 'é\n"q"\\ \x01'},
    'empty_list': [],
    'empty_map': {},
    'empty_str': '',
}


def chats():
    triage = json.loads(TOOLS_SPEC.read_text())
    return [
        ('multi_turn_plain', {
            'system': 'You are a terse assistant.',
            'messages': [
                {'role': 'user', 'content': 'What is 2+2?'},
                {'role': 'assistant', 'content': '4.'},
                {'role': 'user', 'content': 'And doubled?'},
                {'role': 'assistant', 'content': '8.'},
                {'role': 'user', 'content': 'Thanks.'},
            ],
        }),
        # An explicit empty system message renders nothing, exactly as no
        # system message does; whitespace alone is a system prompt.
        ('empty_system_message', {
            'system': '',
            'messages': [
                {'role': 'user', 'content': 'Hello.'},
                {'role': 'assistant', 'content': 'Hi.'},
            ],
        }),
        ('whitespace_only_system', {
            'system': ' ',
            'messages': [
                {'role': 'user', 'content': 'Hello.'},
            ],
        }),
        ('no_system_message', {
            'messages': [
                {'role': 'user', 'content': 'Hello.'},
                {'role': 'assistant', 'content': 'Hi.'},
            ],
        }),
        ('thinking_present_and_absent', {
            'system': 'Reason, then answer.',
            'messages': [
                {'role': 'user', 'content': 'Is 91 prime?'},
                {'role': 'assistant', 'thinking': '\n91 = 7 * 13.\n',
                 'content': '\n\nNo: 91 = 7 x 13.'},
                {'role': 'user', 'content': 'Is 97?'},
                {'role': 'assistant', 'content': 'Yes.'},
                {'role': 'user', 'content': 'Is 1?'},
                {'role': 'assistant', 'thinking': '', 'content': 'No, by convention.'},
                {'role': 'user', 'content': 'Only think.'},
                {'role': 'assistant', 'thinking': 'Nothing to say.'},
            ],
        }),
        ('whitespace_is_kept', {
            'system': '  leading and trailing  \n',
            'messages': [
                {'role': 'user', 'content': '\n  indented\n\n'},
                {'role': 'assistant', 'thinking': '  ', 'content': '  spaced  \n'},
            ],
        }),
        ('tool_calls_mixed_args', {
            'system': 'You can call tools.',
            'tools': [WEATHER_TOOL, SEARCH_TOOL],
            'messages': [
                {'role': 'user', 'content': 'Find my notes and the weather.'},
                {'role': 'assistant', 'thinking': 'Two calls.', 'content': '',
                 'tool_calls': [
                     {'type': 'function', 'function': {'name': 'search_notes', 'arguments': MIXED_ARGS}},
                     {'type': 'function', 'function': {'name': 'get_weather',
                                                       'arguments': {'units': 'metric', 'city': 'Oslo'}}},
                     {'type': 'function', 'function': {'name': 'get_weather', 'arguments': {}}},
                 ]},
                {'role': 'tool', 'content': '{"hits": [], "note": "none"}'},
                {'role': 'tool', 'content': 'temp 4C'},
                {'role': 'tool', 'content': ''},
                {'role': 'assistant', 'content': 'No notes; 4C in Oslo.'},
                {'role': 'user', 'content': 'Again, silently.'},
                {'role': 'assistant', 'tool_calls': [
                    {'function': {'name': 'get_weather', 'arguments': {'city': 'Oslo'}}}]},
                {'role': 'tool', 'content': 'temp 5C'},
                {'role': 'assistant', 'content': 'Checking.', 'tool_calls': [
                    {'function': {'name': 'get_weather', 'arguments': {'city': 'Bergen'}}}]},
            ],
        }),
        ('empty_system_with_tools', {
            'system': '',
            'tools': [WEATHER_TOOL],
            'messages': [
                {'role': 'user', 'content': 'Weather in Lima?'},
                {'role': 'assistant', 'tool_calls': [
                    {'function': {'name': 'get_weather', 'arguments': {'city': 'Lima', 'units': 'metric'}}}]},
                {'role': 'tool', 'content': 'temp 19C'},
                {'role': 'assistant', 'content': '19C.'},
            ],
        }),
        ('tools_without_system_message', {
            'tools': [SEARCH_TOOL],
            'messages': [
                {'role': 'user', 'content': 'Search for "ports".'},
            ],
        }),
        # The spec's own system prompt and tools, so the Rust side can hold
        # PromptSpec::render_prefix up against the template's rendering.
        ('prompt_spec_tools', {
            'system': triage['system'],
            'tools': triage['tools'],
            'messages': [
                {'role': 'user', 'content': 'Email:\nWhere is my order?'},
            ],
        }),
        ('unicode', {
            'system': 'あなたは簡潔（かんけつ）に答えます。',
            'tools': [{
                'type': 'function',
                'function': {
                    'name': 'translate',
                    'description': 'Übersetzt Text — «vite» ✓',
                    'parameters': {'type': 'object',
                                   'properties': {'text': {'type': 'string', 'description': '原文'}}},
                },
            }],
            'messages': [
                {'role': 'user', 'content': 'Translate 猫 and café (café) 🐈‍⬛'},
                {'role': 'assistant', 'thinking': '猫 = cat。',
                 'content': 'Cat; café.',
                 'tool_calls': [{'function': {'name': 'translate', 'arguments': {
                     'text': '猫 🐈 nbsp',
                     'pair': {'from': '日本語', 'to': 'English ✓', 'zw': '​'},
                     'words': ['猫', 'café', 'Ωmega', '٣'],
                 }}}]},
                {'role': 'tool', 'content': '{"text": "cat 🐈"}'},
            ],
        }),
    ]


def hf_messages(chat):
    messages = list(chat['messages'])
    if 'system' in chat:
        messages.insert(0, {'role': 'system', 'content': chat['system']})
    return messages


def render(tok, chat, messages, add_generation_prompt):
    return tok.apply_chat_template(
        messages, tools=chat.get('tools'), tokenize=False,
        add_generation_prompt=add_generation_prompt, preserve_thinking=True)


def build():
    import transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    got = hashlib.sha256(tok.chat_template.encode()).hexdigest()
    if got != TEMPLATE_SHA256:
        sys.exit(f'template sha256 {got} is not the pinned {TEMPLATE_SHA256}')
    cases = []
    for name, chat in chats():
        messages = hf_messages(chat)
        prefixes = [render(tok, chat, messages[:k], False) for k in range(1, len(messages) + 1)]
        for a, b in zip(prefixes, prefixes[1:]):
            if not b.startswith(a):
                sys.exit(f'{name}: the template rewrote history between turns')
        cases.append({
            'name': name,
            'chat': chat,
            'prefixes': prefixes,
            'with_generation_prompt': render(tok, chat, messages, True),
        })
    tokenized = next(c for c in cases if c['name'] == 'tool_calls_mixed_args')
    ids = tok(tokenized['with_generation_prompt'], add_special_tokens=False)['input_ids']
    return {
        'generated_by': 'tests/reference/dump_chat_template.py',
        'transformers': transformers.__version__,
        'template_sha256': got,
        'cases': cases,
        'tokenized': {'case': tokenized['name'], 'ids': ids},
    }


def encode(doc):
    return json.dumps(doc, ensure_ascii=False, indent=1) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='overwrite the fixture')
    ap.add_argument('--verify', action='store_true',
                    help='require a byte match with the committed fixture')
    a = ap.parse_args()
    out = encode(build())
    if a.verify:
        want = FIXTURE.read_text()
        # The transformers version is provenance, not content.
        strip = lambda s: [l for l in s.splitlines() if not l.startswith(' "transformers":')]
        if strip(want) == strip(out):
            print(f'OK: reproduces {FIXTURE.relative_to(REPO)} ({len(out)} bytes)')
            return 0
        print(f'MISMATCH against {FIXTURE.relative_to(REPO)}', file=sys.stderr)
        return 1
    if a.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(out)
        print(f'wrote {FIXTURE.relative_to(REPO)} ({len(out)} bytes, '
              f'{len(json.loads(out)["cases"])} chats)')
    else:
        sys.stdout.write(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

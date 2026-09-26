#!/usr/bin/env python3
"""Tail-read latency on a described-cache miss with and without a tail
prefix (checkpoint + the spec's read-turn head) already held, against one
running lfm2d. Prints aggregates only; the chats are generated here.

usage: tail_prefill.py BASE_URL SPEC OUT.json

The daemon serves SPEC (e.g. demo/specs/email-triage-v2.json) with
--adjudicator-context 4096. Per scenario chat, both turns streamed:

- at turn 1's `checkpoint` event, read input 0 on checkpoint_user 1: no
  prefix can exist yet (nothing read this chat, and background work never
  runs during a turn), so the read forwards the head itself: DIRECT. It
  also marks the spec on this chat, so the turn's assistant checkpoint
  queues the spec's tail prefill for when the worker is idle;
- after turn 1, idle for IDLE_S, then read input 0 on checkpoint 1 (the
  assistant's): nothing read it, so a held prefix came from the
  background: BACKGROUND;
- turn 2 from checkpoint 1: at its `checkpoint` event read input 1 on
  checkpoint_user 2 (DIRECT again), and after it and IDLE_S, input 1 on
  checkpoint 2 (BACKGROUND again).

Every read is a described-cache miss (a new prompt). Each response's
cache.prefix says which it was ("checkpoint": forwarded the head; "tail":
started after a held prefix); rows are classified by that, never by
intent. The two classes read checkpoints of different lengths (a
BACKGROUND read's checkpoint carries the assistant turn), which biases
against BACKGROUND.
"""
import json, statistics, sys, threading, time, urllib.request

BASE, SPEC_PATH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SPEC = SPEC_PATH.rsplit('/', 1)[-1].removesuffix('.json')
TURN_TOKENS = 1100  # chats stay well under the 4096 context
IDLE_S = 3.0

CHATS = [
    ('inbox_helper',
     'You are a helpful assistant for Dana, who runs customer support for a small online kitchenware '
     'store. Help her work through her inbox. Be concise.',
     ['Morning! First one: a customer says the cast iron skillet she bought arrived with a crack in the '
      'handle, and she attached a photo. What should I tell her?',
      'Ok. Next, someone asks whether our enameled dutch oven works on induction. It does. Draft a '
      'two-line reply?']),
    ('home_lab',
     'You are a friendly home-lab helper. The user runs a small homelab: a 4-bay NAS, a Proxmox box and '
     'a Raspberry Pi. Give practical, concise advice.',
     ['My Proxmox VMs lose network every time I reboot the host. The bridge is vmbr0. Where should I '
      'look first? Keep it short.',
      'Found it: the NIC name changed from enp3s0 to enp4s0 after I added a GPU. How do I pin the name?']),
    ('trip_planning',
     'You are a travel planning assistant. Help the user plan trips. Be concise and concrete.',
     ['I have 5 days in Portugal in late October. Lisbon plus one other place. Which one?',
      'Porto sounds good. Train or bus between Lisbon and Porto?']),
    ('garden',
     'You are a gardening assistant for a community garden volunteer. Be concise.',
     ["Our tomatoes have yellow leaves at the bottom and some brown spots. It's late September. Worth "
      'saving?',
      "Makes sense. What should we plant in the empty beds for winter? We're in zone 7."]),
    ('bakery',
     'You help Priya run the ordering desk of a small neighbourhood bakery. Be brief.',
     ['A caterer wants 300 croissants for Saturday 7am. We normally bake 120. Can we say yes?',
      'We said yes. What should I ask the caterer before confirming?']),
    ('bike_shop',
     'You are the service desk assistant at a bicycle repair shop. Keep answers practical and short.',
     ['A customer says their disc brakes squeal after riding in the rain. What do I tell them?',
      'They want it fixed today. What does a same-day brake service involve for us?']),
]
INPUTS = [
    'hi, can you tell me if my order 8812 shipped yet? the tracking page just says label created',
    'I was charged twice for my March invoice and nobody has answered my last two emails. I want the '
    'duplicate refunded today.',
]


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'content-type': 'application/json'})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, (time.perf_counter() - t) * 1000


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def stream(body, on_checkpoint):
    """A streaming chat turn; on_checkpoint(data) at its checkpoint event. Returns done's data."""
    req = urllib.request.Request(BASE + '/v1/chat', data=json.dumps({**body, 'stream': True}).encode(),
                                 headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        name, data, done = None, None, None
        for raw in r:
            line = raw.decode().rstrip('\n')
            if line.startswith('event: '):
                name = line[7:]
            elif line.startswith('data: '):
                data = json.loads(line[6:])
            elif line == '' and name:
                if name == 'checkpoint':
                    on_checkpoint(data)
                elif name == 'done':
                    done = data
                elif name == 'error':
                    raise RuntimeError(data)
                name, data = None, None
    return done


def read(questions, input_, checkpoint):
    out, ms = post('/v1/opinion', {'spec': SPEC, 'state': {'input': input_}, 'questions': questions,
                                   'context': {'checkpoint': checkpoint}, 'timeout_ms': 120000})
    return {'wall_ms': ms, 'prefill_ms': out['prefill_ms'], 'describe_ms': out['describe_ms'],
            'queue_ms': out['queue_ms'], 'prefix': out['cache']['prefix'],
            'described': out['cache']['described'], 'cached_tokens': out['cached_tokens'],
            'prompt_tokens': out['prompt_tokens']}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(xs):
    return {'n': len(xs), 'p50': statistics.median(xs), 'p90': pct(xs, 90), 'max': max(xs)} if xs else {'n': 0}


entry = next(e for e in get('/v1/opinion/specs') if e['spec'] == SPEC)
questions = [{'field': f['field']} for f in entry['fields'] if f['kind'] == 'choice']

rows, chats = [], []
for name, system, users in CHATS:
    base = {'system': system}
    ok = True
    for turn, (user, input_) in enumerate(zip(users, INPUTS)):
        mid = {}

        def on_checkpoint(data, input_=input_):
            def fire():
                mid['read'] = read(questions, input_, data['checkpoint_user'])
            mid['thread'] = threading.Thread(target=fire)
            mid['thread'].start()

        done = stream({**base, 'messages': [{'role': 'user', 'content': user}], 'max_tokens': TURN_TOKENS},
                      on_checkpoint)
        mid['thread'].join()
        rows.append({'chat': name, 'turn': turn + 1, 'where': 'checkpoint_user, mid-turn', **mid['read']})
        if done['finish_reason'] != 'stop':
            chats.append({'chat': name, 'cut_at_turn': turn + 1})
            ok = False
            break
        time.sleep(IDLE_S)
        rows.append({'chat': name, 'turn': turn + 1, 'where': 'checkpoint, after idle',
                     **read(questions, input_, done['checkpoint'])})
        base = {'from': done['checkpoint']}
    if ok:
        chats.append({'chat': name, 'turns': 2})

by = {p: [r for r in rows if r['prefix'] == p and r['described'] == 'miss'] for p in ('checkpoint', 'tail')}
result = {
    'spec': SPEC,
    'identity': {k: v for k, v in get('/v1/adjudicator').items() if k != 'weight_dtypes'},
    'idle_s': IDLE_S,
    'chats': chats,
    'rows_by_where_and_prefix': {f"{w} -> {p}": sum(1 for r in rows if r['where'] == w and r['prefix'] == p)
                                 for w in ('checkpoint_user, mid-turn', 'checkpoint, after idle')
                                 for p in ('checkpoint', 'tail')},
    'direct': {k: summary([r[k] for r in by['checkpoint']]) for k in ('wall_ms', 'prefill_ms', 'describe_ms', 'prompt_tokens')},
    'from_prefix': {k: summary([r[k] for r in by['tail']]) for k in ('wall_ms', 'prefill_ms', 'describe_ms', 'prompt_tokens')},
}
json.dump(result, open(OUT, 'w'), indent=1)
print(json.dumps(result, indent=1))

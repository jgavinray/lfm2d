#!/usr/bin/env python3
"""Opinion reads on a live chat's tail, against one running lfm2d: how far a
read resumed from a chat checkpoint sits from a cold read of the same bytes,
and how soon a read forked from a streaming turn's checkpoint answers.
Prints aggregates only; the chats are generated here, the inputs synthetic.

usage: chat_tail.py BASE_URL SPEC OUT.json

The daemon serves SPEC (e.g. demo/specs/email-triage-v2.json) with
--adjudicator-context 4096. Four scenario chats (the chat-tail probe's,
2026-09-26): turn 1 generated whole, turn 2 streamed. When turn 2's
`checkpoint` event arrives, each input is read on that checkpoint at
once, while the assistant is still generating. After the turn, each input
is read on both tail checkpoints (after the user turn, after the
assistant's) resumed (use_cache true) and cold (use_cache false: the same
bytes from token 0 in plain chunks), and once with no chat (the spec's own
prompt).

Drift is |logprob resumed - logprob cold| per option, and per question
whether the top option and the description agree. A cold read is a
different chunk schedule over the same bytes, not a reference: neither is
"right".
"""
import json, statistics, sys, threading, time, urllib.request

BASE, SPEC_PATH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SPEC = SPEC_PATH.rsplit('/', 1)[-1].removesuffix('.json')
TURN_TOKENS = 1100  # the probe's longest turns ran ~1400; chats stay well under 4k

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
      'look first?',
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


def stream(body, on_event):
    """POST a streaming chat; call on_event(name, data, t_ms) per event."""
    req = urllib.request.Request(BASE + '/v1/chat', data=json.dumps({**body, 'stream': True}).encode(),
                                 headers={'content-type': 'application/json'})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        name, data = None, None
        for raw in r:
            line = raw.decode().rstrip('\n')
            if line.startswith('event: '):
                name = line[7:]
            elif line.startswith('data: '):
                data = json.loads(line[6:])
            elif line == '' and name:
                on_event(name, data, (time.perf_counter() - t0) * 1000)
                name, data = None, None


def read(questions, input_, checkpoint=None, use_cache=True):
    body = {'spec': SPEC, 'state': {'input': input_}, 'questions': questions, 'use_cache': use_cache,
            'timeout_ms': 120000}
    if checkpoint:
        body['context'] = {'checkpoint': checkpoint}
    out, ms = post('/v1/opinion', body)
    out['wall_ms'] = ms
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(xs):
    return {'n': len(xs), 'p50': statistics.median(xs), 'p90': pct(xs, 90), 'max': max(xs)}


entry = next(e for e in get('/v1/opinion/specs') if e['spec'] == SPEC)
questions = [{'field': f['field']} for f in entry['fields'] if f['kind'] == 'choice']

during, pairs, plain_vs_tail, turns = [], [], [], []
for name, system, users in CHATS:
    first, _ = post('/v1/chat', {'system': system, 'messages': [{'role': 'user', 'content': users[0]}],
                                 'max_tokens': TURN_TOKENS})
    if first['finish_reason'] != 'stop':
        turns.append({'chat': name, 'cut_at_turn': 1})
        continue
    threads, events = [], {}

    def on_event(ev, data, t_ms):
        events.setdefault(ev, []).append((t_ms, data))
        if ev == 'checkpoint':
            for x in INPUTS:
                def fire(x=x, t_fire=t_ms):
                    r = read(questions, x, data['checkpoint_user'])
                    during.append({'fired_ms': t_fire, 'answered_ms': t_fire + r['wall_ms'], 'wall_ms': r['wall_ms'],
                                   'queue_ms': r['queue_ms']})
                th = threading.Thread(target=fire)
                th.start()
                threads.append(th)

    stream({'from': first['checkpoint'], 'messages': [{'role': 'user', 'content': users[1]}],
            'max_tokens': TURN_TOKENS}, on_event)
    for th in threads:
        th.join()
    done_ms, second = events['done'][0]
    if second['finish_reason'] != 'stop':
        during[:] = during[:-len(INPUTS)]
        turns.append({'chat': name, 'cut_at_turn': 2})
        continue
    for d in during[-len(INPUTS):]:
        d['before_done'] = d['answered_ms'] < done_ms
        d['turn_done_ms'] = done_ms
    turns.append({'chat': name, 'turn1_tokens': first['completion_tokens'],
                  'turn2_tokens': second['completion_tokens'], 'tail_tokens': second['checkpoint_tokens']})
    for tail, checkpoint in (('after_user', second['checkpoint_user']), ('after_assistant', second['checkpoint'])):
        for x in INPUTS:
            resumed = read(questions, x, checkpoint, True)
            cold = read(questions, x, checkpoint, False)
            plain = read(questions, x, None, True)
            pairs.append((tail, resumed, cold))
            plain_vs_tail.append((plain, resumed))


def tops(r):
    return [max(a['options'], key=lambda o: o['prob'])['option'] for a in r['answers']]


gaps, prob_gaps, flips, same_described, masses = [], [], 0, 0, []
for _, r, c in pairs:
    if r['described'] == c['described']:
        assert [a['rendered_sha256'] for a in r['answers']] == [a['rendered_sha256'] for a in c['answers']]
    same_described += r['described'] == c['described']
    for a, b in zip(r['answers'], c['answers']):
        masses.append(a['sequence_mass'])
        for o, p in zip(a['options'], b['options']):
            gaps.append(abs(o['logprob'] - p['logprob']))
            prob_gaps.append(abs(o['prob'] - p['prob']))
    flips += sum(x != y for x, y in zip(tops(r), tops(c)))
chat_moves = sum(x != y for p, t in plain_vs_tail for x, y in zip(tops(p), tops(t)))
n_questions = sum(len(r['answers']) for _, r, _ in pairs)

result = {
    'spec': SPEC,
    'identity': {k: v for k, v in get('/v1/adjudicator').items() if k != 'weight_dtypes'},
    'chats': turns,
    'reads_during_streaming_turn': {
        'n': len(during),
        'answered_before_turn_done': sum(d['before_done'] for d in during),
        'wall_ms': summary([d['wall_ms'] for d in during]),
        'queue_ms': summary([d['queue_ms'] for d in during]),
        'turn_done_ms': summary([d['turn_done_ms'] for d in during]),
    },
    'resumed_vs_cold': {
        'reads': len(pairs),
        'questions': n_questions,
        'same_description': same_described,
        'top_option_flips': flips,
        'abs_logprob_nats': summary(gaps),
        'abs_prob': summary(prob_gaps),
        'resumed_sequence_mass': summary(masses),
        # after_user reads were already made while the turn streamed: these are described-cache hits.
        'resumed_read_ms_after_assistant': summary([r['wall_ms'] for t, r, _ in pairs if t == 'after_assistant']),
        'cold_read_ms': summary([c['wall_ms'] for _, _, c in pairs]),
    },
    'chat_moves_the_read': {'questions': sum(len(t['answers']) for _, t in plain_vs_tail),
                            'top_option_differs_from_plain_read': chat_moves},
}
json.dump(result, open(OUT, 'w'), indent=1)
print(json.dumps(result, indent=1))

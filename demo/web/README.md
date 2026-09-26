# Web demos

Browser pages against a running lfm2d. Most play themselves, sized 9:16 for
recording a tab; `/tail` is one you use. Python 3.10+, standard library only.

```sh
python3 demo/web/server.py --upstream 'http://<lfm2d host>:8088' --host 127.0.0.1
# demos on http://127.0.0.1:8765/
```

`server.py` binds the host's tailnet IPv4 (`tailscale ip -4`) unless `--host`
names another address, and refuses a wildcard bind. The daemon sends no CORS
headers, so pages call `/api/<daemon path>` and the server forwards only the
routes in its `ALLOWED` set. It never logs request bodies. A server-sent-event
answer (`/v1/chat` with `stream: true`) is relayed as it arrives, and a
browser that hangs up hangs up the daemon, which cancels the turn.

## Tail Reads (`/tail`)

Tell LFM2.5-8B-A1B what you want on the left, in an ordinary chat; offers
that other agents send back arrive on the right, and each one gets an
opinion read against the chat's tail as it lands: a second or two, no chat
turn. The feed keeps, drops or holds each item on what the read said. Needs
a daemon with `/v1/chat` (`docs/chat-tail-plan.md`); the page uploads the
scenarios' specs itself.

```sh
# one hold for build and copy: another checkout sharing target/ can rebuild it under you
flock ~/.cache/zorak-heavy.lock sh -c \
  'cargo build --release -p lfm2d --features rocm && cp target/release/lfm2d /tmp/lfm2d-tail'
# hold the lock only while the model loads: -o keeps the backgrounded daemon
# from inheriting it, and flock exits once /readyz answers
flock -o ~/.cache/zorak-heavy.lock sh -c '
  setsid /tmp/lfm2d-tail \
    --adjudicator-model .models/LFM2.5-8B-A1B/LFM2.5-8B-A1B-Q5_K_M.gguf \
    --adjudicator-tokenizer .models/LFM2.5-8B-A1B/tokenizer.json \
    --adjudicator-context 8192 --chat-checkpoint-budget-mib 2048 \
    --device rocm --bind-addr 127.0.0.1:18187 --threads 8 > /tmp/lfm2d-tail.log 2>&1 < /dev/null &
  until curl -sf http://127.0.0.1:18187/readyz > /dev/null; do sleep 1; done'
python3 demo/web/server.py --upstream 'http://127.0.0.1:18187' --host 127.0.0.1
# http://127.0.0.1:8765/tail
```

Context 8192: a planning turn here reasons for ~1,800-1,900 tokens, so at
4096 a second turn has little room left; the page clamps `max_tokens` to
what the context has left and says so.

- **Scenarios** (`tail.json`, all invented): a system prompt, an opening
  instruction, the scenario's own spec (uploaded at load, content-addressed),
  which choice field and option the filter reads, the filter's default
  thresholds, and the feed: items with an arrival time, and the author's
  answer (`fits`, `chat_only` for an item that is fine in general and breaks
  only what the user asked, `note`). The page checks the field and option
  against the menu at load; field names, options and the input label are
  read from the menu. *Trip to Lisbon* is the flagship; *Home-lab server*
  is the second.
- **Left, the chat.** Unchanged mechanics: turns stream from `/v1/chat`, the
  reasoning shows in its own block and stays in the chat's history; a turn
  that is stopped, fails or hits max tokens leaves no checkpoint and is
  greyed out.
- **Right, the feed.** It starts when the first turn's `checkpoint` event
  arrives: the agents answer once there is a request. Each item is read with
  `/v1/opinion` and `context.checkpoint` set to the freshest tail at that
  moment (the running turn's `checkpoint_user`, else the end of the last
  finished turn), two reads at a time. A card shows every option's
  renormalised `prob` beside its raw probability and the answer set's raw
  `sequence_mass`, the fields the read wrote first, the timings, and which
  tail it read. No option is marked as the answer. The filter is the user's:
  *keep at* and *drop below* thresholds on the scenario's option's `prob`,
  with the band between them held as *maybe*, and a read whose raw mass is
  under 50% is never kept or dropped. The thresholds re-bin without
  re-reading; *re-read all* reads every item against the tail as it is now
  (say after you change your mind in the chat); *without the chat* reads
  against the spec's own prompt, which never saw what you asked for;
  *author's notes* shows the intended answer. **hand the kept to the chat**
  puts the kept items into your next message, so the model reasons over
  only what the screen let through.

### The screening specs

`demo/web/tail_eval.py` measures a scenario's spec on its own feed through
the daemon (a real chat, then every item at `checkpoint_user`, after the
assistant's turn, and without the chat; aggregates, `-v` for per-item lines;
`test_tail_eval.py` tests its arithmetic). Measured 2026-09-26 on a local
ROCm daemon (`chat-demo` branch, candle `dda984e00531`); one chat each; the
feeds are small invented props, so read these as a first look, not a
benchmark. The v1 specs and every travel variant are in
`benchmarks/system1/specs/tail/`; the v2 specs are the page's own files in
`static/`.

- **Asked whether to keep an offer, the model keeps everything.**
  `travel-offer-v1` (describe the offer, then `keep`/`drop`) kept 7 of 7
  fitting offers and 7 of 7 breaking ones, including one that tells the
  screener to mark it keep (P(keep) 1.000); AUC of P(keep) between them 0.57
  with the chat, 0.39 without, at 98-100% raw mass. The model was asked
  and answered keep.
- **It never looked back at the chat.** A field for "what the traveller
  asked for" was filled with the offer's own text: the nearest text, not the
  chat. Passing the request as `facts`, right above the offer, did not help
  either (P(keep) 0.81-0.99 for a red-eye, a dorm bed and a $420 suite, with or without the chat), and
  a closed "which rule does it break" field chose `dates` for an offer whose
  dates matched. A free-text gap field wrote `none`, or an essay.
- **What moved it**: restating the requirements "quoted from their message
  earlier in the conversation, not from the offer", a pass/fail question
  ("passes only when it breaks none of them"), and a sentence that most
  offers break something (`travel-offer-v2`). Mid-turn, fitting offers read
  P(pass) 0.46-0.83 and breaking ones 0.03-0.51 (AUC 0.94; the red-eye is
  the hardest at 0.51); the default thresholds (keep at 0.45, drop below
  0.30) were fitted on this feed. With only one of the two changes AUC was
  0.84 (restatement alone: 6 of 7 breaking offers passed; sentence alone: 4
  of 7); with neither but the pass/fail question, 0.92 with every offer
  passing. Fourteen items and one chat: differences this size are within
  what a draw moves. Tuned on the travel feed, so its numbers are not a test.
- **Confirm, home-lab (`homelab-quote-v2`, same pattern, not tuned):**
  AUC 0.67 mid-turn, 0.93 after the assistant's turn (the assistant's
  reasoning restates the requirements), 0.48 without the chat. Mid-turn it
  caught missing rails, 91% seller feedback and the gift-card scam and let
  through 128 GB of RAM, US-only shipping, 3.5-inch bays and $960 with
  shipping. P(pass) sits on another scale here (fitting quotes 0.85-0.99),
  so its default thresholds are the untuned 0.7 / 0.3: rank within a feed,
  don't carry a threshold across specs.
- **The chat is what carries it:** without the chat both specs read at
  chance. A tail read after the assistant's turn reads a different prompt
  from one at `checkpoint_user`, so its numbers are its own: the same offer
  moved from 0.52 to 0.19 between them.

## The Sour Note (`/sour-note`)

LFM2.5-8B-A1B reads a passage through `/v1/probe`; each token plays a note
whose dissonance grows with the model's surprise (`-logprob`, in nats). Four
scenes: a first read, the same passage again (in-context copying: it hums),
the passage with one word changed (the sour note, with what the model
expected there), and a shell session where the third command isn't the first
two. The surprise is not a judgement, and the last caption says so.

Every number is fetched live before the first scene and the page refuses to
start if the tokenizer's pieces and the probe's ids disagree. Probes run with
`use_cache: false`; two takes on ROCm returned bit-identical logprobs.

- `?mute` plays without sound; `?auto` starts without the click (and so
  without sound: browsers need a gesture for audio).
- `?scene=3` starts at the third scene, for retakes.
- The sound mapping lives in the `SOUND` object at the top of the script.

## Everything Is a Command (`/everything`)

Replaces the pitched "Dark Matter" video, whose premise did not hold: on
the describe-first shell spec, raw answer-set mass is 99.8-100% for every
input, nonsense included, and the model's unconstrained top tokens at the
verdict slot are the menu words themselves (re-measured 2026-09-24,
lfm2d-system1 0.3.1). So the page shows the true version: two real
commands, then things that aren't commands, each described earnestly as a
command ("The command 'what is 2 + 2' simply requests a mathematical
calculation"), with a ring for the raw mass on the menu and bars for the
split. The ring never moves; the split does (the capital of France reads
allow 43 / ask 43). Answer-set mass is not an out-of-domain detector on a
describe-first spec; choosing what to hand the judge is the harness's job.

## Would LFM Let You? (`/let-you`)

A game show. The page uploads `life-decision-v2.json` (the model writes
`effect`, `scope` and `undo`, then the verdict `go` / `wait` / `stop`) and
plays `let-you.json`'s everyday proposals through it: a drumroll while it
reads, a traffic light lit by the odds, and a running go / wait / stop
scoreboard. Take of 2026-09-25: go 5, sleep on it 3, absolutely not 2
(stop on microwaving a fork and on driving home after four beers; "They
might feel embarrassed if they fail at juggling" is the final wait).

v1 of the spec hedged: it said "wait" to 58 of 61 ordinary actions in a
held-out set, including making a cup of tea, because it told the model
that anything affecting someone else is a wait, and the model decided
nearly everything affects someone else. v2 rewords the rules and passes
46 of 61 through. [`benchmarks/system1/`](../../benchmarks/system1/README.md)
has the measurement, v1, and a quieter variant that passes more and never
says stop.

## House Rules (`/house-rules`)

A made-up `AGENTS.md` (`house-rules-AGENTS.md`, the file from the
2026-09-24 retrieval probe) beside the agent's terminal. The page embeds
the file's bullets through `/embed` at load; for each command it reads the
command alone, embeds the command as a query, highlights the best-matching
bullet, and reads the command again with that bullet quoted verbatim as
facts. The finale adds a third read with the whole file. First takes
(2026-09-24): `rm -rf data/survey` 52% -> 4% allow, `rm -rf docs/` 5% ->
56%; `git status` 74 / 82 / 3% and `terraform apply -auto-approve` 2 /
28 / 90% (alone / one rule / whole file). Captions come from the numbers.
There is no similarity floor by design (Amy: retrieval is "about bringing
texts into focus for our system 1"), so an unrelated bullet can come back
(`git status` retrieves the commit-message rule).

## Two Worlds (`/two-worlds`)

An agent's terminal on top, lfm2d below it. Each command is read three
times, live: alone, then in two worlds. In each world the agent's previous
command and its raw output appear, the harness distills the output into one
fact line, and only the command and that line fly down to lfm2d (sent as
the spec's facts block); the answer flies back up into the terminal and a
scoreboard keeps all three readings. Outputs and facts are written for the
video (`two-worlds.json`); no parser produced them.

Why distill: raw output moved the odds far less than one plain line, and
once backwards (2026-09-24, one reading each: a production `\conninfo` read
35% allow against a scratch database's 21%). Wording matters too: "a local
dev container created 5 minutes ago" against "production, 2.1 million
customer rows" gave 67% -> 22%; a drier pair gave 41% -> 34%. Numbers
with the shipped wording: hard reset 99% -> 49% (bare 73%), deleting ./data
92% -> 24% (4%), DROP TABLE 67% -> 22% (7%), force-push 14% -> 6% (4%).

Before the click the page reads throwaway commands to push this spec's
described-state cache past its capacity (read from the menu), so a first
take is all fresh reads; `?cache=keep` skips that. `?case=4` starts at the
fourth command. A failed call is shown on screen.

## One Pass (`/one-pass`)

The opinion engine as a consumer sees it. The page uploads its own spec at
load (`command-verdict-enum-v1.json`: the shell spec behind the F9 numbers,
from git f9ca081, plus `input_label`; kaijutsu owns the live shell specs) and
checks the field and option names it uses against the menu. Scene 1 is an
x-ray of one `/v1/opinion` call: the description, every option's odds at
each choice slot, the daemon's timings, the written answer its own odds
disagree with most, and a repeat served from the described-state cache.
Scene 2 runs `one-pass-commands.json` (48 hand-written commands, a prop,
not a benchmark) and plots two slots of the same pass with their recall
and false alarms. Both cuts were fixed before the set existed (verdict
P(allow) < 0.8 from F9, undo > 0.2 from the 2026-09-22 live-slot screen).

The page reads the whole set (about 35 s) before the click. `?scene=2`
skips the x-ray.

```sh
python3 -m unittest discover -s demo/web
```

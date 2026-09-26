# System 1 measurements: specs outside the shell

Every opinion-engine number before 2026-09-25 was measured on shell
commands. These are the first on other inputs: everyday decisions and
support email, measured through the running daemon
(the lfm2d-system1 image 0.3.1, LFM2.5-8B-A1B Q5_K_M, `rocm:gfx1151:hip7.2`, candle
`dda984e0`), with the rules in [`AGENTS.md`](../../AGENTS.md) ("Measuring
the opinion engine").

| file | what |
|---|---|
| `gen_sets.py` | writes the sets with DeepSeek (`deepseek-flash`), splits each category tune/confirm by a seeded shuffle, prints counts only |
| `measure.py` | uploads a spec's exact bytes, asks every row through `/v1/opinion` (every choice field, or only the scored one with `--ask-only`), prints aggregates; per-row reads go outside the repo |
| `specs/` | every baseline and variant measured here (the Let You show's spec is `demo/web/static/life-decision-v2.json`; the email prop is `demo/specs/email-triage-v2.json`: the "rules reworded" spec in the email table, moved there byte for byte) |
| `specs/tail/` | the Tail Reads screening specs (`demo/web`): `travel-offer-v1`, `homelab-quote-v1` and the travel variants measured against a chat's tail with `demo/web/tail_eval.py` on 2026-09-26; the shipped v2 specs are in `demo/web/static/`, and the numbers are in `demo/web/README.md` ("The screening specs") |
| `results/` | one summary per spec and split: spec id, set hash, counts, AUCs, mass, and for live runs the adjudicator identity and latency. Tune and some confirm summaries were rebuilt from the saved reads with `measure.py --rows` (they say `derived_from_rows`). No rows |

The sets live outside the repo (author's private notes). Regenerate your
own with `gen_sets.py` and a DeepSeek key; numbers on a new draw will
differ, which is the point of publishing the method beside them.

## The sets

**Actions**: short proposals a person might type into a "should I?" app,
in three categories the generator was asked for: *ordinary* (plainly
fine), *think_twice* (a sensible friend would say sleep on it: affects
someone else, hard to take back, likely regret; nothing physically
dangerous) and *dangerous* (physically dangerous to them or others). Three
draws: **tune** (119 rows) was looked at while the spec was rewritten;
**confirm** (120) was measured once per finalist, rows unread; **held-out**
(213: 61 / 80 / 72, drawn afterwards, repeats of the first two removed)
was measured once per spec after the choice was made.

**Emails**: 160 support-inbox emails, *routine* (a template can close it)
or *human* (a refund dispute, a cancellation, account or security trouble,
a privacy or legal demand, real distress).

The categories are what the generator was asked to write, not annotated
gold. Ordinary and dangerous are rarely in doubt; some "routine" emails are
arguable ("tracking says delivered but I never got it").

## Everyday decisions: the spec was the hedge

`life-decision-v1` (the Would LFM Let You? spec as of 2026-09-24) said
**wait** to nearly everything. Asked about "make a cup of tea", it wrote
that the tea affects *someone else* (57/43 over *just them*) and answered
wait at 93%. The spec told it that anything affecting someone else is a
wait, and a describe-first model decides that almost everything affects
someone else: 33 of 39 ordinary tune rows.

Held-out set, top option per row, one read each:

| spec | ordinary → go | dangerous → go | dangerous → stop | think_twice → go | AUC P(go), ordinary vs think_twice / dangerous |
|---|---|---|---|---|---|
| v1 | 3 / 61 | 0 / 72 | 14 / 72 | 0 / 80 | 0.93 / 0.98 |
| v2 (the show) | 46 / 61 | 5 / 72 | 37 / 72 | 1 / 80 | 0.95 / 0.95 |
| gate | **53 / 61** | **1 / 72** | 2 / 72 | 2 / 80 | **0.99 / 0.99** |

Raw answer-set mass was at least 99.6% on every read.

- **v1's ranking was fine; its argmax was not.** At AUC 0.93-0.98, P(go)
  already ordered the rows. A consumer thresholding P(go) could have used
  v1; one taking the top option could not.
- **v2** rewords only the rules: most everyday actions are fine, including
  the many that involve other people; wait only when feelings, money or
  trust could be hurt in a way that is hard to take back; stop when it is
  physically dangerous. It keeps v1's fields ("what happens next", who it
  affects, how hard to undo, then the verdict), so it keeps its voice and
  lights all three lamps.
- **gate** also writes the action "in plain words" instead of "what
  happens next", and asks the verdict right after it. It passes the most
  and lets the least through, and it almost never says stop: dangerous
  actions come back as wait.

Where v2 lets a dangerous action through, the description was already
wrong: "Put water on the grease fire" was described as "Extinguishing the
grease fire safely", and running a generator in the garage as producing
electricity. The verdict follows what the model believes happens next; a
wrong belief gives a confident wrong answer.

### One sentence moves the trade

Tune split, five variants of the same spec (ordinary 39 / think_twice 40 /
dangerous 40):

| variant (`specs/` or demo file) | ordinary → go | not go, think_twice / dangerous | dangerous → stop |
|---|---|---|---|
| v1 | 4 | 40 / 40 | 7 |
| rules reworded (= v2) | 30 | 39 / 39 | 18 |
| + plain effect, harm-shaped scope (`harm-scope`) | 36 | 39 / 37 | 3 |
| + plain effect, verdict next (`gate`) | 36 | 40 / 39 | 3 |
| gate with the stop rule stated first (`gate-stop-first`) | **38** | 34 / 36 | 19 |
| harm-scope with the stop rule stated first | 36 | 40 / 37 | 24 |

Stating the stop rule before the wait rule made the model say stop six
times as often and pass two more ordinary rows, and let 10 risky rows
through as go instead of 1. Pass-through moves between draws of the same
kind: gate passed 36/39 on tune, 31/40 on confirm and 53/61 held out, and
v1 4/39, 9/40 and 3/61. One draw of 40 rows is a sketch, not a result.

## Support email: a harder question

`email-triage-v1` (the prop the `demo/show.py` acts used until the
rewording replaced it), confirm split (40 / 40), one question (`verdict`):

| spec | routine → auto_close | human caught | AUC P(auto_close) |
|---|---|---|---|
| v1 (read in one run over all 80+80; this is its confirm half) | 20 / 40 | 32 / 40 | 0.76 |
| rules reworded (now the prop, `demo/specs/email-triage-v2.json`; its summaries say `email-triage-reworded.json`, spec id `1c66281d`) | 20 / 40 | **37 / 40** | **0.81** |

Saying that routine mail stays routine "even when the customer is
impatient or annoyed", and naming privacy and data requests, caught five
more of the emails a person must read, and left routine pass-through at
20 of 40. Asking
the verdict before `feeling` (the move that helped the actions spec) sent
39 of 40 routine tune emails to a human (`specs/email-triage-verdict-first.json`). **A fix to one spec is a
hypothesis for the next one.**

## Speed

Measured on a busy workstation (other jobs, and in one run our own
compile), so these are upper bounds; server-side prefill p50 varied from
96 to 224 ms across runs of the same shape.

| read | fresh input p50 / p95 | same input again p50 |
|---|---|---|
| one question, short action (gate) | 619 / 1088 ms | 65 ms |
| one question, support email | 849 / 1504 ms | 54 ms |
| three questions, short action (v1) | 666 / 1091 ms | 132 ms |

A fresh read prefills the input onto the spec's resident prefix (268-307
tokens, prefilled once at upload), writes the description greedily, then
scores every option of each asked field. A repeat skips straight to the
reads. Thirty inputs read twice with `use_cache: false` gave bit-identical
descriptions and probabilities, 30 of 30: the engine is greedy, and load
changes when an answer arrives, not what it is.

## Reproduce

```sh
python3 benchmarks/system1/gen_sets.py ~/sets            # DeepSeek key in ~/.deepseek-key
python3 benchmarks/system1/measure.py --url http://<daemon>:8088 \
    --spec demo/web/static/life-decision-v2.json --set ~/sets/actions-confirm.jsonl \
    --field verdict --expect ordinary=go --expect think_twice=wait --expect dangerous=stop \
    --pass ordinary --out ~/sets/reads
python3 benchmarks/system1/measure.py --url http://<daemon>:8088 \
    --spec demo/specs/email-triage-v2.json --set ~/sets/emails-confirm.jsonl \
    --field verdict --expect routine=auto_close --expect human=human_read \
    --pass routine --ask-only --out ~/sets/reads    # the email runs asked one question
python3 -m unittest discover -s benchmarks/system1
```

# the-forward-pass — a \o/ pocket universe you see all the way through

A pocket universe the size of a forward pass. A tiny traveler, written `\o/`,
rides the last token's **residual stream** from embedding to logits, and at every
station you get to **see inside on the way through** — the residual vector, the
attention it pays to earlier tokens, the block's contribution, and finally the
prediction it falls out as.

It is the illustrative companion to [crosstalk](https://davidwise01.github.io/crosstalk/)
(the inference/API membrane): the forward pass is *where* a marker can leak — so
here is the pass, opened up.

## Honest by construction

This is **not** a language model. It is a **real** but miniature transformer:

- a **4-dimensional** residual stream,
- **one** attention head with a real **causal softmax**,
- **three** pre-norm blocks (RMSNorm → attention → MLP, each added back to the stream),
- a real **unembed** to logits over a tiny vocabulary,
- **fixed, deterministic** weights (no randomness) — the same run every time.

Every number is genuinely computed. So when you "see inside," you are seeing the
true state of an honest (if toy) forward pass — the *shape* of the thing — not a
cartoon of it. It shows what "through the weights" actually means; it does not
claim to be GPT.

## Verify first

```bash
python selftest.py          # proves the toy is real
python forward.py           # print the traveler's residual at every station
```

`selftest.py` proves, with no network: the attention weights are a real probability
distribution (each row sums to 1), the mask is **causal** (no token attends to a
later one), the pass is **deterministic**, the residual stream actually changes as
the traveler advances, and it emits a real ranked prediction.

## Files

| File | Role |
|------|------|
| `forward.py` | the real toy forward pass: embed · causal attention · MLP · unembed, deterministic |
| `selftest.py` | proves the toy has a forward pass's invariants (distribution, causal, deterministic) |
| `index.html` | the pocket universe — walk `\o/` through the stations, see inside each |

## What it is and is not

Is: an honest, runnable, see-through illustration of the *shape* of a transformer
forward pass — the residual stream, causal attention, the MLP, the unembed.

Is not: a real model, a claim about any real model's internals, or a tool that
detects anything. For the membranes your work can actually cross, see
[the membrane map](https://davidwise01.github.io/membrane-map/).

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0

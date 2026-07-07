#!/usr/bin/env python3
"""Verify-first self-test for the toy forward pass. Proves, with no network, that
the miniature transformer is a REAL computation with the invariants a forward pass
must have -- so "see inside on the way through" shows true state, not a cartoon:
(1) attention weights are a probability distribution (rows sum to 1); (2) the mask
is causal (no token attends to a later one); (3) the pass is deterministic (same
input -> same output every time); (4) the residual stream actually changes as the
traveler moves; (5) it emits a real prediction over the vocab.
"""
from __future__ import annotations
import math
from forward import trace, attention, embed, _layer_weights, softmax, SEQ, D, LAYERS

fails = 0
def check(cond, msg):
    global fails
    print(("ok  · " if cond else "FAIL· ") + msg)
    fails += 0 if cond else 1


# 1. Attention weights are a real probability distribution.
x = [embed(t) for t in SEQ]
_, weights = attention(x, _layer_weights(0))
check(all(abs(sum(row) - 1.0) < 1e-9 for row in weights), "every attention row sums to 1 (a distribution)")
check(all(all(w >= 0 for w in row) for row in weights), "attention weights are non-negative")

# 2. The mask is causal: token i puts ~zero weight on any j > i.
causal_ok = all(weights[i][j] < 1e-6 for i in range(len(weights)) for j in range(len(weights)) if j > i)
check(causal_ok, "causal mask holds -- no token attends to a later one")

# 3. Determinism: the same input yields the same trace every time.
a, b = trace(), trace()
check(a["stations"] == b["stations"] and a["prediction"] == b["prediction"], "the pass is deterministic")

# 4. The residual stream actually moves through the pass.
vecs = [s["vec"] for s in a["stations"]]
moved = any(vecs[i] != vecs[i + 1] for i in range(len(vecs) - 1))
check(moved, "the residual stream changes as the traveler advances")
check(len(a["stations"]) == 2 + 2 * LAYERS, f"one station per stage (embed + {LAYERS}x[attn,mlp] + unembed)")

# 5. A real prediction comes out.
top = a["stations"][-1]["logits"]
check(top and isinstance(top[0][0], str), "unembed emits a ranked prediction over the vocab")
check(a["prediction"] == top[0][0], "the reported prediction is the top logit")

# 6. softmax sanity on a known vector.
sm = softmax([0.0, 0.0])
check(abs(sm[0] - 0.5) < 1e-12, "softmax of equal logits is uniform")

print("\n" + ("SOME CHECKS FAILED" if fails else "all forward-pass checks passed"))
raise SystemExit(1 if fails else 0)

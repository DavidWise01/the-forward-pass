#!/usr/bin/env python3
"""the forward pass — a tiny, REAL, deterministic transformer forward pass you
can see all the way through.

This is the honest core of the pocket universe. It is NOT a real language model:
it is a 4-dimensional residual stream, one attention head, three blocks, over a
five-token sequence, with fixed deterministic weights. But every number is really
computed -- real causal softmax attention, real residual adds, a real MLP, a real
unembed -- so when the page lets you "see inside on the way through", what you see
is the true state of a genuine (if miniature) forward pass, not a cartoon.

A traveler, written \\o/, rides the last token's residual stream from embedding to
logits; `trace()` records its vector at every station so you can watch it change.
"""
from __future__ import annotations
import math

D = 4                                   # residual width
LAYERS = 3
SEQ = ["the", "cat", "sat", "on", "\\o/"]   # the traveler is the last token
VOCAB = ["the", "cat", "sat", "on", "\\o/", "mat", "run", "sky"]


def _grid(rows, cols, salt):
    """Deterministic pseudo-weights in [-1, 1] -- no RNG, fully reproducible."""
    return [[round(math.sin((i + 1) * 2.3 + (j + 1) * 1.7 + salt * 0.9), 4)
             for j in range(cols)] for i in range(rows)]


def _tok_salt(t):
    return sum(ord(c) for c in t) % 11


def embed(tok):
    s = _tok_salt(tok)
    return [round(math.sin((k + 1) * 1.3 + s * 0.7), 4) for k in range(D)]


def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def relu(v):
    return [x if x > 0 else 0.0 for x in v]


def rmsnorm(v):
    """Pre-norm, as in modern transformers -- keeps the residual bounded so you can
    actually see the stream instead of it exploding."""
    ms = sum(x * x for x in v) / len(v)
    scale = 1.0 / math.sqrt(ms + 1e-6)
    return [x * scale for x in v]


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


# fixed weights per layer (deterministic)
def _layer_weights(li):
    return {
        "Wq": _grid(D, D, li * 3 + 1), "Wk": _grid(D, D, li * 3 + 2), "Wv": _grid(D, D, li * 3 + 3),
        "Wo": _grid(D, D, li * 3 + 4),
        "W1": _grid(D * 2, D, li * 3 + 5), "W2": _grid(D, D * 2, li * 3 + 6),
    }


def attention(x, W):
    """Causal single-head attention over the sequence x (list of D-vectors).
    Returns the per-token attention output and the attention weight rows."""
    n = len(x)
    Q = [matvec(W["Wq"], t) for t in x]
    K = [matvec(W["Wk"], t) for t in x]
    V = [matvec(W["Wv"], t) for t in x]
    out, weights = [], []
    for i in range(n):
        scores = [dot(Q[i], K[j]) / math.sqrt(D) if j <= i else -1e9 for j in range(n)]  # causal mask
        w = softmax(scores)
        weights.append(w)                       # full precision (a real distribution)
        ctx = [0.0] * D
        for j in range(n):
            ctx = add(ctx, [w[j] * V[j][k] for k in range(D)])
        out.append(matvec(W["Wo"], ctx))
    return out, weights


def mlp(t, W):
    return matvec(W["W2"], relu(matvec(W["W1"], t)))


def unembed(vec):
    """Dot the final residual with each vocab embedding -> logits -> top token."""
    logits = [(tok, round(dot(vec, embed(tok)), 4)) for tok in VOCAB]
    logits.sort(key=lambda kv: -kv[1])
    return logits


def trace(seq=None):
    """Run the forward pass and record the TRAVELER (last token) at every station."""
    seq = seq or SEQ
    x = [embed(t) for t in seq]
    traveler = len(seq) - 1
    stations = [{"name": "embed", "vec": [round(v, 4) for v in x[traveler]], "attn": None}]
    for li in range(LAYERS):
        W = _layer_weights(li)
        attn_out, weights = attention([rmsnorm(t) for t in x], W)   # pre-norm attention
        x = [add(x[i], attn_out[i]) for i in range(len(x))]         # residual + attention
        stations.append({"name": f"block {li+1} · attention", "vec": [round(v, 4) for v in x[traveler]],
                         "attn": [round(v, 4) for v in weights[traveler]]})
        x = [add(x[i], mlp(rmsnorm(x[i]), W)) for i in range(len(x))]  # pre-norm MLP + residual
        stations.append({"name": f"block {li+1} · MLP", "vec": [round(v, 4) for v in x[traveler]], "attn": None})
    logits = unembed(rmsnorm(x[traveler]))
    stations.append({"name": "unembed · logits", "vec": [round(v, 4) for v in x[traveler]],
                     "attn": None, "logits": logits})
    return {"seq": seq, "traveler": traveler, "stations": stations, "prediction": logits[0][0]}


if __name__ == "__main__":
    t = trace()
    print("sequence:", " ".join(t["seq"]))
    for s in t["stations"]:
        print(f"  {s['name']:22} residual={s['vec']}")
    print("prediction (top token):", t["prediction"])

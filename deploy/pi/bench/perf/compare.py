import json, sys
def load(p):
    return [json.loads(l) for l in open(p)]
a, b = load(sys.argv[1]), load(sys.argv[2])
assert len(a) == len(b), (len(a), len(b))
key = lambda r: {k: v for k, v in r.items() if k not in ("i", "text")}
mismatch = [i for i, (x, y) in enumerate(zip(a, b)) if key(x) != key(y)]
verdict = [i for i, (x, y) in enumerate(zip(a, b)) if (x["risk"], x["action"]) != (y["risk"], y["action"])]
score_diff = max((abs(x["score"] - y["score"]) for x, y in zip(a, b)), default=0.0)
print(json.dumps({"n": len(a), "field_mismatches": len(mismatch), "verdict_flips": len(verdict),
                  "max_abs_score_diff": score_diff, "first_mismatch": mismatch[:5]}))

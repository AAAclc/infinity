"""Map a natural-language instruction to a task_index (0-9).

Training conditions on a learned per-task embedding, so at eval we must map
the harness's instruction string to the same index. Exact match first, then
object-noun match, then fuzzy fallback.
"""
import difflib

TEMPLATES = [
    "pick up the orange juice and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the alphabet soup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
]

OBJECTS = [
    "orange juice", "ketchup", "cream cheese", "bbq sauce", "alphabet soup",
    "milk", "salad dressing", "butter", "tomato sauce", "chocolate pudding",
]


def _norm(s):
    return " ".join(s.lower().strip().split())


def match_instruction(text):
    t = _norm(text)
    norms = [_norm(x) for x in TEMPLATES]
    if t in norms:
        return norms.index(t)
    for i, obj in enumerate(OBJECTS):
        if obj in t:
            return i
    best = difflib.get_close_matches(t, norms, n=1, cutoff=0.4)
    if best:
        return norms.index(best[0])
    raise ValueError("cannot map instruction to a task: %r" % text)

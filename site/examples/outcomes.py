"""Learn from what happened: the world and the policy are written; what to expect
from each action is learned from observed outcomes. No one labels the right action.

A tiny authored world, like the repository's learn_action_outcomes.py example: a
mechanism demonstration, not a benchmark.
Runs offline on CPU:  pip install 'tensorcode[vec]'  then  python outcomes.py
"""
import re
import torch
from tensorcode import trace, training
from tensorcode.ops.vec import CandidateSet, Decide, Score
from tensorcode.ops.vec.encode import VocabularyEncoder

ACTIONS = ["cool", "reindex", "serve"]
FIXES = {("cool", "hot"), ("reindex", "corrupt")}  # written: how this world works

TRAIN = [("hot", "fans at max and cpu at 94c"), ("corrupt", "index checksum mismatch"),
         ("ready", "all health checks passing"), ("hot", "thermal throttling on node 3"),
         ("corrupt", "corrupt pages in the search index"), ("ready", "warm and ready for traffic")]
TEST = [("hot", "node 7 is thermal throttling"), ("corrupt", "checksum mismatch in pages"),
        ("hot", "cpu at 91c and climbing"), ("corrupt", "search index returns corrupt rows"),
        ("hot", "fans at max on node 2"), ("corrupt", "index pages fail checksum")]


def world(status, action):
    """Written environment: (status, action) -> (next status, reward)."""
    if action == "serve":
        return ("serving", 1.0) if status == "ready" else (status, -0.5)
    if (action, status) in FIXES:
        return "ready", 0.5
    return status, -0.5


TELEMETRY = {"ready": "all health checks passing"}
torch.manual_seed(12)
texts = [t for _, t in TRAIN + TEST] + ACTIONS
words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t)})
space = {"name": "telemetry", "dimensions": 16}
encode = VocabularyEncoder({"vocabulary": words, "dimensions": 16, "output_space": space})
expect = Score({"architecture": "mlp", "hidden_dimensions": [16], "query_space": space,
                "candidate_space": space, "meaning": "expected reward of an action"})
decide = Decide()


def options(telemetry):
    return CandidateSet(encode(telemetry), encode(ACTIONS), ACTIONS)


def succeeds(status, telemetry, steps=2):
    for _ in range(steps):
        action = decide(expect(options(telemetry))).identity  # written policy: best expected
        status, _ = world(status, action)
        if status == "serving":
            return True
        telemetry = TELEMETRY.get(status, telemetry)
    return False


def outcome(scores, target):
    index, reward = target
    return (scores.values[index] - reward) ** 2


def evaluate():
    with torch.no_grad():
        return sum(succeeds(status, telemetry) for status, telemetry in TEST)


print(f"before: {evaluate()}/{len(TEST)} test scenarios served")
sessions = []
for status, telemetry in TRAIN:  # explore: try every action once, record what happened
    for index, action in enumerate(ACTIONS):
        _, reward = world(status, action)
        with trace() as t:
            scores = expect(options(telemetry))
        t.supervise(scores, [index, reward], loss="outcome", source=f"observed:{status}:{action}")
        sessions.append(t)

trainer = training.Trainer.from_ops({"encode": encode, "expect": expect}, losses={"outcome": outcome},
                                    optimizer=lambda p: torch.optim.Adam(p, lr=0.01))
losses = trainer.fit(sessions, epochs=20)
print(f"{len(sessions)} observed outcomes, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
print(f"after: {evaluate()}/{len(TEST)} test scenarios served")

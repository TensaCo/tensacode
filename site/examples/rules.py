"""Rules the model can't break: a learned Score ranks actions, a written mask decides
which actions exist right now, and Decide never selects a masked one.

Runs offline on CPU:  pip install 'tensorcode[vec]'  then  python rules.py
"""
import re
import torch
from tensorcode import trace, training
from tensorcode.ops.vec import CandidateSet, Decide, Latent, Score
from tensorcode.ops.vec.encode import VocabularyEncoder

ACTIONS = {
    "restart": "restart the service process",
    "rollback": "roll back the latest deploy",
    "scale": "scale out more instances",
    "page": "page the on call engineer",
}
POSTMORTEMS = [
    ("errors started right after the deploy", "rollback"),
    ("the new release throws exceptions", "rollback"),
    ("memory keeps growing until the process dies", "restart"),
    ("the service hung and stopped responding", "restart"),
    ("traffic spike and requests are queueing", "scale"),
    ("cpu saturated during the sale", "scale"),
    ("the database disk is almost full", "page"),
    ("strange alerts from the payment provider", "page"),
]


def allowed(*, peak_hours, minutes_since_deploy, budget_left):
    return {
        "restart": not peak_hours,              # never at peak
        "rollback": minutes_since_deploy < 60,  # recent deploys only
        "scale": budget_left > 0,               # it costs money
        "page": True,                           # a person, always
    }


torch.manual_seed(5)
texts = [t for t, _ in POSTMORTEMS] + list(ACTIONS.values())
words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]", t)})
space = {"name": "operations text", "dimensions": 24}
encode = VocabularyEncoder({"vocabulary": words, "dimensions": 24, "output_space": space})
score = Score({"architecture": "mlp", "hidden_dimensions": [24], "query_space": space,
               "candidate_space": space, "meaning": "learned fit of an action to an incident"})
decide = Decide()


def options(incident, rules=None):
    actions = encode(list(ACTIONS.values()))
    mask = None if rules is None else torch.tensor([rules[a] for a in ACTIONS])
    return CandidateSet(encode(incident), Latent(actions.tensor, actions.space, mask=mask), list(ACTIONS))


sessions = []
for incident, action in POSTMORTEMS:
    with trace() as t:
        scores = score(options(incident))
    t.supervise(scores, list(ACTIONS).index(action), loss="choice", source="postmortem review")
    sessions.append(t)


def choice(scores, target):
    return torch.nn.functional.cross_entropy(scores.values.unsqueeze(0), torch.tensor([target]))


trainer = training.Trainer.from_ops({"encode": encode, "score": score}, losses={"choice": choice},
                                    optimizer=lambda p: torch.optim.Adam(p, lr=0.01))
losses = trainer.fit(sessions, epochs=25)
print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")

incident = "errors after the deploy an hour ago"
with torch.no_grad():
    for minutes in (20, 90):
        rules = allowed(peak_hours=True, minutes_since_deploy=minutes, budget_left=100)
        scores = score(options(incident, rules))
        values = ", ".join(f"{a} {v:.2f}" for a, v in zip(ACTIONS, scores.values.tolist()))
        print(f"deploy {minutes:>2} min ago: {decide(scores).identity:8} scores: {values}")

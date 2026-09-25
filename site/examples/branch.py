"""Train a branch: replace an if/elif with a Classify that learns from examples.

Runs offline on CPU:  pip install 'tensorcode[vec]'  then  python branch.py
"""
import re
import torch
from tensorcode import trace, training
from tensorcode.ops.vec import Classify
from tensorcode.ops.vec.encode import VocabularyEncoder

EXAMPLES = [
    ("i was charged twice for my order", "billing"),
    ("please refund my last payment", "billing"),
    ("why is there an extra fee on my invoice", "billing"),
    ("my card was billed after i cancelled", "billing"),
    ("i want my money back", "billing"),
    ("the subscription price went up without notice", "billing"),
    ("can i get a receipt for last month", "billing"),
    ("how do i update my payment method", "billing"),
    ("the app crashes when i open it", "technical"),
    ("page will not load on my phone", "technical"),
    ("i get an error when uploading a file", "technical"),
    ("the website is really slow today", "technical"),
    ("sync stopped working after the update", "technical"),
    ("notifications are not arriving", "technical"),
    ("the export button does nothing", "technical"),
    ("screen goes blank after login", "technical"),
    ("i forgot my password", "account"),
    ("how do i change my email address", "account"),
    ("please delete my account", "account"),
    ("i cannot log in to my account", "account"),
    ("someone else is using my account", "account"),
    ("how do i add a team member", "account"),
    ("i want to change my username", "account"),
    ("my two factor code does not work", "account"),
]
HELD_OUT = [
    ("you took money from me twice", "billing"),
    ("cancel my plan and send a refund", "billing"),
    ("the app freezes on startup", "technical"),
    ("upload fails with a weird message", "technical"),
    ("locked out after too many attempts", "account"),
    ("change the email on my account", "account"),
]
LABELS = ["billing", "technical", "account"]


def route_by_hand(ticket):
    if "refund" in ticket or "charged" in ticket:
        return "billing"
    if "error" in ticket or "crash" in ticket:
        return "technical"
    if "password" in ticket or "log in" in ticket:
        return "account"
    return "unknown"


torch.manual_seed(7)
words = sorted({w for text, _ in EXAMPLES for w in re.findall(r"\w+|[^\w\s]", text)})
space = {"name": "tickets", "dimensions": 24}
encode = VocabularyEncoder({"vocabulary": words, "dimensions": 24, "output_space": space})
route = Classify({"architecture": "linear", "input_space": space, "labels": LABELS})

with trace() as t:
    guess = route(encode([text for text, _ in EXAMPLES]))
t.supervise(guess, [label for _, label in EXAMPLES], source="support team")
trainer = training.Trainer.from_ops({"encode": encode, "route": route},
                                    optimizer=lambda p: torch.optim.Adam(p, lr=0.05))
losses = trainer.fit([t], epochs=30)
print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")

with torch.no_grad():
    learned = route(encode([text for text, _ in HELD_OUT]))
for (text, label), p, value in zip(HELD_OUT, learned.probabilities, learned.values):
    print(f"{text!r:40} {label:9} by hand: {route_by_hand(text):9} learned: {value} ({p.max().item():.2f})")

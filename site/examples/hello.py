from tensorcode import trace, training
from tensorcode.ops.vec import Classify
from tensorcode.ops.vec.encode import VocabularyEncoder

tickets = ["refund my order", "charged twice", "the app crashed",
           "error on upload", "forgot my password", "cannot log in"]
labels = ["billing", "billing", "technical", "technical", "account", "account"]
words = sorted({w for t in tickets for w in t.split()})

space = {"name": "tickets", "dimensions": 8}
encode = VocabularyEncoder({"vocabulary": words, "dimensions": 8,
                            "output_space": space})
route = Classify({"architecture": "linear", "input_space": space,
                  "labels": ["billing", "technical", "account"]})

with trace() as t:
    guess = route(encode(tickets))
t.supervise(guess, labels, source="me")
trainer = training.Trainer.from_ops({"encode": encode, "route": route}, lr=0.5)
trainer.fit([t], epochs=50)

print(route(encode("the upload crashed")).value)  # technical

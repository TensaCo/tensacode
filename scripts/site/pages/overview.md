# Architecture overview

TensorCode builds trainable programs out of four kinds of parts. **Operations**
are callable units that turn one value into another. **Tools** compose
operations into complete models that own every parameter. **Tracing and
training** record which operation produced which value, attach reviewed targets
with their source, and fit the parameters. **Artifacts** save a trained operation
or tool as data (JSON configuration and safetensors weights) that reloads in a
fresh process, from the Hugging Face Hub, or in the other language.

This page describes each part and how they connect. The API names are Python's;
TypeScript uses the same names in camelCase (`save_pretrained` is
`savePretrained`). The language guides have the full reference.

| Part | Python | TypeScript | Holds weights |
|---|---|---|---|
| Vector operations | `tensorcode.ops.vec` | `tensorcode/ops/vec` | Learned ones do |
| Text operations | `tensorcode.ops.text` | `tensorcode/ops/text` | Owned ones do; provider-backed ones do not |
| Graph operations | `tensorcode.ops.graph` | `tensorcode/ops/graph` | No (symbolic interfaces) |
| Tools | `tensorcode.tools` | `tensorcode/tools` | Yes |
| Tracing | `tensorcode.trace()` | `trace()` from `tensorcode` | No |
| Training | `tensorcode.training` | `tensorcode/training` | Updates them |
| Providers | `tensorcode.integrations` | `tensorcode/integrations` | No (external models) |

## Operations

Every operation has the same contract: call it with a value and an optional
context, and it returns a value.

```python tab=Python
result = op(value, context=None)
result = await op.acall(value, context=None)   # async form
```

```ts tab=TypeScript
const result = op.call(value, { context: null });
const later = await op.acall(value, { context: null });   // async form
```

A subclass implements `forward(value, context)`. Calling the instance, rather
than `forward`, is what lets an active trace record the call. An operation marks
itself `replayable` only when it has no side effects, because training recomputes
replayable calls.

**Construction is data.** Public constructors take a JSON configuration, never an
executable model. A learned operation creates all its parameters in the
constructor and owns them. Unknown or obsolete configuration fields raise an
error instead of being ignored. Explicit factories cover the other cases:

- `from_foundation(repo, revision=...)` builds the operation around a supported
  pretrained architecture from the Hugging Face Hub. New heads and bridges start
  untrained.
- `from_module(module)` (vector operations) and `from_model(provider)` (text
  operations) wrap an implementation you supply. They can be traced and used,
  but they cannot be saved as an artifact, because TensorCode cannot rebuild what
  it did not construct.

### Vector operations: `ops.vec`

Vector operations work on tensors tagged with what they mean. A `Space` names a
representation (name, dimensions, version, organization). A `Latent` is a tensor
tagged with its `Space`, plus an optional mask, coordinates and source
references. Two spaces are equal only when all their fields are equal: equal
width does not make two independently trained encoders interchangeable.

| Operation | Maps |
|---|---|
| `TextEncoder`, `ImageEncoder` | Text or images to a `Latent`, with an owned Hugging Face transformer |
| `VocabularyEncoder`, `PatchEncoder` | Text (mean-pooled embeddings) or images (spatial patches with coordinates) to a `Latent` |
| `Transform` | `Latent` to `Latent` (linear, MLP or transformer) |
| `Classify` | `Latent` to a `Prediction` over fixed labels |
| `Score` | A `CandidateSet` (query and candidates) to `Scores` with an authored meaning |
| `Decide`, `Retrieve` | `Scores` to the best candidate, or the top k (no parameters) |
| `Decode`, `TextDecoder`, `ImageDecoder` | A `Latent` to a tensor, to text (sequence-to-sequence) or to an image (latent diffusion) |

Learned vector operations also expose `loss(value, targets)` and the bindings the
trainer needs.

### Text operations: `ops.text`

Text operations work on chat `Message` records (roles `system`, `user`,
`assistant` and `tool`, with text and image parts that can carry a source
reference). Each one is either **owned** (a native sequence-to-sequence model
built from configuration or `from_foundation`, trainable and saveable) or
**provider-backed** (`from_model` around an external model such as an
OpenAI-compatible endpoint).

- `Transform` appends an assistant reply. `TextEncoder` and `TextDecoder`
  convert between text and messages.
- `Classify`, `Decide`, `Score` and `Retrieve` return structured results with a
  distribution, a confidence and an explicit `abstained` flag.
- With `decoding='likelihood'`, an owned model scores every alternative in one
  encoder pass and returns a full distribution. Otherwise the model generates
  JSON, which is validated against the operation's schema. Invalid output
  raises `InvalidModelOutput`; nothing is repaired or retried.
- `ask(messages, questions)` answers several structured questions about the
  same messages. A provider that supports it answers them in one request.

### Graph operations: `ops.graph`

`Graph` and `SourceAnchor` are immutable records for nodes, typed edges and their
sources. The graph operations (`Encode`, `Transform`, `Score`, `Retrieve`,
`Decide`, `Classify` and the rest) declare their interfaces only: every call
raises `NotImplementedError`, in both languages.

## Tools

A tool is a complete model with a public interaction contract. It owns its
encoders, a recurrent slot workspace that attends over the evidence, and its
prediction or decoding heads. Constructing a tool initializes every parameter
and downloads nothing, so a new tool has no pretrained competence until you
train it or load an artifact.

| Tool | Input | Returns |
|---|---|---|
| `Investigator` | A question, sourced evidence and supplied hypotheses | A ranking of the hypotheses, with the source-linked evidence and attention. Without hypotheses it can propose and verify its own |
| `Decision` | Same as `Investigator` | The same architecture under its own saved identity |
| `Planner` | A goal, evidence and candidate plans | Predicted outcomes for each plan. It never executes anything; `new_executor(actions=...)` runs a plan only through actions you supply |
| `Chatbot` | Text | A reply from an owned sequence-to-sequence model conditioned on the workspace, with separate sessions |
| `Scene` | A decoded image, a question and candidate descriptions | A ranking of the descriptions; in language mode, an unverified interpretation |

The result of a call is a receipt: every candidate's score and probability, the
evidence with its source IDs, and where the workspace attended. Probabilities are
uncalibrated unless you calibrate them.

**Sessions keep runtime state out of the weights.** `new_session()` and
`new_cognitive_session()` hold evidence, revisions and episodic memory for one
conversation or investigation, and they save to their own files. The records in
`tools.cognition` keep kinds apart: `Evidence` comes from a named source, a
generated `Hypothesis` is never evidence, and an `Assessment` records which model
produced a score. `tools.actions.action_loop(...)` builds a bounded loop over
actions you define and runs nothing on its own.

## Tracing and training

`trace()` records each operation call: its inputs, the call that produced each
input, and its output. From a trace you can supervise an output with a target and
the source of that target (a reviewer, a label file, an observed outcome), save
the result as an **experience** file, and train on it later. The source is
required, so every training target says where it came from.

```python tab=Python
import torch
from tensorcode import trace, training
from tensorcode.ops.vec import Classify, latent_codecs
from tensorcode.ops.vec.encode import VocabularyEncoder

torch.manual_seed(7)
space = {"name": "example.reviewed-text", "dimensions": 8}
operations = {
    "evidence": VocabularyEncoder({"vocabulary": ["database", "network", "refused", "loss"],
                                   "dimensions": 8, "output_space": space}),
    "interpretation": Classify({"architecture": "linear", "input_space": space,
                                "labels": ["database", "network"]}),
}

with trace() as session:
    prediction = operations["interpretation"](operations["evidence"](["database refused", "network loss"]))
session.supervise(prediction, ["database", "network"], source="review:1")

codecs = latent_codecs()
session.save("experience.json", operations=operations, codecs=codecs, release=True)
experience = training.load_experience("experience.json", operations=operations, codecs=codecs)

trainer = training.Trainer.from_ops(operations, lr=0.05)
losses = trainer.fit([experience], epochs=5)
operations["interpretation"].save_pretrained("interpretation")
```

```ts tab=TypeScript
import { trace } from 'tensorcode';
import { manualSeed } from 'tensorcode/nn';
import { Classify, VocabularyEncoder, latentCodecs, type Prediction } from 'tensorcode/ops/vec';
import { Trainer, loadExperience } from 'tensorcode/training';

manualSeed(7);
const space = { name: 'example.reviewed-text', dimensions: 8 };
const operations = {
  evidence: new VocabularyEncoder({ vocabulary: ['database', 'network', 'refused', 'loss'],
                                    dimensions: 8, output_space: space }),
  interpretation: new Classify({ architecture: 'linear', input_space: space,
                                 labels: ['database', 'network'] }),
};

const session = trace();
const prediction = session.run(() =>
  operations.interpretation.call(operations.evidence.call(['database refused', 'network loss'])) as Prediction);
session.supervise(prediction, ['database', 'network'], { source: 'review:1' });

const codecs = latentCodecs();
await session.save('experience.json', { operations, codecs, release: true });
const experience = await loadExperience('experience.json', { operations, codecs });

const trainer = Trainer.fromOps(operations, { lr: 0.05 });
const losses = trainer.fit([experience], { epochs: 5 });
await operations.interpretation.savePretrained('interpretation');
```

There are two trainers:

- `Trainer.from_ops(operations)` trains an explicitly supervised program. The
  experience is bound to operation names, and loading it checks that each named
  operation still has the same configuration.
- `Trainer.from_tool(tool)` trains a tool's declared objective.
  `trainer.capture(inputs, target, source=...)` snapshots one reviewed example
  without taking an optimizer step.

`fit` and `step` replay the recorded calls with gradients and update the
parameters. `save_checkpoint` and `load_checkpoint` save and restore weights,
optimizer state, module modes, the step count, random generator state and your
own progress, so training resumes exactly. `training` also has held-out
calibration utilities (temperature scaling, calibration metrics and threshold
fitting).

Tracing makes supported local tensor paths trainable. It does not make arbitrary
Python or JavaScript, remote model calls or discrete choices differentiable.

## Pretrained artifacts

`save_pretrained(directory)` writes an artifact that is data only:

- `tensorcode_config.json`: the class identity and its complete configuration.
- `model.safetensors`: the weights.
- `README.md`: a model card, plus any assets the model needs, such as tokenizer
  or processor files, with their hashes.

`Cls.from_pretrained(path_or_repo, revision=..., local_files_only=...)` rebuilds
the class from the configuration and loads the weights, locally or from the
Hugging Face Hub. It rejects an artifact whose class, configuration or assets do
not match, and it never executes code from the artifact. `push_to_hub(repo_id)`
publishes one. Artifacts written by Python load in TypeScript and the other way
round. Pin a `revision` for reproducible loading.

Hosted checkpoints, what each was measured on and its limits are listed on the
[pretrained tools](/docs/python/pretrained/) page. They are small experiments:
[validation](/docs/python/validation/) reports that they do not yet show a
consistent benefit from the recurrent workspace.

## Design rules

- Importing the package loads no machine learning framework, reads no files and
  makes no network calls. Heavy modules load on first use.
- Provenance is explicit. Training targets need a source, generated hypotheses
  and plans are never evidence, and probabilities are uncalibrated unless
  calibrated.
- The core has no domain ontology, no hidden policies, no implicit retries and no
  provider fallbacks. Policies and actions live in your code.
- Everything that is saved is data: JSON configuration, safetensors weights and
  JSON experience and session files.

## Next steps

- [Install](/docs/install/) the Python or TypeScript package.
- Python: [quickstart](/docs/python/quickstart/),
  [operations](/docs/python/operations/), [tools](/docs/python/tools/),
  [training](/docs/python/training/).
- TypeScript: [quickstart](/docs/typescript/quickstart/),
  [operations](/docs/typescript/operations/),
  [tools](/docs/typescript/tools/), [training](/docs/typescript/training/),
  [parity with Python](/docs/typescript/parity/).

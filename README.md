# TensorCode

**Website:** [tensorcode.dev](https://tensorcode.dev) ·
**Docs:** [tensorcode.dev/docs](https://tensorcode.dev/docs/) ·
**Python:** [tensacode-py](https://github.com/TensaCo/tensacode-py) ·
**TypeScript:** [tensacode-ts](https://github.com/TensaCo/tensacode-ts)

TensorCode builds trainable programs from callable operations and small tools
that own their models. You compose encoders, scorers and decoders
(`ops.vec`, `ops.text`, `ops.graph`), or use a complete tool such as
`Investigator`, `Planner`, `Decision`, `Chatbot` or `Scene`. You collect reviewed
feedback with explicit provenance, train on it, and save the result as a
data-only artifact (JSON configuration and safetensors weights) that reloads in a
fresh process, from the Hugging Face Hub, or in the other language. Tracing
records which operation produced which value, so supervised local tensor paths
can be replayed and trained.

> **Status: 0.4.0 alpha.** APIs may change between alphas. The published
> checkpoints are small experiments; the measured behavior and its limits are in
> [validation](https://tensorcode.dev/docs/python/validation/).

This is the umbrella repository. Each language implementation is its own
repository, included here as a git submodule under `tensacode/`. This repository
also holds the product site for [tensorcode.dev](https://tensorcode.dev).

## Implementations

| Language | Directory | Package | Status |
|---|---|---|---|
| Python | [`tensacode/python`](https://github.com/TensaCo/tensacode-py) | `tensorcode` 0.4.0a3 (pip) | Implemented. The reference implementation |
| TypeScript | [`tensacode/typescript`](https://github.com/TensaCo/tensacode-ts) | `tensorcode` 0.4.0-alpha.3 (npm) | Implemented. A port that matches Python and shares its file formats ([parity](https://tensorcode.dev/docs/typescript/parity/)) |
| C++ | `tensacode/cpp` | | Not yet implemented |
| C# | `tensacode/csharp` | | Not yet implemented |
| Dart | `tensacode/dart` | | Not yet implemented |
| Java | `tensacode/java` | | Not yet implemented |
| Kotlin | `tensacode/kotlin` | | Not yet implemented |
| Swift | `tensacode/swift` | | Not yet implemented |

## Install

Python 3.11 or newer. The `tensorcode` name on PyPI has only an older 0.1 alpha,
so install 0.4 from GitHub:

```bash
python -m pip install "tensorcode[tools] @ git+https://github.com/TensaCo/tensacode-py"
```

Node.js 20.16 or newer. The package is not on the npm registry yet, so install it
from GitHub:

```bash
npm install github:TensaCo/tensacode-ts
```

The [install guide](https://tensorcode.dev/docs/install/) lists the Python extras
and how to set up each repository for development.

## A first program

This trains an `Investigator` to rank two supplied hypotheses from log evidence,
then saves and reloads it. It runs offline on a CPU in seconds.

```python
import torch
from tensorcode import training
from tensorcode.tools.investigator import Investigator

torch.manual_seed(0)
model = Investigator({"vocabulary": ["database", "network", "connection", "refused", "packet", "loss"],
                      "dimensions": 16, "slots": 2, "steps": 1})
trainer = training.Trainer.from_tool(model, optimizer=torch.optim.AdamW(model.parameters(), lr=0.01))

hypotheses = [{"id": "database", "text": "database connection refused"},
              {"id": "network", "text": "network packet loss"}]

def case(log_line):
    return {"question": "which component failed",
            "evidence": [{"source_id": "log:1", "text": log_line}],
            "hypotheses": hypotheses}

# Reviewed feedback, with explicit provenance, becomes training experience.
experiences = [trainer.capture(case("connection refused"), "database", source="review:1"),
               trainer.capture(case("packet loss"), "network", source="review:2")]
losses = trainer.fit(experiences, epochs=30)

model.save_pretrained("./investigator")
restored = Investigator.from_pretrained("./investigator")
print(restored(case("packet loss"))["selected_id"])  # network
```

Two authored cases show the lifecycle; they do not show that the model can
investigate anything. The same program in TypeScript is in the
[TypeScript README](https://github.com/TensaCo/tensacode-ts#30-second-example),
and the [architecture overview](https://tensorcode.dev/docs/overview/) explains
operations, tools, tracing, training and artifacts.

## Repository layout

| Path | Contents |
|---|---|
| `tensacode/<language>/` | Language implementations (git submodules) |
| `site/` | The tensorcode.dev product site: static marketing pages and assets ([site/README.md](site/README.md)) |
| `scripts/site/build-docs.mjs` | Generates `site/docs/` from the implementations' READMEs and guides, plus the shared pages in `scripts/site/pages/` |
| `scripts/site/check-links.mjs` | Checks every internal link, anchor and code block of the built site |
| `wrangler.jsonc`, `scripts/site/worker.js` | Cloudflare Worker that serves `site/` at tensorcode.dev |
| `assets/`, `docs/`, `examples/`, `planning/` | Early design material from before the current implementations |

Clone with the submodules:

```bash
git clone --recurse-submodules https://github.com/TensaCo/tensacode
```

## Build and deploy the site

```bash
node scripts/site/build-docs.mjs      # writes site/docs/ (git-ignored)
node scripts/site/check-links.mjs     # internal links, anchors and code blocks
npx wrangler dev                      # preview at http://localhost:8787
npx wrangler deploy                   # publish to tensorcode.dev
```

## License

MIT. Each implementation carries its own license file.

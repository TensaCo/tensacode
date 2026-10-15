# Install

TensorCode has two implementations with the same contracts: the Python package
`tensorcode` (the reference) and its TypeScript port. Artifacts, experience
files and session files move between them, so you can train in one language and
serve in the other.

| | Python | TypeScript |
|---|---|---|
| Package | `tensorcode` | `tensorcode` |
| Runtime | Python 3.11 or newer | Node.js 20.16 or newer, ESM only |
| Numerics | PyTorch, CPU or GPU | Built-in CPU autograd; separate restricted program WebGPU backend |
| Source | [TensaCo/tensacode-py](https://github.com/TensaCo/tensacode-py) | [TensaCo/tensacode-ts](https://github.com/TensaCo/tensacode-ts) |

## Python (pip)

Python 3.11 or newer:

```bash
python -m pip install 'tensorcode[tools]'
```

The core package has no dependencies, and importing it loads neither PyTorch
nor the network. Choose the extras for the parts you use:

| Extra | Adds |
|---|---|
| `tools` | Owned models, training and Hugging Face loading (PyTorch, Transformers) |
| `vec` | Vector operations and compiled-program training (PyTorch, NumPy, safetensors) |
| `local` | Adapter for a local multimodal Transformers model |
| `diffusion` | `tools` plus diffusers, for image decoders |
| `pretrained` | Alias of `tools` |
| `dev` | pytest, build, Pillow and PyArrow. The full test suite also needs `diffusion` |

Check the install:

```bash
python -c "import tensorcode; print(tensorcode.__version__)"
```

To work on the library itself, clone it and install it in editable mode:

```bash
git clone https://github.com/TensaCo/tensacode-py
cd tensacode-py
python -m pip install -e '.[tools,diffusion,dev]'
python -m pytest -q
```

## TypeScript (npm)

Node.js 20.16 or newer:

```bash
npm install tensorcode
```

The package is ESM only (`import`, not `require`) and ships its own type
declarations. Source compilation depends on `typescript`, installed automatically by npm. The optional peer
`@huggingface/transformers` is needed only for `integrations.LocalModel`.

Check the install:

```bash
node --input-type=module -e "import { version } from 'tensorcode'; console.log(version)"
```

To work on the library itself:

```bash
git clone https://github.com/TensaCo/tensacode-ts
cd tensacode-ts
npm install          # also builds dist/
npm test             # program interop tests need python3 and ../python/src
```

## Imports side by side

The entry points match module for module. TypeScript uses camelCase for the API
and keeps Python's snake_case for saved and reported data.

```python tab=Python
from tensorcode import trace, training
from tensorcode.ops.vec import Classify
from tensorcode.ops.vec.encode import VocabularyEncoder
from tensorcode.tools.investigator import Investigator
```

```ts tab=TypeScript
import { trace } from 'tensorcode';
import { Trainer, loadExperience } from 'tensorcode/training';
import { Classify, VocabularyEncoder } from 'tensorcode/ops/vec';
import { Investigator } from 'tensorcode/tools';
```

## Compiler development setup

The compiler guides describe the current SDK source, not a guarantee about the
latest registry release. Use the editable Python checkout or built TypeScript
checkout above. Python's compiler and reference runtime need no extras; training
compiled parameters needs `vec`. For the TypeScript interoperability tests, place
the Python checkout at `../python`, as in the umbrella repository layout.

Start with compiled programs in [Python](/docs/python/program/) or
[TypeScript](/docs/typescript/program/). The parser-free browser core is available
from `tensorcode/program/runtime`; the event Runtime class uses the full
`tensorcode/program` entry.

## Next steps

- [Architecture overview](/docs/overview/): how operations, tools, tracing,
  training and artifacts fit together.
- Quickstart: [Python](/docs/python/quickstart/) or
  [TypeScript](/docs/typescript/quickstart/).
- [TypeScript parity with Python](/docs/typescript/parity/).

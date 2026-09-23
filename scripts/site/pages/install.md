# Install

TensorCode has two implementations with the same contracts: the Python package
`tensorcode` (the reference) and its TypeScript port. Artifacts, experience
files and session files move between them, so you can train in one language and
serve in the other.

| | Python | TypeScript |
|---|---|---|
| Package | `tensorcode` 0.4.0a3 | `tensorcode` 0.4.0-alpha.3 |
| Runtime | Python 3.11 or newer | Node.js 20.16 or newer, ESM only |
| Numerics | PyTorch, CPU or GPU | Built-in autograd core, CPU, no runtime dependencies |
| Source | [TensaCo/tensacode-py](https://github.com/TensaCo/tensacode-py) | [TensaCo/tensacode-ts](https://github.com/TensaCo/tensacode-ts) |

## Python (pip)

Install 0.4 from GitHub. The name `tensorcode` on PyPI currently has only an
older 0.1 alpha, so `pip install tensorcode` does not give you this version yet.

```bash
python -m pip install "tensorcode[tools] @ git+https://github.com/TensaCo/tensacode-py"
```

The core package has no dependencies, and importing it loads neither PyTorch
nor the network. Choose the extras for the parts you use:

| Extra | Adds |
|---|---|
| `tools` | Owned models, training and Hugging Face loading (PyTorch, Transformers) |
| `vec` | Vector operations only (PyTorch, NumPy, safetensors) |
| `local` | Adapter for a local multimodal Transformers model |
| `diffusion` | `tools` plus diffusers, for image decoders |
| `pretrained` | Alias of `tools` |
| `dev` | pytest, build, Pillow and PyArrow. The full test suite also needs `diffusion` |

Check the install:

```bash
python -c "import tensorcode; print(tensorcode.__version__)"   # 0.4.0a3
```

To work on the library itself, clone it and install it in editable mode:

```bash
git clone https://github.com/TensaCo/tensacode-py
cd tensacode-py
python -m pip install -e '.[tools,diffusion,dev]'
python -m pytest -q
```

## TypeScript (npm)

Install from GitHub with npm. The package is not on the npm registry yet. Its
`prepare` script builds `dist/` during the install.

```bash
npm install github:TensaCo/tensacode-ts
```

The package is ESM only (`import`, not `require`) and ships its own type
declarations. It has no runtime dependencies. The optional peer
`@huggingface/transformers` is needed only for `integrations.LocalModel`.

Check the install:

```bash
node --input-type=module -e "import { version } from 'tensorcode'; console.log(version)"   # 0.4.0-alpha.3
```

To work on the library itself:

```bash
git clone https://github.com/TensaCo/tensacode-ts
cd tensacode-ts
npm install          # also builds dist/
npm test             # needs neither Python nor network access
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

## Next steps

- [Architecture overview](/docs/overview/): how operations, tools, tracing,
  training and artifacts fit together.
- Quickstart: [Python](/docs/python/quickstart/) or
  [TypeScript](/docs/typescript/quickstart/).
- [TypeScript parity with Python](/docs/typescript/parity/).

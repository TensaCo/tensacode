#!/usr/bin/env node
// Checks that the landing page's live demos compute what the Python programs they show
// compute. Each site/examples/<name>.py has a twin, site/assets/programs/<name>.js, that
// runs the same program on the bundled TensorCode for TypeScript. Both print a report;
// the reports must match line for line.
//
//   PYTHON=tensacode/python/.venv/bin/python node scripts/site/check-examples.mjs [name ...]
//
// PYTHON needs `tensorcode[vec]` installed. Without PYTHON only the JavaScript side runs.
// Words must match exactly. Numbers may differ by float32 rounding that training
// compounds (the TypeScript port agrees with PyTorch within float tolerance): up to 1%
// or 0.05, whichever is larger. Exits 1 on a mismatch.

import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const python = process.env.PYTHON && (process.env.PYTHON.includes('/') ? resolve(process.env.PYTHON) : process.env.PYTHON);
const wanted = process.argv.slice(2);
const names = readdirSync(join(root, 'site/examples'))
  .filter((file) => file.endsWith('.py')).map((file) => file.slice(0, -3))
  .filter((name) => !wanted.length || wanted.includes(name));

const number = /-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi;
function close(a, b) {
  if (a.replace(number, '#') !== b.replace(number, '#')) return false;
  const x = a.match(number) ?? [], y = b.match(number) ?? [];
  return x.every((v, i) => Math.abs(v - y[i]) <= Math.max(0.05, 0.01 * Math.abs(v)));
}

let failed = 0;
for (const name of names) {
  const program = join(root, 'site/assets/programs', `${name}.js`);
  if (!existsSync(program)) { console.log(`${name}: no JavaScript twin`); failed += 1; continue; }
  const js = [];
  await (await import(pathToFileURL(program).href)).main((line) => js.push(line));
  if (!python) { console.log(`${name} (JavaScript only)\n  ${js.join('\n  ')}`); continue; }
  const py = execFileSync(python, [join(root, 'site/examples', `${name}.py`)], { encoding: 'utf8', cwd: join(root, 'site/examples') })
    .trimEnd().split('\n');
  const exact = py.length === js.length && py.every((line, i) => line === js[i]);
  const same = exact || (py.length === js.length && py.every((line, i) => close(line, js[i])));
  console.log(`${same ? 'ok  ' : 'DIFF'} ${name}${same && !exact ? ' (numbers within float tolerance)' : ''}`);
  if (!same) {
    failed += 1;
    for (let i = 0; i < Math.max(py.length, js.length); i += 1) {
      if (py[i] !== js[i]) console.log(`  py: ${py[i] ?? ''}\n  js: ${js[i] ?? ''}`);
    }
  }
}
process.exit(failed ? 1 : 0);

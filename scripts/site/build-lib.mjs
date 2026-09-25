#!/usr/bin/env node
// Bundles the TensorCode TypeScript library (the tensacode/typescript submodule) into one
// browser module, site/assets/lib/tensorcode.js. Every live demo on the landing page
// imports it: the demos run TensorCode itself, not a re-implementation.
//
//   (cd tensacode/typescript && npm install)   # once: builds dist/ and installs rolldown
//   node scripts/site/build-lib.mjs
//
// The output is committed, so the site build and CI need neither npm nor the submodule's
// dependencies. Rebuild it when the typescript submodule pointer moves.
//
// The library performs file and network I/O only through node: built-ins, which the
// demos never reach. They are replaced by stubs that throw if called. Without
// node:worker_threads and AsyncLocalStorage the library runs its WebAssembly kernels on
// one thread and keeps trace context on a synchronous stack, as it documents.

import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const ts = join(root, 'tensacode/typescript');
const dist = join(ts, 'dist');
const out = join(root, 'site/assets/lib/tensorcode.js');

const { rolldown } = await import(pathToFileURL(join(ts, 'node_modules/rolldown/dist/index.mjs')).href)
  .catch(() => { throw new Error('rolldown not found: run `npm install` in tensacode/typescript first'); });
const version = JSON.parse(readFileSync(join(ts, 'package.json'), 'utf8')).version;

// The public surface the demos use.
const entry = `
export { trace } from ${JSON.stringify(join(dist, 'index.js'))};
export { manualSeed, noGrad, tensor, zerosLike, where, Adam, SGD } from ${JSON.stringify(join(dist, 'nn/index.js'))};
export { crossEntropy } from ${JSON.stringify(join(dist, 'nn/functional.js'))};
export { Space, Latent, CandidateSet, VocabularyEncoder, Classify, Score, Decide, Decode } from ${JSON.stringify(join(dist, 'ops/vec/index.js'))};
export { Operation, ModuleOperation } from ${JSON.stringify(join(dist, 'ops/index.js'))};
export { Trainer } from ${JSON.stringify(join(dist, 'training/index.js'))};
export const version = ${JSON.stringify(version)};
`;

// Every name imported from a node: built-in anywhere in dist/, stubbed.
const walk = (dir) => readdirSync(dir).flatMap((name) => {
  const path = join(dir, name);
  return statSync(path).isDirectory() ? walk(path) : path.endsWith('.js') ? [path] : [];
});
const names = new Set();
for (const file of walk(dist)) {
  for (const [, list] of readFileSync(file, 'utf8').matchAll(/import\s*\{([^}]+)\}\s*from\s*'node:[^']+'/g)) {
    for (const part of list.split(',')) {
      const name = part.replace(/\s+as\s+.*/, '').replace(/^\s*type\s+/, '').trim();
      if (name) names.add(name);
    }
  }
}
const stub = [
  "const unavailable = () => { throw new Error('Not available in the browser build of TensorCode'); };",
  'export default new Proxy({}, { get: () => unavailable });',
  ...[...names].sort().map((name) => `export const ${name} = unavailable;`),
].join('\n');

const bundle = await rolldown({
  input: 'tensorcode-browser-entry',
  platform: 'browser',
  plugins: [{
    name: 'tensorcode-browser',
    resolveId(id) {
      if (id === 'tensorcode-browser-entry') return id;
      if (id.startsWith('node:')) return '\0node-stub';
      return null;
    },
    load(id) {
      if (id === 'tensorcode-browser-entry') return entry;
      if (id === '\0node-stub') return stub;
      return null;
    },
  }],
});
const { output } = await bundle.generate({ format: 'esm', minify: true, codeSplitting: false });
const banner = `/*! TensorCode ${version} for TypeScript (MIT), bundled for the browser by scripts/site/build-lib.mjs. Source: https://github.com/TensaCo/tensacode-ts */\n`;
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, banner + output[0].code);
console.log(`wrote ${out.slice(root.length + 1)} (TensorCode ${version}, ${(statSync(out).size / 1024).toFixed(0)} KiB)`);

// The landing page. Each section initializes when it's about to be seen, and every
// demo imports site/assets/lib/tensorcode.js: TensorCode itself, running in this tab.
import { render, whenVisible } from './code.js';

const lazy = (selector, load, margin = '300px 0px') => {
  const el = document.querySelector(selector);
  if (!el) return;
  whenVisible(el, (visible) => { if (visible) load().then((m) => m.init(el)); }, { margin });
};

// Static code blocks get their gutter and typographic highlighting right away.
for (const pre of document.querySelectorAll('pre.code:not(.live)')) {
  if (!pre.closest('.demo')) render(pre);
}

import('./hero.js').then((m) => m.init(document.querySelector('#hero')));
lazy('#explode', () => import('./explode.js'), '0px');
lazy('#extrap', () => import('./extrap.js'), '0px');
lazy('#mover', () => import('./mover.js'));
lazy('#demo-branch', () => import('./demo-branch.js'));
lazy('#demo-rules', () => import('./demo-rules.js'));
lazy('#demo-physics', () => import('./demo-physics.js'));
lazy('#demo-outcomes', () => import('./demo-outcomes.js'));

// Install commands copy themselves.
for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const note = button.querySelector('.copied');
    try { await navigator.clipboard.writeText(button.dataset.copy); note.textContent = 'copied'; } catch { note.textContent = 'select and copy'; }
    setTimeout(() => { note.textContent = ''; }, 1600);
  });
}

// Run hello.py's twin here.
const runButton = document.querySelector('#hello-run');
const out = document.querySelector('#hello-out');
runButton?.addEventListener('click', async () => {
  runButton.disabled = true;
  out.textContent = '$ python hello.py\n';
  const t0 = performance.now();
  const [{ main }, { version }] = await Promise.all([import('../programs/hello.js'), import('../lib/tensorcode.js')]);
  const printed = [];
  await main((line) => printed.push(line));
  const ms = performance.now() - t0;
  out.innerHTML = '';
  out.append('$ python hello.py\n');
  const b = document.createElement('b');
  b.textContent = printed.join('\n');
  out.append(b, `\n\nRan as TensorCode ${version} for TypeScript, in this tab, in ${ms.toFixed(0)} ms.\nUnseeded, so every run trains from new random weights.`);
  runButton.disabled = false;
  runButton.textContent = 'Run it again ▸';
});

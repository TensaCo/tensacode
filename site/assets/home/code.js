// Code as material: a small Python highlighter, line rendering with the phase gutter,
// the melt / condense / type transitions, and superposed values. See ../../DESIGN.md.

export const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, reduced ? 0 : ms));
export const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

const KEYWORDS = new Set(['def', 'return', 'if', 'elif', 'else', 'for', 'in', 'not', 'and', 'or', 'with', 'as',
  'class', 'import', 'from', 'True', 'False', 'None', 'lambda', 'assert', 'raise', 'while', 'is', 'pass']);
const TOKEN = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)|(\s+)|([^\w\s])/gy;
const escape = (text) => text.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);

/** One line of Python as typographic HTML. `learned` names get the dotted underline. */
export function highlight(line, learned = new Set()) {
  let html = '', previous = '', match;
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < line.length && (match = TOKEN.exec(line))) {
    const [text, comment, string, number, name, space] = match;
    if (comment) html += `<span class="c">${escape(text)}</span>`;
    else if (string) html += `<span class="s">${escape(text)}</span>`;
    else if (number) html += `<span class="n">${text}</span>`;
    else if (name) {
      if (KEYWORDS.has(name)) html += `<span class="k">${name}</span>`;
      else if (previous === 'def' || previous === 'class') html += `<span class="f">${name}</span>`;
      else if (learned.has(name)) html += `<span class="l">${name}</span>`;
      else html += name;
    } else if (space) html += text;
    else html += `<span class="p">${escape(text)}</span>`;
    if (!space) previous = name ?? text;
  }
  return html;
}

/** Which phase a line belongs to, from what it does. */
export function phaseOf(text, learned) {
  const code = text.replace(/#.*$/, '').trim();
  if (!code) return '';
  if (/^(from|import)\b/.test(code)) return 'written';
  if (/\.supervise\(|\.fit\(|\bTrainer\b/.test(code)) return 'signal';
  for (const name of learned) if (new RegExp(`\\b${name}\\b`).test(code)) return 'learned';
  return 'written';
}

export function lineElement(text, { learned = new Set(), phase, n } = {}) {
  const el = document.createElement('span');
  el.className = 'ln';
  el.dataset.n = n ?? '';
  el.dataset.phase = phase ?? phaseOf(text, learned);
  el.innerHTML = highlight(text, learned) || ' ';
  return el;
}

/** Replace a <pre class="code"> with numbered, phase-marked lines. */
export function render(pre, text = pre.textContent, options = {}) {
  const learned = options.learned ?? new Set((pre.dataset.learned ?? '').split(/\s+/).filter(Boolean));
  const code = pre.querySelector('code') ?? pre;
  code.textContent = '';
  const past = pre.classList.contains('past');
  // A statement spans lines while brackets are open; its lines share one phase.
  let depth = 0, statement = [];
  const flush = () => {
    const phases = statement.map((line) => phaseOf(line.text, learned));
    const phase = phases.includes('signal') ? 'signal' : phases.includes('learned') ? 'learned' : phases.find(Boolean) ?? '';
    for (const line of statement) line.el.dataset.phase = past && line.text.trim() ? 'written' : line.text.trim() ? phase : '';
    statement = [];
  };
  text.replace(/\n$/, '').split('\n').forEach((line, i) => {
    const el = lineElement(line, { learned, n: i + 1 });
    code.appendChild(el);
    statement.push({ el, text: line });
    for (const ch of line.replace(/#.*$/, '').replace(/"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/g, '')) {
      if ('([{'.includes(ch)) depth += 1; else if (')]}'.includes(ch)) depth = Math.max(0, depth - 1);
    }
    if (depth === 0) flush();
  });
  flush();
  return [...code.children];
}

export function renumber(pre) {
  [...(pre.querySelector('code') ?? pre).children].forEach((el, i) => { el.dataset.n = i + 1; });
}

/** Solid → liquid → gone: the given lines melt and are removed. */
export async function melt(lines) {
  if (!lines.length) return;
  for (const el of lines) { el.classList.add('phase'); el.style.setProperty('--u', '0'); }
  await frame();
  for (const el of lines) { el.style.transition = '--u .52s var(--ease-liquid), opacity .52s ease'; el.style.setProperty('--u', '1'); el.style.opacity = '0'; }
  await sleep(520);
  for (const el of lines) el.remove();
}

/** Liquid → set: a new learned line condenses out of a superposition. */
export async function condense(el, { settle = 0 } = {}) {
  el.classList.add('phase');
  el.style.setProperty('--u', '1');
  el.style.opacity = '0';
  await frame(); await frame();
  el.style.transition = '--u .64s var(--ease-liquid), opacity .4s ease';
  el.style.setProperty('--u', String(settle));
  el.style.opacity = '1';
  await sleep(640);
}

/** Written text appears the way it's written: one character per frame, then it's set. */
export async function type(el, html, text) {
  if (reduced) { el.innerHTML = html; return; }
  for (let i = 1; i <= text.length; i += 1) {
    el.textContent = text.slice(0, i);
    await sleep(16);
  }
  el.innerHTML = html;
}

/** A superposed value: every label stacked, opacity = probability, uncertainty = blur. */
export function superpose(el, labels, probabilities) {
  if (!el.classList.contains('sup')) { el.textContent = ''; el.classList.add('sup'); }
  let spans = [...el.children];
  if (spans.length !== labels.length || spans.some((s, i) => s.textContent !== labels[i])) {
    el.textContent = '';
    spans = labels.map((label) => { const s = document.createElement('span'); s.textContent = label; el.appendChild(s); return s; });
  }
  const top = Math.max(...probabilities);
  const spread = 1 - top;
  labels.forEach((label, i) => {
    const p = probabilities[i];
    const s = spans[i];
    s.style.opacity = p < 0.02 ? '0' : String(p ** 1.15);
    s.style.filter = p === top && top > 0.97 ? 'none' : `blur(${((1 - p) * 2.4).toFixed(2)}px)`;
    s.style.transform = `translate(${((i - (labels.length - 1) / 2) * spread * 7).toFixed(2)}px, ${((i % 2 ? 1 : -1) * spread * 2.5).toFixed(2)}px)`;
  });
}

/** A single set value (the written phase of a value). */
export function solidify(el, text) {
  el.classList.remove('sup');
  el.textContent = text;
}

export function whenVisible(el, callback, { once = true, margin = '0px 0px -15% 0px' } = {}) {
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      callback(entry.isIntersecting);
      if (entry.isIntersecting && once) io.disconnect();
    }
  }, { rootMargin: margin });
  io.observe(el);
  return io;
}

/** Paper, ink and signal colours for canvas drawing, read from the live theme. */
export function palette() {
  const style = getComputedStyle(document.documentElement);
  const rgb = style.getPropertyValue('--ink-rgb').trim().split(/\s+/).join(',');
  return {
    ink: `rgb(${rgb})`, inkA: (a) => `rgba(${rgb},${a})`,
    paper: style.getPropertyValue('--paper').trim(),
    signal: style.getPropertyValue('--signal').trim(),
    faint: style.getPropertyValue('--faint').trim(),
    rule: style.getPropertyValue('--rule').trim(),
    mono: '"IBM Plex Mono", ui-monospace, monospace',
  };
}

/** A halftone fill for canvas: ink dots on a grid, alpha scaled by `density`. */
export function dotPattern(ctx, ink, size = 4, radius = 0.95) {
  const tile = document.createElement('canvas');
  tile.width = tile.height = size;
  const t = tile.getContext('2d');
  t.fillStyle = ink;
  t.beginPath(); t.arc(size / 2, size / 2, radius, 0, Math.PI * 2); t.fill();
  return ctx.createPattern(tile, 'repeat');
}

/** Size a canvas's backing store to its CSS box and the device pixel ratio. */
export function fitCanvas(canvas, aspect) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth || canvas.width;
  const height = Math.round(width * aspect);
  if (canvas.width !== Math.round(width * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

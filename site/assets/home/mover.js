// 03: move the line. The same route() at four points between written and learned,
// on six tickets the learned part never trained on. Outputs are computed here.
import { lineElement, melt, condense, type, highlight, whenVisible } from './code.js';

const LEARNED = new Set(['classify', 'encode']);
const L = '    return classify(encode(text)).value';
const RULES = [
  '    if "refund" in text or "charged" in text: return "billing"',
  '    if "error" in text or "crash" in text: return "technical"',
  '    if "password" in text or "log in" in text: return "account"',
];
const STOPS = [
  { label: 'all written', lines: ['def route(text):', ...RULES, '    return "unknown"'] },
  { label: 'written first, learned for the rest', lines: ['def route(text):', ...RULES, L] },
  { label: 'one written rule, then learned', lines: ['def route(text):', RULES[0], L] },
  { label: 'all learned', lines: ['def route(text):', L] },
];

export async function init(section) {
  const range = section.querySelector('#mover-range');
  const pre = section.querySelector('#mover-code');
  const code = pre.querySelector('code');
  const body = section.querySelector('#mover-table tbody');
  const score = section.querySelector('#mover-score');
  const { build, HELD_OUT, routeByHand } = await import('../programs/branch.js');
  const model = build();
  for (let epoch = 0; epoch < 30; epoch += 1) model.epoch();
  const learned = model.predict(HELD_OUT.map(([text]) => text)).labels;

  const answer = (stop, text, i) => {
    const hand = routeByHand(text);
    if (stop === 0) return { value: hand, by: 'written' };
    if (stop === 1 && hand !== 'unknown') return { value: hand, by: 'written' };
    if (stop === 2 && (text.includes('refund') || text.includes('charged'))) return { value: 'billing', by: 'written' };
    return { value: learned[i], by: 'learned' };
  };

  let current = 0, busy = Promise.resolve();
  const draw = (lines) => { code.textContent = ''; lines.forEach((t, i) => code.appendChild(lineElement(t, { learned: LEARNED, n: i + 1 }))); };

  async function morph(to) {
    const now = [...code.children];
    const target = STOPS[to].lines;
    const keep = new Set(target);
    await melt(now.filter((el) => !keep.has(el.textContent)));
    const have = new Map([...code.children].map((el) => [el.textContent, el]));
    const fresh = [];
    target.forEach((text, i) => {
      let el = have.get(text);
      if (!el) { el = lineElement('', { learned: LEARNED, n: i + 1 }); el.dataset.phase = text === L ? 'learned' : 'written'; fresh.push([el, text]); }
      code.appendChild(el); // appending in order also reorders
      el.dataset.n = i + 1;
    });
    await Promise.all(fresh.map(([el, text]) => {
      if (text === L) { el.innerHTML = highlight(text, LEARNED); return condense(el); }
      return type(el, highlight(text, LEARNED), text);
    }));
  }

  function table(stop) {
    body.textContent = '';
    let right = 0;
    HELD_OUT.forEach(([text, truth], i) => {
      const { value, by } = answer(stop, text, i);
      const ok = value === truth;
      right += ok;
      const tr = document.createElement('tr');
      tr.innerHTML = '<td></td><td><span></span></td>';
      tr.cells[0].textContent = text;
      tr.cells[1].firstChild.textContent = value;
      tr.cells[1].classList.add(by === 'learned' ? 'by-learned' : 'by-written');
      if (!ok) tr.cells[1].classList.add('wrong');
      tr.cells[1].title = `${by === 'learned' ? 'answered by the learned call' : 'answered by a written rule'}; ${ok ? 'right' : `should be ${truth}`}`;
      body.appendChild(tr);
    });
    score.textContent = `${right} / ${HELD_OUT.length}`;
  }

  function set(stop) {
    range.style.setProperty('--t', String(stop / 3));
    range.setAttribute('aria-valuetext', STOPS[stop].label);
    table(stop);
    if (stop === current) return;
    current = stop;
    busy = busy.then(() => morph(stop));
  }

  draw(STOPS[0].lines);
  table(0);
  range.style.setProperty('--t', '0');
  range.addEventListener('input', () => set(Number(range.value)));
}

export function lazy(section) {
  whenVisible(section, (visible) => { if (visible) init(section); }, { margin: '200px 0px' });
}

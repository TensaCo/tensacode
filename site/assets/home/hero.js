// The hero: one function, route(), whose body moves across the line.
//   written  → rules, some tickets fall through to "unknown"
//   trained  → the body becomes a learned call and really trains here (TensorCode, in this tab)
//   written + trained → a written rule goes back in front of the learned call
// The headline follows: "software" melts when it's trained; "AI" sets when it's programmed.
import { highlight, lineElement, melt, condense, type, superpose, solidify, sleep, frame, reduced, whenVisible } from './code.js';

const LEARNED = new Set(['classify', 'encode']);
const STAGES = [
  ['def route(text):',
    '    if "refund" in text: return "billing"',
    '    if "crash" in text:  return "technical"',
    '    return "unknown"'],
  ['def route(text):',
    '    return classify(encode(text)).value'],
  ['def route(text):',
    '    if "lawyer" in text: return "legal"',
    '    return classify(encode(text)).value'],
];
const TICKETS = ['give me my money back', 'the app crashes on login', 'forgot my password again'];
const LAWYER = 'my lawyer says you charged me twice';
const LABELS = ['billing', 'technical', 'account'];

function byHand(text) {
  if (text.includes('refund')) return 'billing';
  if (text.includes('crash')) return 'technical';
  return 'unknown';
}

export function init(root) {
  const pre = root.querySelector('#hero-code');
  const code = pre.querySelector('code');
  const io = root.querySelector('#hero-io');
  const meter = root.querySelector('#hero-meter');
  const steps = [...root.querySelectorAll('.step')];
  const hl = document.querySelector('.hl');
  const ai = hl.querySelector('.w-ai');
  const software = hl.querySelector('.w-sw');
  let run = 0; // a new run cancels the previous one
  let visible = true;
  let programs = null;
  const programLoad = () => (programs ??= import('../programs/branch.js'));

  const setStep = (i) => steps.forEach((s, j) => (i === j ? s.setAttribute('aria-current', 'step') : s.removeAttribute('aria-current')));
  const rows = () => [...io.children];
  const outputOf = (li) => li.querySelector('output');

  function draw(stage) {
    code.textContent = '';
    STAGES[stage].forEach((text, i) => code.appendChild(lineElement(text, { learned: LEARNED, n: i + 1 })));
  }
  function setRows(texts) {
    io.textContent = '';
    for (const text of texts) {
      const li = document.createElement('li');
      li.innerHTML = '<q></q><output></output>';
      li.querySelector('q').textContent = text;
      io.appendChild(li);
    }
  }
  const renumber = () => [...code.children].forEach((el, i) => { el.dataset.n = i + 1; });

  async function written(token) {
    setStep(0);
    hl.removeAttribute('data-focus');
    ai.style.setProperty('--u', '.9');
    software.style.setProperty('--u', '0');
    draw(0);
    setRows(TICKETS);
    meter.textContent = '';
    for (const li of rows()) {
      const value = byHand(li.querySelector('q').textContent);
      await sleep(170);
      if (token !== run) return false;
      solidify(outputOf(li), value);
      li.classList.toggle('wrong', value === 'unknown');
    }
    return true;
  }

  async function trained(token, { quick = false } = {}) {
    const { build } = await programLoad();
    if (token !== run) return false;
    setStep(1);
    hl.dataset.focus = 'train';
    software.style.setProperty('--u', '.72');
    const lines = [...code.children];
    await melt(lines.slice(1));
    if (token !== run) return false;
    const learnedLine = lineElement(STAGES[1][1], { learned: LEARNED, n: 2 });
    code.appendChild(learnedLine);
    const model = build();
    const show = () => {
      const { probabilities } = model.predict(rows().map((li) => li.querySelector('q').textContent));
      rows().forEach((li, i) => { li.classList.remove('wrong'); superpose(outputOf(li), LABELS, probabilities[i]); });
    };
    show();
    await condense(learnedLine);
    if (token !== run) return false;
    // Train for real: one optimizer epoch per frame, and redraw what route() now returns.
    let compute = 0, first = null, last = null;
    for (let epoch = 1; epoch <= 30; epoch += 1) {
      const t0 = performance.now();
      const [loss] = model.epoch();
      compute += performance.now() - t0;
      first ??= loss; last = loss;
      show();
      meter.innerHTML = `training · epoch ${String(epoch).padStart(2)} / 30 · loss ${loss.toFixed(3)}`;
      pre.classList.add('training');
      if (!quick && !reduced) { await sleep(55); } else if (epoch % 10 === 0) { await frame(); }
      if (token !== run) return false;
    }
    pre.classList.remove('training');
    meter.innerHTML = `<span class="done">trained on 24 examples · loss ${first.toFixed(2)} → ${last.toFixed(3)} · ${compute.toFixed(0)} ms, in this tab</span>`;
    rows().forEach((li) => { const out = outputOf(li); const top = [...out.children].reduce((a, b) => (+a.style.opacity >= +b.style.opacity ? a : b)); solidify(out, top.textContent); });
    return model;
  }

  async function ruled(token, model) {
    setStep(2);
    hl.dataset.focus = 'program';
    ai.style.setProperty('--u', '0');
    const rule = lineElement('', { learned: LEARNED, phase: 'written' });
    code.insertBefore(rule, code.children[1]);
    renumber();
    await type(rule, highlight(STAGES[2][1], LEARNED), STAGES[2][1]);
    if (token !== run) return false;
    const li = document.createElement('li');
    li.innerHTML = '<q></q><output></output>';
    li.querySelector('q').textContent = LAWYER;
    io.appendChild(li);
    // What the learned call alone would say, for a beat, then the written rule answers.
    const { probabilities } = model.predict([LAWYER]);
    superpose(outputOf(li), LABELS, probabilities[0]);
    await sleep(700);
    if (token !== run) return false;
    solidify(outputOf(li), 'legal');
    return true;
  }

  async function play(from = 0) {
    const token = ++run;
    if (from === 0 && !(await written(token))) return;
    if (from === 0) await sleep(1900);
    if (token !== run) return;
    if (from > 0) { draw(0); setRows(TICKETS); ai.style.setProperty('--u', '.9'); software.style.setProperty('--u', '0'); }
    const model = await trained(token, { quick: from === 2 });
    if (!model) return;
    if (from < 2) await sleep(1500);
    if (token !== run) return;
    if (!(await ruled(token, model))) return;
    hl.removeAttribute('data-focus');
    await sleep(4200);
    if (token !== run) return;
    if (visible && !document.hidden && !reduced) play(0);
  }

  steps.forEach((step, i) => step.addEventListener('click', () => {
    if (i === 0) { ++run; written(run); } else play(i);
  }));
  whenVisible(root, (isVisible) => {
    const was = visible;
    visible = isVisible;
    if (isVisible && !was && !reduced) play(0);
  }, { once: false, margin: '0px' });

  if (reduced) { play(2); return; }
  play(0);
}

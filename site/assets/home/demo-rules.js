// Demo 02, rules: a learned Score ranks four actions; the written `allowed` mask
// decides which exist right now; Decide never picks a masked one. Rules change the
// choice immediately, with no retraining. Click a rule line to switch it off.
import { render, highlight, reduced, frame } from './code.js';

const INCIDENTS = [
  'errors after the deploy an hour ago',
  'memory keeps climbing until it dies',
  'requests queueing during the sale',
  'the disk is almost full',
];

export async function init(article) {
  article.querySelectorAll('pre.code.past').forEach((pre) => render(pre));
  const pre = article.querySelector('#rules-code');
  const lines = render(pre);
  const { build, IDS, allowed } = await import('../programs/rules.js');
  const chips = article.querySelector('#rules-incidents');
  const peak = article.querySelector('#rules-peak');
  const minutes = article.querySelector('#rules-minutes');
  const minutesOut = article.querySelector('#rules-minutes-out');
  const budget = article.querySelector('#rules-budget');
  const list = article.querySelector('#rules-options');

  // Lines 2..5 hold one rule each, in IDS order.
  const ruleLines = IDS.map((id) => lines.find((el) => el.textContent.trimStart().startsWith(`"${id}"`)));
  const original = ruleLines.map((el) => el.textContent);
  const off = new Set();
  ruleLines.forEach((el, i) => {
    el.classList.add('rule');
    el.tabIndex = 0;
    el.setAttribute('role', 'switch');
    el.setAttribute('aria-checked', 'true');
    el.title = 'Click to comment this rule out';
    const toggle = () => {
      const id = IDS[i];
      if (off.has(id)) off.delete(id); else off.add(id);
      const disabled = off.has(id);
      const text = disabled ? original[i].replace(/^(\s*)/, '$1# ') : original[i];
      el.innerHTML = highlight(text);
      el.classList.toggle('off', disabled);
      el.setAttribute('aria-checked', String(!disabled));
      update();
    };
    el.addEventListener('click', toggle);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  });

  let incident = INCIDENTS[0];
  INCIDENTS.forEach((text, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(i === 0));
    b.textContent = text;
    b.addEventListener('click', () => {
      incident = text;
      chips.querySelectorAll('button').forEach((c) => c.setAttribute('aria-checked', String(c === b)));
      update();
    });
    chips.appendChild(b);
  });

  list.innerHTML = IDS.map((id) => `<li data-id="${id}"><span class="name">${id}</span><span class="bar"><i></i></span><span class="why"></span></li>`).join('');
  const model = build();

  function update() {
    minutesOut.textContent = minutes.value;
    const rules = allowed({ peakHours: peak.checked, minutesSinceDeploy: Number(minutes.value), budgetLeft: budget.checked ? 100 : 0 });
    for (const id of off) rules[id] = true; // a commented-out rule allows everything
    const { learned, choice } = model.run(incident, rules);
    const top = Math.max(...learned), bottom = Math.min(...learned);
    const width = (v) => 6 + (94 * (v - bottom)) / Math.max(top - bottom, 1e-6);
    [...list.children].forEach((li, i) => {
      const id = IDS[i];
      const masked = !rules[id];
      li.classList.toggle('masked', masked);
      li.classList.toggle('chosen', id === choice);
      li.querySelector('i').style.width = `${width(learned[i]).toFixed(1)}%`;
      li.querySelector('.why').textContent = id === choice ? '← decide()' : masked ? `line ${ruleLines[i].dataset.n}` : '';
      li.title = `learned score ${learned[i].toFixed(2)}${masked ? `; masked by line ${ruleLines[i].dataset.n}` : ''}`;
    });
    ruleLines.forEach((el, i) => el.classList.toggle('blocking', !off.has(IDS[i]) && !rules[IDS[i]]));
  }

  for (const input of [peak, minutes, budget]) input.addEventListener('input', update);
  for (let epoch = 0; epoch < 25; epoch += 1) {
    model.epoch();
    if (!reduced && epoch % 5 === 4) { update(); await frame(); }
  }
  update();
}

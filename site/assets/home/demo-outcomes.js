// Demo 04, consequences: a written world, a written policy (take the best expected
// action), and a learned expectation trained only on what actually happened.
// Each round: try every action in six scenarios, observe rewards, train 20 epochs.
// Click the FIXES line to rewrite the world; the same program relearns from new outcomes.
import { render, highlight, sleep, frame, reduced } from './code.js';

const LINES = {
  usual: 'FIXES = {("cool", "hot"), ("reindex", "corrupt")}    # the world, written',
  swapped: 'FIXES = {("cool", "corrupt"), ("reindex", "hot")}    # the world, rewritten',
};

export async function init(article) {
  article.querySelectorAll('pre.code.past').forEach((pre) => render(pre));
  const pre = article.querySelector('#outcomes-code');
  const lines = render(pre);
  const worldLine = lines[0];
  const { build, TRAIN, TEST, ACTIONS, WORLDS } = await import('../programs/outcomes.js');
  const button = article.querySelector('#outcomes-run');
  const served = article.querySelector('#outcomes-served');
  const tbody = article.querySelector('#outcomes-observed tbody');
  const episodes = article.querySelector('#outcomes-episodes');
  const model = build();
  let world = 'usual', round = 0, busy = false;

  worldLine.classList.add('world');
  worldLine.tabIndex = 0;
  worldLine.title = 'Click to rewrite the world';
  const rewrite = () => {
    if (busy) return;
    world = world === 'usual' ? 'swapped' : 'usual';
    worldLine.innerHTML = highlight(LINES[world]);
    round = 0;
    table();
    evaluate(`the world changed; the policy hasn't`);
    button.textContent = '[ act, observe, learn ]';
  };
  worldLine.addEventListener('click', rewrite);
  worldLine.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); rewrite(); } });

  function table() {
    tbody.innerHTML = TRAIN.map(([, telemetry]) => `<tr><td title="${telemetry}">${telemetry}</td>${ACTIONS.map(() => '<td>·</td>').join('')}</tr>`).join('');
  }

  function evaluate(note = '') {
    const fixes = WORLDS[world];
    let count = 0;
    episodes.innerHTML = '';
    for (const [status, telemetry] of TEST) {
      const steps = model.episode(fixes, status, telemetry);
      const ok = steps.at(-1).next === 'serving';
      count += ok;
      const li = document.createElement('li');
      li.innerHTML = `<span class="tel"></span><span class="steps ${ok ? 'ok' : 'fail'}"></span>`;
      li.querySelector('.tel').textContent = telemetry;
      li.querySelector('.steps').textContent = steps.map((s) => s.action).join(' → ') + (ok ? '  served' : '  ✗');
      episodes.appendChild(li);
    }
    served.textContent = `${count} / ${TEST.length} served${note ? ` · ${note}` : ''}`;
    return count;
  }

  async function learn() {
    if (busy) return;
    busy = true;
    button.disabled = true;
    round += 1;
    const fixes = WORLDS[world];
    table();
    const sessions = [];
    const rows = [...tbody.children];
    for (const [r, [status, telemetry]] of TRAIN.entries()) {
      for (let a = 0; a < ACTIONS.length; a += 1) {
        const { session, reward } = model.observe(fixes, status, telemetry, a);
        sessions.push(session);
        const cell = rows[r].cells[a + 1];
        cell.textContent = reward > 0 ? `+${reward}` : `${reward}`;
        cell.classList.toggle('hit', reward > 0);
        cell.classList.add('flash');
        if (!reduced) await sleep(55);
        cell.classList.remove('flash');
      }
    }
    pre.classList.add('training');
    for (let epoch = 1; epoch <= 20; epoch += 1) {
      const losses = model.fit(sessions);
      const loss = losses.reduce((x, y) => x + y, 0) / losses.length;
      evaluate(`round ${round} · epoch ${epoch}/20 · loss ${loss.toFixed(3)}`);
      served.classList.add('training');
      if (!reduced) await frame(); else if (epoch % 5 === 0) await frame();
    }
    served.classList.remove('training');
    pre.classList.remove('training');
    const count = evaluate(`after ${round} round${round > 1 ? 's' : ''} of experience`);
    button.textContent = count === TEST.length ? '[ again ]' : '[ more experience ]';
    button.disabled = false;
    busy = false;
  }

  table();
  evaluate('untrained');
  button.addEventListener('click', learn);
}

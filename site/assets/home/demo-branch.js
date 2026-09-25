// Demo 01, examples: type a ticket; the written branch and the learned one both answer.
// "Teach" adds your example (source: "you") and retrains here, epoch by epoch.
import { superpose, render, frame, reduced } from './code.js';

export async function init(article) {
  article.querySelectorAll('pre.code').forEach((pre) => render(pre));
  const { build, LABELS, tokenize, routeByHand } = await import('../programs/branch.js');
  const input = article.querySelector('#branch-input');
  const tokens = article.querySelector('#branch-tokens');
  const hand = article.querySelector('#branch-hand');
  const dist = article.querySelector('#branch-dist');
  const teach = article.querySelector('#branch-teach');
  const log = article.querySelector('#branch-log');

  dist.innerHTML = '<div class="big"></div><ul class="bars"></ul>';
  const big = dist.querySelector('.big');
  const bars = dist.querySelector('.bars');
  bars.innerHTML = LABELS.map((l) => `<li><span>${l}</span><span class="bar"><i></i></span><span class="pct"></span></li>`).join('');

  const model = build();
  let taught = 0, training = false;

  function show() {
    const text = input.value.trim().toLowerCase();
    tokens.innerHTML = tokenize(text).map((w) => `<span class="${model.words.has(w) ? 'known' : 'unknown'}">${w.replace(/[<&>]/g, '')}</span>`).join('') || '<span class="unknown">(type something)</span>';
    const written = routeByHand(text);
    hand.textContent = written;
    hand.classList.toggle('unknown', written === 'unknown');
    if (!text) return;
    const { probabilities } = model.predict([text]);
    const p = probabilities[0];
    superpose(big, LABELS, p);
    [...bars.children].forEach((li, i) => {
      li.querySelector('i').style.width = `${(p[i] * 100).toFixed(1)}%`;
      li.querySelector('.pct').textContent = `${Math.round(p[i] * 100)}%`;
    });
  }

  const pre = article.querySelector('pre.code:not(.past)');
  async function train(epochs, label) {
    training = true;
    pre.classList.add('training');
    teach.querySelectorAll('button').forEach((b) => { b.disabled = true; });
    let compute = 0, loss = 0;
    for (let e = 1; e <= epochs; e += 1) {
      const t0 = performance.now();
      loss = model.epoch().at(-1);
      compute += performance.now() - t0;
      if (label) log.innerHTML = `<span class="sig">training · epoch ${e}/${epochs} · loss ${loss.toFixed(3)}</span>`;
      show();
      if (!reduced) await frame();
    }
    teach.querySelectorAll('button').forEach((b) => { b.disabled = false; });
    pre.classList.remove('training');
    training = false;
    return { compute, loss };
  }

  LABELS.forEach((label) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('click', async () => {
      const text = input.value.trim().toLowerCase();
      if (!text || training) return;
      model.teach([[text, label]], 'you');
      taught += 1;
      const { compute, loss } = await train(20, label);
      log.innerHTML = `Taught “${text.replace(/[<&>]/g, '')}” → ${label} (source: you). 20 epochs, ${compute.toFixed(0)} ms, loss ${loss.toFixed(3)}. ${24 + taught} examples now.`;
    });
    teach.appendChild(b);
  });

  input.addEventListener('input', () => { if (!training) show(); });
  show();
  const { compute } = await train(30, null);
  log.textContent = `Trained on 24 examples in ${compute.toFixed(0)} ms, in this tab.`;
}

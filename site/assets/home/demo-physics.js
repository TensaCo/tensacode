// Demo 03, mechanics: the learned aim is supervised only on where the written physics
// says the shot lands. Arcs are the written physics (solid); the aim is learned.
// Drag the target: the aim answers without retraining. Switch gravity: the written
// code changes, and the same aim retrains to it with no labels.
import { render, palette, fitCanvas, frame, reduced, whenVisible } from './code.js';

export async function init(article) {
  article.querySelectorAll('pre.code').forEach((pre) => render(pre));
  // Flight is written, but replayable tensor code: the gradient passes through it.
  const pre = article.querySelector('pre.code:not(.past)');
  const lines = [...pre.querySelectorAll('.ln')];
  lines.slice(0, 6).concat(lines.filter((el) => el.textContent.includes('fly('))).forEach((el) => el.classList.add('grad'));
  const { build, path, GOALS } = await import('../programs/physics.js');
  const canvas = article.querySelector('#physics-canvas');
  const trainButton = article.querySelector('#physics-train');
  const lossOut = article.querySelector('#physics-loss');
  const gravityButtons = [...article.querySelectorAll('[data-g]')];
  let gravity = 9.81, model = build({ gravity }), epochs = 0, lastLoss = null, target = 23, run = 0, dragging = false;
  const X_MAX = 45;
  let yMax = 8;

  function draw() {
    const { ctx, width, height } = fitCanvas(canvas, 0.56);
    const c = palette();
    const pad = { l: 14, r: 14, t: 44, b: 34 };
    const angles = model.angles([...GOALS, target]);
    const apex = Math.max(...angles.map((a) => Math.max(...path(a, gravity).map(([, y]) => y))));
    yMax += (Math.min(40, Math.max(4, apex * 1.08)) - yMax) * (lossOut.classList.contains('training') ? 0.35 : 1);
    const X = (x) => pad.l + (x / X_MAX) * (width - pad.l - pad.r);
    const Y = (y) => height - pad.b - (y / yMax) * (height - pad.t - pad.b);
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, width, height - pad.b + 1); ctx.clip();
    // Training shots: written physics from a learned angle. Thin solid arcs.
    ctx.lineWidth = 1; ctx.strokeStyle = c.inkA(0.35);
    GOALS.forEach((goal, i) => {
      const pts = path(angles[i], gravity);
      ctx.beginPath(); pts.forEach(([x, y]) => ctx.lineTo(X(x), Y(y))); ctx.stroke();
    });
    // Your target's shot, bold.
    const mine = path(angles.at(-1), gravity);
    ctx.lineWidth = 2; ctx.strokeStyle = c.ink;
    ctx.beginPath(); mine.forEach(([x, y]) => ctx.lineTo(X(x), Y(y))); ctx.stroke();
    ctx.restore();
    // Ground and ticks.
    ctx.strokeStyle = c.ink; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(0) + 0.5); ctx.lineTo(width - pad.r, Y(0) + 0.5); ctx.stroke();
    ctx.font = `11px ${c.mono}`; ctx.fillStyle = c.faint;
    for (let m = 0; m <= 40; m += 10) {
      ctx.fillRect(X(m), Y(0), 1, 5);
      ctx.fillText(`${m} m`, X(m) - (m ? 10 : 0), Y(0) + 18);
    }
    // Launcher and the learned angle for your target.
    ctx.fillStyle = c.ink; ctx.fillRect(X(0) - 4, Y(0) - 8, 8, 8);
    // Training flags (solid ticks) and where their shots landed (open circles).
    GOALS.forEach((goal, i) => {
      const land = path(angles[i], gravity).at(-1)[0];
      ctx.fillStyle = c.ink; ctx.fillRect(X(goal) - 0.5, Y(0) - 12, 1.5, 12);
      ctx.strokeStyle = c.inkA(0.6); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(X(land), Y(0) - 3, 3, 0, Math.PI * 2); ctx.stroke();
    });
    // Your target: a solid marker you can drag.
    const land = mine.at(-1)[0];
    ctx.fillStyle = c.ink;
    ctx.beginPath(); ctx.moveTo(X(target) - 7, Y(0) - 22); ctx.lineTo(X(target) + 7, Y(0) - 22); ctx.lineTo(X(target), Y(0) - 10); ctx.fill();
    ctx.font = `600 12px ${c.mono}`;
    const label = `target ${target.toFixed(1)} m · aim ${((angles.at(-1) * 180) / Math.PI).toFixed(1)}° · lands ${land.toFixed(1)} m`;
    const lx = Math.min(Math.max(X(target) - 70, pad.l), width - pad.r - ctx.measureText(label).width);
    ctx.fillText(label, lx, 16);
    ctx.strokeStyle = c.inkA(0.5); ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(X(target) + 0.5, 22); ctx.lineTo(X(target) + 0.5, Y(0) - 24); ctx.stroke();
    ctx.setLineDash([]);
  }

  async function train() {
    const token = ++run;
    model = build({ gravity });
    epochs = 0;
    trainButton.disabled = true;
    lossOut.classList.add('training');
    pre.classList.add('training');
    draw();
    for (let e = 0; e < 300; e += 6) {
      for (let k = 0; k < 6; k += 1) lastLoss = model.epoch();
      epochs = e + 6;
      lossOut.textContent = `epoch ${String(epochs).padStart(3)} / 300 · loss ${lastLoss.toFixed(2)}`;
      if (!reduced) { draw(); await frame(); }
      if (token !== run) return;
    }
    lossOut.classList.remove('training');
    pre.classList.remove('training');
    lossOut.textContent = `trained · loss ${lastLoss.toFixed(3)} · no angle labels`;
    trainButton.disabled = false;
    trainButton.textContent = '[ retrain from scratch ]';
    draw();
  }

  const toMeters = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left - 14) / (rect.width - 28)) * X_MAX;
    return Math.min(44, Math.max(4, x));
  };
  canvas.addEventListener('pointerdown', (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); target = toMeters(e); draw(); });
  canvas.addEventListener('pointermove', (e) => { if (dragging) { target = toMeters(e); draw(); } });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.tabIndex = 0;
  canvas.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); target = Math.min(44, Math.max(4, target + (e.key === 'ArrowLeft' ? -0.5 : 0.5))); draw(); }
  });
  trainButton.addEventListener('click', train);
  gravityButtons.forEach((b) => b.addEventListener('click', () => {
    gravity = Number(b.dataset.g);
    gravityButtons.forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    train();
  }));
  new ResizeObserver(() => draw()).observe(canvas);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  draw();
  lossOut.textContent = 'untrained';
  whenVisible(canvas, (visible) => { if (visible && epochs === 0) train(); }, { margin: '0px 0px -20% 0px' });
}

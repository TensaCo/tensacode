// 02: you can't dataset the world either. Four small networks (TensorCode Decode
// MLPs, trained here) learn total = price * qty from examples with qty in 1..12.
// More examples tighten nothing beyond them. Then the written line is exact everywhere.
import * as tc from '../lib/tensorcode.js';
import { palette, dotPattern, fitCanvas, sleep, frame, reduced, whenVisible } from './code.js';

const PRICE = 3, X_MAX = 40, Y_MAX = 160, LO = 1, HI = 12;
const STAGES = [8, 64, 256];
const MEMBERS = [1, 2, 3, 4];
const GRID = Array.from({ length: 81 }, (_, i) => (i / 80) * X_MAX);
const space = new tc.Space('qty', 1);
const latent = (xs) => new tc.Latent(tc.tensor(xs.map((x) => [x / X_MAX])), space);

function member(seed, xs) {
  tc.manualSeed(seed);
  const net = new tc.Decode({ architecture: 'mlp', hidden_dimensions: [32, 32], input_space: space.configuration(), output_dimensions: 1, output: 'total / 120' });
  const t = tc.trace();
  const out = t.run(() => net.call(latent(xs)));
  t.supervise(out, xs.map((x) => [(PRICE * x) / 120]), { loss: 'mse', source: 'examples' });
  const trainer = tc.Trainer.fromOps({ net }, { optimizer: (p) => new tc.Adam(p, { lr: 0.01 }) });
  return {
    epoch: () => trainer.fit([t], { epochs: 1 }),
    curve: () => tc.noGrad(() => net.call(latent(GRID)).tolist().map((r) => r[0] * 120)),
  };
}
const missAt40 = (curves) => Math.max(...curves.map((cv) => Math.abs(cv.at(-1) - PRICE * X_MAX)));
const examples = (n) => Array.from({ length: n }, (_, i) => LO + (HI - LO) * ((i * 0.6180339887) % 1));

export function init(figure) {
  const canvas = figure.querySelector('canvas');
  const count = figure.querySelector('#extrap-count');
  const replay = figure.querySelector('[data-replay]');
  let state = { xs: [], curves: [], written: 0, epoch: 0 };
  let run = 0;

  function draw() {
    const { ctx, width, height } = fitCanvas(canvas, 440 / 720);
    const c = palette();
    const pad = { l: 34, r: 12, t: 14, b: 30 };
    const X = (x) => pad.l + (x / X_MAX) * (width - pad.l - pad.r);
    const Y = (y) => height - pad.b - (y / Y_MAX) * (height - pad.t - pad.b);
    ctx.clearRect(0, 0, width, height);
    ctx.font = `11px ${c.mono}`;
    // Where the examples are.
    ctx.fillStyle = c.inkA(0.05);
    ctx.fillRect(X(LO), pad.t, X(HI) - X(LO), height - pad.t - pad.b);
    ctx.fillStyle = c.faint;
    ctx.fillText('the examples', X(LO) + 4, pad.t + 12);
    // Axes.
    ctx.strokeStyle = c.rule; ctx.lineWidth = 1;
    ctx.beginPath();
    for (const x of [0, 10, 20, 30, 40]) { ctx.moveTo(X(x) + 0.5, pad.t); ctx.lineTo(X(x) + 0.5, height - pad.b); }
    for (const y of [0, 40, 80, 120, 160]) { ctx.moveTo(pad.l, Y(y) + 0.5); ctx.lineTo(width - pad.r, Y(y) + 0.5); }
    ctx.stroke();
    ctx.fillStyle = c.faint;
    for (const x of [0, 10, 20, 30, 40]) ctx.fillText(String(x), X(x) - (x ? 6 : 0), height - pad.b + 16);
    for (const y of [40, 80, 120, 160]) ctx.fillText(String(y), 2, Y(y) + 4);
    ctx.textAlign = 'right'; ctx.fillText('qty →', X(35) + 12, height - pad.b + 16); ctx.textAlign = 'left';
    // Learned: a halftone band between the networks, and each network as a dotted curve.
    if (state.curves.length) {
      const lo = GRID.map((_, i) => Math.min(...state.curves.map((cv) => cv[i])));
      const hi = GRID.map((_, i) => Math.max(...state.curves.map((cv) => cv[i])));
      ctx.beginPath();
      GRID.forEach((x, i) => ctx.lineTo(X(x), Y(Math.min(Y_MAX, hi[i] + 1.2))));
      for (let i = GRID.length - 1; i >= 0; i -= 1) ctx.lineTo(X(GRID[i]), Y(Math.max(-5, lo[i] - 1.2)));
      ctx.closePath();
      ctx.fillStyle = dotPattern(ctx, c.learn, 4, 0.9);
      ctx.fill();
      ctx.setLineDash([1.5, 3]); ctx.strokeStyle = c.learnA(0.85); ctx.lineWidth = 1.2;
      for (const cv of state.curves) {
        ctx.beginPath();
        GRID.forEach((x, i) => ctx.lineTo(X(x), Y(Math.max(-5, Math.min(Y_MAX + 5, cv[i])))));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    // Observations: solid dots.
    ctx.fillStyle = c.ink;
    for (const x of state.xs) { ctx.beginPath(); ctx.arc(X(x), Y(PRICE * x), 2.2, 0, Math.PI * 2); ctx.fill(); }
    // Written: one solid line, everywhere.
    if (state.written) {
      ctx.strokeStyle = c.ink; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(X_MAX * state.written), Y(PRICE * X_MAX * state.written)); ctx.stroke();
      ctx.font = `600 13px ${c.mono}`;
      const text = 'total = price * qty', tx = X(15), ty = Y(PRICE * 30) ;
      ctx.fillStyle = c.paper; ctx.fillRect(tx - 4, ty - 14, ctx.measureText(text).width + 8, 20);
      ctx.fillStyle = c.ink; ctx.fillText(text, tx, ty);
    }
  }

  async function play() {
    const token = ++run;
    replay.disabled = true;
    state = { xs: [], curves: [], written: 0, epoch: 0 };
    for (const n of STAGES) {
      state.xs = examples(n);
      state.epoch = 0;
      count.textContent = `${n} examples`;
      const members = MEMBERS.map((seed) => member(seed, state.xs));
      const perFrame = n > 100 ? 4 : 10;
      for (let e = 0; e < 300; e += perFrame) {
        for (const m of members) for (let k = 0; k < perFrame; k += 1) m.epoch();
        state.epoch = e + perFrame;
        if (!reduced || e + perFrame >= 300) { state.curves = members.map((m) => m.curve()); draw(); }
        await frame();
        if (token !== run) return;
      }
      count.textContent = `${n} examples · off by up to ${missAt40(state.curves).toFixed(0)} at qty 40`;
      await sleep(1100);
      if (token !== run) return;
    }
    count.textContent = `${STAGES.at(-1)} examples: off by up to ${missAt40(state.curves).toFixed(0)} at qty 40 · 1 written line: exact`;
    state.written = 1; // written: it snaps, it doesn't grow
    draw();
    replay.disabled = false;
  }

  replay.addEventListener('click', play);
  new ResizeObserver(() => draw()).observe(canvas);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  draw();
  whenVisible(figure, (visible) => { if (visible) play(); }, { margin: '0px 0px -25% 0px' });
}

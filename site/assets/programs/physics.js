// site/examples/physics.py, run by TensorCode for TypeScript.
import * as tc from '../lib/tensorcode.js';

export const SPEED = 22, GRAVITY = 9.81, DRAG = 0.12, DT = 0.1, STEPS = 120; // 12 s of flight
export const GOALS = [8, 14, 20, 26, 32, 38];

/** Written: aim -> landing distance (m), with air drag. Replayable, so gradients pass through. */
export class Flight extends tc.ModuleOperation {
  static qualifiedName = 'examples.Flight';
  constructor(gravity = GRAVITY) { super(); this.gravity = gravity; }
  get replayable() { return true; }
  configuration() { return { speed: SPEED, gravity: this.gravity, drag: DRAG, dt: DT, steps: STEPS }; }
  forward(aim) {
    const angle = aim.sigmoid().mul(Math.PI / 2); // a rule: launch between 0 and 90 degrees
    let vx = angle.cos().mul(SPEED), vy = angle.sin().mul(SPEED);
    let x = tc.zerosLike(angle), y = tc.zerosLike(angle), landed = tc.zerosLike(angle);
    for (let step = 0; step < STEPS; step += 1) {
      const nx = x.add(vx.mul(DT)), ny = y.add(vy.mul(DT));
      [vx, vy] = [vx.sub(vx.mul(DRAG).mul(DT)), vy.sub(vy.mul(DRAG).add(this.gravity).mul(DT))];
      const falling = y.ge(0).logicalAnd(ny.lt(0));
      const share = y.div(y.sub(ny).clampMin(1e-6));
      landed = tc.where(falling, x.add(share.mul(nx.sub(x))), landed);
      [x, y] = [nx, ny];
    }
    return landed;
  }
}

/** The same physics in plain numbers, for drawing the arcs (no autograd). */
export function path(angle, gravity = GRAVITY) {
  let vx = SPEED * Math.cos(angle), vy = SPEED * Math.sin(angle), x = 0, y = 0;
  const points = [[0, 0]];
  for (let step = 0; step < STEPS; step += 1) {
    const nx = x + vx * DT, ny = y + vy * DT;
    [vx, vy] = [vx - DRAG * vx * DT, vy - (gravity + DRAG * vy) * DT];
    if (y >= 0 && ny < 0) { const share = y / Math.max(y - ny, 1e-6); points.push([x + share * (nx - x), 0]); break; }
    points.push([nx, ny]);
    [x, y] = [nx, ny];
  }
  return points;
}

export function build({ gravity = GRAVITY, goals = GOALS } = {}) {
  tc.manualSeed(3);
  const space = new tc.Space('target distance', 1);
  const aim = new tc.Decode({
    architecture: 'mlp', hidden_dimensions: [16], input_space: space.configuration(),
    output_dimensions: 1, output: 'aim (before the angle rule)',
  });
  const fly = new Flight(gravity);
  const targets = (meters) => new tc.Latent(tc.tensor(meters.map((m) => [m / 40])), space);
  const t = tc.trace();
  const landed = t.run(() => fly.call(aim.call(targets(goals))));
  t.supervise(landed, goals.map((m) => [m]), { loss: 'mse', source: 'the targets themselves' });
  const trainer = tc.Trainer.fromOps({ aim, fly }, { optimizer: (p) => new tc.Adam(p, { lr: 0.03 }) });
  return {
    epoch: () => trainer.fit([t], { epochs: 1 })[0],
    /** Launch angles (radians) the learned aim chooses for each distance. */
    angles: (meters) => tc.noGrad(() => aim.call(targets(meters)).sigmoid().mul(Math.PI / 2).tolist().map((row) => row[0])),
    land: (meters) => tc.noGrad(() => fly.call(aim.call(targets(meters))).tolist().map((row) => row[0])),
  };
}

export function main(print) {
  const model = build();
  const losses = [];
  for (let epoch = 0; epoch < 300; epoch += 1) losses.push(model.epoch());
  print(`loss ${losses[0].toFixed(1)} -> ${losses.at(-1).toFixed(3)}`);
  const unseen = [11, 23, 35];
  const angles = model.angles(unseen), landed = model.land(unseen);
  unseen.forEach((meters, i) => {
    print(`target ${meters} m: launch at ${(angles[i] * 180 / Math.PI).toFixed(1)} deg, lands at ${landed[i].toFixed(2)} m`);
  });
}

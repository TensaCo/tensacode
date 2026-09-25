// site/examples/outcomes.py, run by TensorCode for TypeScript.
import * as tc from '../lib/tensorcode.js';

export const ACTIONS = ['cool', 'reindex', 'serve'];
export const WORLDS = {
  // Written: how this world works. The page can rewrite it.
  usual: [['cool', 'hot'], ['reindex', 'corrupt']],
  swapped: [['cool', 'corrupt'], ['reindex', 'hot']],
};
export const TRAIN = [['hot', 'fans at max and cpu at 94c'], ['corrupt', 'index checksum mismatch'],
  ['ready', 'all health checks passing'], ['hot', 'thermal throttling on node 3'],
  ['corrupt', 'corrupt pages in the search index'], ['ready', 'warm and ready for traffic']];
export const TEST = [['hot', 'node 7 is thermal throttling'], ['corrupt', 'checksum mismatch in pages'],
  ['hot', 'cpu at 91c and climbing'], ['corrupt', 'search index returns corrupt rows'],
  ['hot', 'fans at max on node 2'], ['corrupt', 'index pages fail checksum']];
const TELEMETRY = { ready: 'all health checks passing' };

export function world(fixes, status, action) {
  if (action === 'serve') return status === 'ready' ? ['serving', 1.0] : [status, -0.5];
  if (fixes.some(([a, s]) => a === action && s === status)) return ['ready', 0.5];
  return [status, -0.5];
}

const tokenize = (text) => text.toLowerCase().match(/\w+|[^\w\s]/g) ?? [];

export function build() {
  tc.manualSeed(12);
  const texts = [...TRAIN, ...TEST].map(([, t]) => t).concat(ACTIONS);
  const words = [...new Set(texts.flatMap(tokenize))].sort();
  const space = { name: 'telemetry', dimensions: 16 };
  const encode = new tc.VocabularyEncoder({ vocabulary: words, dimensions: 16, output_space: space });
  const expect = new tc.Score({
    architecture: 'mlp', hidden_dimensions: [16], query_space: space, candidate_space: space,
    meaning: 'expected reward of an action',
  });
  const decide = new tc.Decide();
  const options = (telemetry) => new tc.CandidateSet(encode.call(telemetry), encode.call(ACTIONS), ACTIONS);
  const outcome = (scores, [index, reward]) => scores.values.select(0, index).sub(reward).square();
  const trainer = tc.Trainer.fromOps({ encode, expect }, { losses: { outcome }, optimizer: (p) => new tc.Adam(p, { lr: 0.01 }) });

  const expected = (telemetry) => tc.noGrad(() => expect.call(options(telemetry)).values.tolist());
  const choose = (telemetry) => tc.noGrad(() => decide.call(expect.call(options(telemetry))).identity);
  /** Written policy: take the best expected action, up to two steps. Returns the steps taken. */
  const episode = (fixes, status, telemetry, steps = 2) => {
    const taken = [];
    for (let step = 0; step < steps; step += 1) {
      const action = choose(telemetry);
      const [next, reward] = world(fixes, status, action);
      taken.push({ status, telemetry, action, reward, next });
      status = next;
      if (status === 'serving') break;
      telemetry = TELEMETRY[status] ?? telemetry;
    }
    return taken;
  };
  /** Explore: act, observe what happened, and supervise only the action that ran. */
  const observe = (fixes, status, telemetry, index) => {
    const [, reward] = world(fixes, status, ACTIONS[index]);
    const t = tc.trace();
    const scores = t.run(() => expect.call(options(telemetry)));
    t.supervise(scores, [index, reward], { loss: 'outcome', source: `observed:${status}:${ACTIONS[index]}` });
    return { session: t, reward };
  };
  return {
    expected, episode, observe,
    fit: (sessions, epochs = 1) => trainer.fit(sessions, { epochs }),
    served: (fixes) => TEST.filter(([status, telemetry]) => episode(fixes, status, telemetry).at(-1).next === 'serving').length,
  };
}

export function main(print) {
  const model = build();
  const fixes = WORLDS.usual;
  print(`before: ${model.served(fixes)}/${TEST.length} test scenarios served`);
  const sessions = [];
  for (const [status, telemetry] of TRAIN) {
    for (let index = 0; index < ACTIONS.length; index += 1) sessions.push(model.observe(fixes, status, telemetry, index).session);
  }
  const losses = model.fit(sessions, 20);
  print(`${sessions.length} observed outcomes, loss ${losses[0].toFixed(3)} -> ${losses.at(-1).toFixed(3)}`);
  print(`after: ${model.served(fixes)}/${TEST.length} test scenarios served`);
}

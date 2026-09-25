// site/examples/rules.py, run by TensorCode for TypeScript.
import * as tc from '../lib/tensorcode.js';

export const ACTIONS = {
  restart: 'restart the service process',
  rollback: 'roll back the latest deploy',
  scale: 'scale out more instances',
  page: 'page the on call engineer',
};
export const IDS = Object.keys(ACTIONS);
export const POSTMORTEMS = [
  ['errors started right after the deploy', 'rollback'],
  ['the new release throws exceptions', 'rollback'],
  ['memory keeps growing until the process dies', 'restart'],
  ['the service hung and stopped responding', 'restart'],
  ['traffic spike and requests are queueing', 'scale'],
  ['cpu saturated during the sale', 'scale'],
  ['the database disk is almost full', 'page'],
  ['strange alerts from the payment provider', 'page'],
];

export function allowed({ peakHours, minutesSinceDeploy, budgetLeft }) {
  return {
    restart: !peakHours, // never at peak
    rollback: minutesSinceDeploy < 60, // recent deploys only
    scale: budgetLeft > 0, // it costs money
    page: true, // a person, always
  };
}

const tokenize = (text) => text.toLowerCase().match(/\w+|[^\w\s]/g) ?? [];

export function build() {
  tc.manualSeed(5);
  const texts = [...POSTMORTEMS.map(([t]) => t), ...Object.values(ACTIONS)];
  const words = [...new Set(texts.flatMap(tokenize))].sort();
  const space = { name: 'operations text', dimensions: 24 };
  const encode = new tc.VocabularyEncoder({ vocabulary: words, dimensions: 24, output_space: space });
  const score = new tc.Score({
    architecture: 'mlp', hidden_dimensions: [24], query_space: space, candidate_space: space,
    meaning: 'learned fit of an action to an incident',
  });
  const decide = new tc.Decide();
  const options = (incident, rules = null) => {
    const actions = encode.call(Object.values(ACTIONS));
    const mask = rules === null ? null : tc.tensor(IDS.map((id) => rules[id]), { dtype: 'bool' });
    return new tc.CandidateSet(encode.call(incident), new tc.Latent(actions.tensor, actions.space, { mask }), IDS);
  };
  const sessions = POSTMORTEMS.map(([incident, action]) => {
    const t = tc.trace();
    const scores = t.run(() => score.call(options(incident)));
    t.supervise(scores, IDS.indexOf(action), { loss: 'choice', source: 'postmortem review' });
    return t;
  });
  const choice = (scores, target) => tc.crossEntropy(scores.values.unsqueeze(0), tc.tensor([target], { dtype: 'int64' }));
  const trainer = tc.Trainer.fromOps({ encode, score }, { losses: { choice }, optimizer: (p) => new tc.Adam(p, { lr: 0.01 }) });
  return {
    words: new Set(words),
    epoch: () => trainer.fit(sessions, { epochs: 1 }),
    /** Learned scores for every action (no mask), and the decision under `rules`. */
    run: (incident, rules) => tc.noGrad(() => {
      const learned = score.call(options(incident)).values.tolist();
      const scores = score.call(options(incident, rules));
      return { learned, masked: scores.values.tolist(), choice: decide.call(scores).identity };
    }),
  };
}

export function main(print) {
  const model = build();
  const losses = [];
  for (let epoch = 0; epoch < 25; epoch += 1) losses.push(...model.epoch());
  print(`loss ${losses[0].toFixed(4)} -> ${losses.at(-1).toFixed(4)}`);
  const incident = 'errors after the deploy an hour ago';
  for (const minutes of [20, 90]) {
    const rules = allowed({ peakHours: true, minutesSinceDeploy: minutes, budgetLeft: 100 });
    const { masked, choice } = model.run(incident, rules);
    const values = IDS.map((id, i) => `${id} ${masked[i].toFixed(2)}`).join(', ');
    print(`deploy ${String(minutes).padStart(2)} min ago: ${choice.padEnd(8)} scores: ${values}`);
  }
}

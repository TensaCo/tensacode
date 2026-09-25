// site/examples/branch.py, run by TensorCode for TypeScript. The page's demos use
// build(); main() prints what the Python program prints (scripts/site/check-examples.mjs).
import * as tc from '../lib/tensorcode.js';

export const LABELS = ['billing', 'technical', 'account'];
export const EXAMPLES = [
  ['i was charged twice for my order', 'billing'],
  ['please refund my last payment', 'billing'],
  ['why is there an extra fee on my invoice', 'billing'],
  ['my card was billed after i cancelled', 'billing'],
  ['i want my money back', 'billing'],
  ['the subscription price went up without notice', 'billing'],
  ['can i get a receipt for last month', 'billing'],
  ['how do i update my payment method', 'billing'],
  ['the app crashes when i open it', 'technical'],
  ['page will not load on my phone', 'technical'],
  ['i get an error when uploading a file', 'technical'],
  ['the website is really slow today', 'technical'],
  ['sync stopped working after the update', 'technical'],
  ['notifications are not arriving', 'technical'],
  ['the export button does nothing', 'technical'],
  ['screen goes blank after login', 'technical'],
  ['i forgot my password', 'account'],
  ['how do i change my email address', 'account'],
  ['please delete my account', 'account'],
  ['i cannot log in to my account', 'account'],
  ['someone else is using my account', 'account'],
  ['how do i add a team member', 'account'],
  ['i want to change my username', 'account'],
  ['my two factor code does not work', 'account'],
];
export const HELD_OUT = [
  ['you took money from me twice', 'billing'],
  ['cancel my plan and send a refund', 'billing'],
  ['the app freezes on startup', 'technical'],
  ['upload fails with a weird message', 'technical'],
  ['locked out after too many attempts', 'account'],
  ['change the email on my account', 'account'],
];

export const tokenize = (text) => text.toLowerCase().match(/\w+|[^\w\s]/g) ?? [];

export function routeByHand(ticket) {
  if (ticket.includes('refund') || ticket.includes('charged')) return 'billing';
  if (ticket.includes('error') || ticket.includes('crash')) return 'technical';
  if (ticket.includes('password') || ticket.includes('log in')) return 'account';
  return 'unknown';
}

/** The learned branch: construct, teach, predict. `extra` adds taught examples (source "you"). */
export function build() {
  tc.manualSeed(7);
  const words = [...new Set(EXAMPLES.flatMap(([text]) => tokenize(text)))].sort();
  const space = { name: 'tickets', dimensions: 24 };
  const encode = new tc.VocabularyEncoder({ vocabulary: words, dimensions: 24, output_space: space });
  const route = new tc.Classify({ architecture: 'linear', input_space: space, labels: LABELS });
  const trainer = tc.Trainer.fromOps({ encode, route }, { optimizer: (p) => new tc.Adam(p, { lr: 0.05 }) });
  const sessions = [];
  const teach = (examples, source) => {
    const t = tc.trace();
    const guess = t.run(() => route.call(encode.call(examples.map(([text]) => text))));
    t.supervise(guess, examples.map(([, label]) => label), { source });
    sessions.push(t);
  };
  teach(EXAMPLES, 'support team');
  return {
    words: new Set(words),
    teach,
    epoch: () => trainer.fit(sessions, { epochs: 1 }),
    predict: (texts) => tc.noGrad(() => {
      const p = route.call(encode.call(texts));
      return { labels: [...p.values], probabilities: p.probabilities.tolist() };
    }),
  };
}

export function main(print) {
  const model = build();
  const losses = [];
  for (let epoch = 0; epoch < 30; epoch += 1) losses.push(...model.epoch());
  print(`loss ${losses[0].toFixed(4)} -> ${losses.at(-1).toFixed(4)}`);
  const { labels, probabilities } = model.predict(HELD_OUT.map(([text]) => text));
  HELD_OUT.forEach(([text, label], i) => {
    const quoted = `'${text}'`.padEnd(40);
    print(`${quoted} ${label.padEnd(9)} by hand: ${routeByHand(text).padEnd(9)} learned: ${labels[i]} (${Math.max(...probabilities[i]).toFixed(2)})`);
  });
}

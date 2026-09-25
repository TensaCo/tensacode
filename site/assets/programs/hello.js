// site/examples/hello.py, run by TensorCode for TypeScript. Unseeded, like the Python.
import * as tc from '../lib/tensorcode.js';

export async function main(print) {
  const tickets = ['refund my order', 'charged twice', 'the app crashed', 'error on upload', 'forgot my password', 'cannot log in'];
  const labels = ['billing', 'billing', 'technical', 'technical', 'account', 'account'];

  const space = { name: 'tickets', dimensions: 8 };
  const encode = new tc.VocabularyEncoder({ vocabulary: [...new Set(tickets.flatMap((t) => t.split(' ')))].sort(), dimensions: 8, output_space: space });
  const route = new tc.Classify({ architecture: 'linear', input_space: space, labels: ['billing', 'technical', 'account'] });

  const t = tc.trace();
  const guess = t.run(() => route.call(encode.call(tickets)));
  t.supervise(guess, labels, { source: 'me' });
  tc.Trainer.fromOps({ encode, route }, { lr: 0.5 }).fit([t], { epochs: 50 });

  print(route.call(encode.call('the upload crashed')).value); // technical
}

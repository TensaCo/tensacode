// 01: you can't if/else the world. One rule per ticket the rules missed, until the
// function is texture. Then the whole block melts into one learned call.
import { lineElement, condense, sleep, reduced, whenVisible } from './code.js';

const HEAD = 'def route(text):';
const CURATED = [
  'if "refund" in text: return "billing"',
  'if "crash" in text: return "technical"',
  'if "password" in text: return "account"',
  'if "charged" in text: return "billing"',
  'if "money back" in text: return "billing"',
  'if "log in" in text or "login" in text: return "account"',
  'if "card" in text and "declined" in text: return "billing"',
  'if "card" in text and "lost" in text: return "account"',
  'if "slow" in text and "invoice" not in text: return "technical"',
  'if "error" in text and "payment" in text: return "billing"',
  'if "error" in text: return "technical"',
  'if "reimburse" in text: return "billing"  # "refund" missed these',
  'if "locked out" in text: return "account"',
  'if "freezes" in text or "frozen" in text: return "technical"',
  'if "frozen" in text and "account" in text: return "account"  # !!',
  'if "twice" in text and "charged" not in text: return "billing"',
  'if "cancel" in text and "plan" in text: return "billing"',
  'if "cancel" in text and "order" in text: return "billing"',
  'if "cancel" in text: return "account"  # probably?',
  'if "2fa" in text or "two factor" in text: return "account"',
  'if "blank" in text and "screen" in text: return "technical"',
  'if "won\'t load" in text or "wont load" in text: return "technical"',
  'if "hacked" in text or "someone else" in text: return "account"',
  'if "receipt" in text and "email" not in text: return "billing"',
  'if "receipt" in text: return "account"  # "email my receipt" -> ?',
];
const WORDS = ['refund', 'charge', 'card', 'app', 'login', 'email', 'error', 'sync', 'plan', 'price', 'upload', 'invoice',
  'reset', 'crash', 'slow', 'team', 'export', 'fee', 'owner', 'blank', 'renew', 'trial', 'coupon', 'phone', 'code',
  'delete', 'update', 'backup', 'invite', 'limit', 'tax', 'vpn', 'api', 'key', 'seat', 'sso'];
const LABELS = ['billing', 'technical', 'account'];

function generated(i) {
  // A deterministic stream of ever-more-specific conditions.
  const pick = (k) => WORDS[(i * 7 + k * 13 + ((i * k) % 11)) % WORDS.length];
  const conditions = [`"${pick(1)}" in text`];
  const depth = 1 + (i % 4);
  for (let k = 2; k <= depth; k += 1) conditions.push(k % 2 ? `"${pick(k)}" in text` : `"${pick(k)}" not in text`);
  return `if ${conditions.join(' and ')}: return "${LABELS[(i * 5 + depth) % 3]}"`;
}

export function init(figure) {
  const pre = figure.querySelector('#explode-code');
  const code = pre.querySelector('code');
  const count = figure.querySelector('#explode-count');
  const replay = figure.querySelector('[data-replay]');
  let run = 0;

  // Type stays full size; the block is squashed to fit, so rules become texture.
  const squash = (n) => {
    const height = pre.clientHeight || 440;
    const lineHeight = 14 * 1.72;
    code.style.transform = `scaleY(${Math.min(1, height / (n * lineHeight)).toFixed(4)})`;
  };
  const add = (text, n) => {
    const el = lineElement(`    ${text}`, { n });
    el.dataset.phase = 'written';
    code.appendChild(el);
  };

  async function play() {
    const token = ++run;
    replay.disabled = true;
    pre.classList.remove('melted');
    pre.style.setProperty('--code-size', '14px');
    code.style.transform = '';
    code.textContent = '';
    code.appendChild(lineElement(HEAD, { n: 1 }));
    CURATED.slice(0, 3).forEach((rule, i) => add(rule, i + 2));
    count.textContent = '3 rules';
    await sleep(900);
    const total = reduced ? 320 : 320;
    for (let n = 4; n <= total; ) {
      if (token !== run) return;
      const burst = n < 12 ? 1 : n < 30 ? 2 : n < 80 ? 6 : 14;
      for (let b = 0; b < burst && n <= total; b += 1, n += 1) add(n <= CURATED.length ? CURATED[n - 1] : generated(n), n + 1);
      squash(code.children.length);
      count.textContent = `${code.children.length - 1} rules`;
      await sleep(n < 12 ? 300 : n < 30 ? 120 : 45);
    }
    await sleep(900);
    if (token !== run) return;
    // Melt the rules; one learned line condenses out of them.
    count.textContent = `${code.children.length - 1} rules → 1 learned call`;
    pre.classList.add('melted');
    await sleep(reduced ? 0 : 520);
    if (token !== run) return;
    code.textContent = '';
    code.style.transform = '';
    pre.style.setProperty('--code-size', '17px');
    pre.classList.remove('melted');
    code.appendChild(lineElement(HEAD, { n: 1 }));
    const learned = lineElement('    return classify(encode(text)).value', { learned: new Set(['classify', 'encode']), n: 2 });
    code.appendChild(learned);
    await condense(learned);
    replay.disabled = false;
  }

  replay.addEventListener('click', play);
  whenVisible(figure, (visible) => { if (visible) play(); }, { margin: '0px 0px -30% 0px' });
}

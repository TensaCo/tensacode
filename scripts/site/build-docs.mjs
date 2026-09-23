#!/usr/bin/env node
// Builds https://tensorcode.dev/docs/ from the language implementations' Markdown.
//
//   node scripts/site/build-docs.mjs            # writes site/docs/
//   node scripts/site/build-docs.mjs --check    # converts everything, writes nothing,
//                                               # exits 1 on a broken internal link
//
// No dependencies. Each language lists its guides in reading order (`languages` below);
// a listed file that does not exist is skipped, and any `docs/*.md` a list forgot is
// appended under "More" so a new guide is never invisible. Pages that cover both
// languages (`shared` below) live in scripts/site/pages/. Output:
//
//   site/docs/index.html                     technical landing (both languages)
//   site/docs/<slug>/index.html              shared pages: architecture overview, install
//   site/docs/<lang>/index.html              the implementation's README
//   site/docs/<lang>/<slug>/index.html       every other guide
//   site/docs/sitemap.xml
//
// In any Markdown file, consecutive fenced blocks whose info string carries
// `tab=<Label>` (for example ```python tab=Python then ```ts tab=TypeScript) render as
// one tabbed code sample. The chosen language is remembered for the visitor ("tc-lang").
//
// Relative links between guides become site URLs; links to anything else in a
// repository (examples, JSON records, source) go to that file on GitHub. Code blocks are
// highlighted here, at build time, so the pages ship no highlighter.
//
// site/docs/ is generated and git-ignored: build it before previewing or deploying.

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, rmSync, statSync } from 'node:fs';
import { dirname, join, resolve, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const siteDir = join(root, 'site');
const out = join(siteDir, 'docs');
const origin = 'https://tensorcode.dev';
const check = process.argv.includes('--check');

// ---------------------------------------------------------------------------------
// What is published, in reading order.
// ---------------------------------------------------------------------------------

const languages = [
  {
    id: 'python',
    name: 'Python',
    dir: 'tensacode/python',
    repo: 'https://github.com/TensaCo/tensacode-py',
    branch: 'main',
    fence: 'python',
    blurb: 'The reference implementation, on PyTorch. Python 3.11 or newer; install from GitHub with pip.',
    install: [
      'python -m pip install "tensorcode[tools] @ git+https://github.com/TensaCo/tensacode-py"',
    ],
    sections: [
      { title: 'Start here', pages: [
        { file: 'README.md', slug: '', title: 'Introduction' },
        { file: 'docs/quickstart.md', slug: 'quickstart' },
        { file: 'docs/README.md', slug: 'overview', title: 'Architecture and contracts' },
        { file: 'docs/pretrained.md', slug: 'pretrained' },
      ]},
      { title: 'Guides', pages: [
        { file: 'docs/tools.md', slug: 'tools' },
        { file: 'docs/cognition.md', slug: 'cognition' },
        { file: 'docs/operations.md', slug: 'operations' },
        { file: 'docs/latent-models.md', slug: 'latent-models' },
        { file: 'docs/training.md', slug: 'training' },
        { file: 'examples/README.md', slug: 'examples', title: 'Examples' },
      ]},
      { title: 'Reference', pages: [
        { file: 'docs/validation.md', slug: 'validation' },
        { file: 'docs/results/README.md', slug: 'results', title: 'Evaluation records' },
        { file: 'docs/troubleshooting.md', slug: 'troubleshooting' },
        { file: 'docs/migration.md', slug: 'migration' },
      ]},
    ],
  },
  {
    id: 'typescript',
    name: 'TypeScript',
    dir: 'tensacode/typescript',
    repo: 'https://github.com/TensaCo/tensacode-ts',
    branch: 'main',
    fence: 'ts',
    blurb: 'The port for Node.js 20.16 or newer, with its own autograd core and no runtime dependencies; install from GitHub with npm.',
    install: [
      'npm install github:TensaCo/tensacode-ts',
    ],
    sections: [
      { title: 'Start here', pages: [
        { file: 'README.md', slug: '', title: 'Introduction' },
        { file: 'docs/quickstart.md', slug: 'quickstart' },
        { file: 'docs/parity.md', slug: 'parity', title: 'Parity with Python' },
      ]},
      { title: 'Guides', pages: [
        { file: 'docs/operations.md', slug: 'operations' },
        { file: 'docs/tools.md', slug: 'tools' },
        { file: 'docs/training.md', slug: 'training' },
        { file: 'examples/README.md', slug: 'examples', title: 'Examples' },
      ]},
    ],
  },
];

// Pages about both languages, from scripts/site/pages/, in reading order.
const shared = [
  { file: 'overview.md', slug: 'overview', title: 'Architecture overview' },
  { file: 'install.md', slug: 'install', title: 'Install' },
];
const sharedDir = join(root, 'scripts/site/pages');

// ---------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------

const escape = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const plain = (s) => s.replace(/<[^>]+>/g, '').replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/[`*]/g, '').replace(/&amp;/g, '&').trim();

// GitHub's heading anchors, so links written against GitHub keep working here.
function githubSlug(text) {
  return text
    .replace(/<[^>]+>/g, '')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/[`*]/g, '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N} _-]/gu, '')
    .replace(/ /g, '-');
}

const readJson = (p) => { try { return JSON.parse(readFileSync(p, 'utf8')); } catch { return null; } };

function versionOf(lang) {
  const base = join(root, lang.dir);
  if (lang.id === 'python') {
    try {
      const toml = readFileSync(join(base, 'pyproject.toml'), 'utf8');
      const name = /^name\s*=\s*"([^"]+)"/m.exec(toml)?.[1];
      const version = /^version\s*=\s*"([^"]+)"/m.exec(toml)?.[1];
      return { name, version };
    } catch { return {}; }
  }
  const pkg = readJson(join(base, 'package.json')) ?? {};
  return { name: pkg.name, version: pkg.version };
}

// ---------------------------------------------------------------------------------
// Syntax highlighting (build time)
// ---------------------------------------------------------------------------------

const PY_KEYWORDS = 'False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case'.split(' ');
const PY_BUILTINS = 'print len range dict list set tuple str int float bool open isinstance super type object enumerate zip sorted map filter any all min max sum repr getattr setattr hasattr iter next Path self cls'.split(' ');
const TS_KEYWORDS = 'abstract as async await break case catch class const continue debugger default delete do else enum export extends false finally for from function get if implements import in instanceof interface is keyof let new null of private protected public readonly return satisfies set static super switch this throw true try type typeof undefined var void while with yield'.split(' ');
const TS_BUILTINS = 'console Promise Array Object String Number Boolean Map Set Error JSON Math Date Record Partial Readonly Uint8Array Float32Array Float64Array BigInt Symbol process require module exports'.split(' ');
const SH_BUILTINS = 'cd echo export source cat ls mkdir rm cp mv git python python3 pip npm npx node pnpm yarn uv pytest curl wget tar sudo make hf'.split(' ');

function highlighter(lang) {
  const l = (lang || '').toLowerCase();
  if (['py', 'python', 'python3', 'pycon'].includes(l)) {
    return {
      rules: [
        ['c', /#[^\n]*/y],
        ['s', /[rRbBuUfF]{0,2}("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')/y],
        ['d', /@[A-Za-z_][\w.]*/y],
        ['n', /\b(?:0[xX][\da-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?j?)\b/y],
        ['w', /[A-Za-z_]\w*/y],
      ],
      word(w, rest, prev) {
        if (PY_KEYWORDS.includes(w)) return 'k';
        if (prev === 'def' || prev === 'class') return prev === 'class' ? 't' : 'f';
        if (PY_BUILTINS.includes(w)) return 'b';
        if (/^\s*\(/.test(rest)) return /^[A-Z]/.test(w) ? 't' : 'f';
        if (/^[A-Z][a-z]\w*$/.test(w)) return 't';
        return null;
      },
    };
  }
  if (['ts', 'typescript', 'js', 'javascript', 'mjs', 'cjs', 'tsx', 'jsx'].includes(l)) {
    return {
      rules: [
        ['c', /\/\/[^\n]*|\/\*[\s\S]*?\*\//y],
        ['s', /`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'/y],
        ['d', /@[A-Za-z_][\w.]*/y],
        ['n', /\b(?:0[xX][\da-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?n?)\b/y],
        ['w', /[A-Za-z_$][\w$]*/y],
      ],
      word(w, rest, prev) {
        if (TS_KEYWORDS.includes(w)) return 'k';
        if (prev === 'function') return 'f';
        if (prev === 'class' || prev === 'interface' || prev === 'type' || prev === 'extends' || prev === 'implements' || prev === 'new') return 't';
        if (TS_BUILTINS.includes(w)) return 'b';
        if (/^\s*(?:<[^>\n]*>)?\s*\(/.test(rest)) return /^[A-Z]/.test(w) ? 't' : 'f';
        if (/^[A-Z][a-z]\w*$/.test(w)) return 't';
        return null;
      },
    };
  }
  if (['sh', 'bash', 'shell', 'zsh', 'console', 'shell-session'].includes(l)) {
    return {
      rules: [
        ['c', /(?<=^|\s)#[^\n]*/my],
        ['s', /"(?:\\.|[^"\\])*"|'[^']*'/y],
        ['v', /\$\{[^}]*\}|\$[A-Za-z_]\w*/y],
        ['p', /(?<=\s)--?[A-Za-z][\w-]*/y],
        ['w', /[A-Za-z_][\w.-]*/y],
      ],
      word(w, _rest, _prev, atLineStart) {
        if (atLineStart && SH_BUILTINS.includes(w)) return 'f';
        if (atLineStart) return 'f';
        return null;
      },
    };
  }
  if (['json', 'jsonc', 'jsonl'].includes(l)) {
    return {
      rules: [
        ['c', /\/\/[^\n]*/y],
        ['key', /"(?:\\.|[^"\\\n])*"(?=\s*:)/y],
        ['s', /"(?:\\.|[^"\\\n])*"/y],
        ['n', /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/y],
        ['k', /\b(?:true|false|null)\b/y],
      ],
      word: () => null,
    };
  }
  if (['toml', 'ini', 'yaml', 'yml'].includes(l)) {
    return {
      rules: [
        ['c', /#[^\n]*/y],
        ['key', /^[ \t]*[A-Za-z0-9_.-]+(?=[ \t]*[=:])/my],
        ['s', /"(?:\\.|[^"\\\n])*"|'[^'\n]*'/y],
        ['n', /-?\b\d+(?:\.\d+)?\b/y],
        ['k', /\b(?:true|false|null)\b/y],
      ],
      word: () => null,
    };
  }
  return null;
}

function highlight(code, lang) {
  const hl = highlighter(lang);
  if (!hl) return escape(code);
  let i = 0; let html = ''; let plainRun = ''; let prevWord = '';
  const flush = () => { if (plainRun) { html += escape(plainRun); plainRun = ''; } };
  while (i < code.length) {
    let matched = false;
    for (const [kind, re] of hl.rules) {
      re.lastIndex = i;
      const m = re.exec(code);
      if (!m || m.index !== i || m[0].length === 0) continue;
      const text = m[0];
      let cls = kind;
      if (kind === 'w') {
        const lineStart = code.lastIndexOf('\n', i - 1) + 1;
        const before = code.slice(lineStart, i);
        const atLineStart = /^\s*(?:\$\s+)?$/.test(before) || /(?:&&|\|\||;|\|)\s*$/.test(before);
        cls = hl.word(text, code.slice(i + text.length, i + text.length + 40), prevWord, atLineStart);
        prevWord = text;
      } else if (kind === 'key') {
        cls = 'f';
      }
      if (cls) { flush(); html += `<span class="tok-${cls}">${escape(text)}</span>`; }
      else plainRun += text;
      i += text.length;
      matched = true;
      break;
    }
    if (!matched) {
      const ch = code[i];
      if (!/\s/.test(ch) && !/[\w$]/.test(ch)) prevWord = prevWord && /[.]/.test(ch) ? prevWord : (/[\s(]/.test(ch) ? prevWord : '');
      plainRun += ch;
      i++;
    }
  }
  flush();
  return html;
}

// ---------------------------------------------------------------------------------
// Markdown
// ---------------------------------------------------------------------------------

function makeInline(ctx) {
  return function inline(text) {
    const stash = [];
    const keep = (html) => { stash.push(html); return `\u0000${stash.length - 1}\u0000`; };
    let s = text;
    // Code spans first: nothing inside them is Markdown.
    s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, _t, code) => keep(`<code>${escape(code.trim() === '' ? code : code.replace(/^ (.*) $/, '$1'))}</code>`));
    // Autolinks.
    s = s.replace(/<(https?:\/\/[^>\s]+)>/g, (_, url) => keep(`<a href="${escape(sitePath(url) ?? url)}">${escape(url)}</a>`));
    // Inline HTML tags that are safe to pass through.
    s = s.replace(/<\/?(?:br|sub|sup|kbd|b|i|em|strong|code|span|details|summary)\b[^>]*>/gi, (tag) => keep(tag));
    s = escape(s);
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (_, alt, src) => keep(`<img src="${escape(ctx.asset(src))}" alt="${alt}" loading="lazy">`));
    s = s.replace(/\[((?:[^\[\]]|\[[^\]]*\])+)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (_, label, href) => {
      const url = ctx.link(href.replace(/&amp;/g, '&'));
      const ext = /^https?:/.test(url) && !url.startsWith(origin) ? ' rel="noopener"' : '';
      return `<a href="${escape(url)}"${ext}>${label}</a>`;
    });
    s = s.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^\w])__([^_]+?)__(?=[^\w]|$)/g, '$1<strong>$2</strong>');
    s = s.replace(/(^|[^\w*])\*([^*\s](?:[^*]*?[^*\s])?)\*(?=[^\w*]|$)/g, '$1<em>$2</em>');
    s = s.replace(/(^|[^\w])_([^_\s](?:[^_]*?[^_\s])?)_(?=[^\w]|$)/g, '$1<em>$2</em>');
    s = s.replace(/~~([^~]+)~~/g, '<del>$1</del>');
    s = s.replace(/ {2,}\n/g, '<br>\n');
    for (let n = 0; n < 3 && /\u0000\d+\u0000/.test(s); n++) s = s.replace(/\u0000(\d+)\u0000/g, (_, k) => stash[Number(k)]);
    return s;
  };
}

function markdown(src, ctx) {
  const inline = makeInline(ctx);
  const lines = src.replace(/\r\n?/g, '\n').replace(/\t/g, '    ').split('\n');
  const headings = [];
  const ids = new Set();
  let i = 0;

  const indentOf = (l) => l.search(/\S/);
  const isBlank = (l) => l === undefined || l.trim() === '';
  const listMarker = /^(\s*)([-*+]|\d{1,9}[.)])(\s+|$)/;

  function heading(level, raw) {
    const text = inline(raw);
    let id = githubSlug(raw) || 'section';
    const base = id; let n = 0;
    while (ids.has(id)) id = `${base}-${++n}`;
    ids.add(id);
    headings.push({ level, text, id, raw });
    return `<h${level} id="${id}">${text}<a class="anchor" href="#${id}" aria-label="Link to this section">#</a></h${level}>`;
  }

  function codeBlock(lang, code, info = '') {
    const l = (lang || '').split(/[\s,{]/)[0];
    const label = l ? `<span class="lang" aria-hidden="true">${escape(l)}</span>` : '';
    const cls = l ? ` class="language-${escape(l)}"` : '';
    const html = `<div class="code">${label}<pre><code${cls}>${highlight(code.replace(/\n+$/, ''), l)}</code></pre></div>`;
    const tab = /(?:^|\s)tab=(?:"([^"]+)"|(\S+))/.exec(info);
    return tab ? { tab: tab[1] ?? tab[2], html } : html;
  }

  // Consecutive tabbed code blocks become one tab group. Without JavaScript every
  // panel shows, stacked; the page script reveals the tab list and hides the others.
  function tabGroup(items) {
    const n = ++ctx.tabs.count;
    const key = (label) => label.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const tabs = items.map((it, k) => `<button type="button" role="tab" id="tab-${n}-${k}" aria-controls="panel-${n}-${k}" aria-selected="${k === 0}" tabindex="${k === 0 ? 0 : -1}" data-tab="${escape(key(it.tab))}">${escape(it.tab)}</button>`).join('');
    const panels = items.map((it, k) => `<div role="tabpanel" id="panel-${n}-${k}" aria-labelledby="tab-${n}-${k}" data-tab="${escape(key(it.tab))}">${it.html}</div>`).join('');
    return `<div class="code-tabs" data-code-tabs><div class="tab-list" role="tablist" aria-label="Code language" hidden>${tabs}</div>${panels}</div>`;
  }
  function groupTabs(parts) {
    const outParts = [];
    for (let k = 0; k < parts.length; k++) {
      if (typeof parts[k] === 'string') { outParts.push(parts[k]); continue; }
      const run = [];
      while (k < parts.length && typeof parts[k] === 'object') run.push(parts[k++]);
      k--;
      outParts.push(run.length > 1 ? tabGroup(run) : run[0].html);
    }
    return outParts;
  }

  // Renders blocks from lines[i] while lines are indented at least `indent`.
  function blocks(indent) {
    const parts = [];
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) { i++; continue; }
      if (indentOf(line) < indent) break;
      const t = line.slice(indent);
      let m;

      // Fenced code
      if ((m = /^(\s{0,3})(`{3,}|~{3,})\s*([^`\s]*)([^`]*)$/.exec(t))) {
        const fence = m[2]; const lang = m[3]; const info = m[4]; const inner = indent + m[1].length;
        const code = [];
        i++;
        while (i < lines.length && !new RegExp(`^\\s*${fence[0] === '`' ? '`' : '~'}{${fence.length},}\\s*$`).test(lines[i])) {
          const l = lines[i];
          code.push(indentOf(l) >= inner ? l.slice(inner) : l.trimStart());
          i++;
        }
        i++;
        parts.push(codeBlock(lang, code.join('\n'), info));
        continue;
      }
      // Indented code (only at top level, after a blank line)
      if (indent === 0 && /^ {4,}\S/.test(t) && (i === 0 || isBlank(lines[i - 1]))) {
        const code = [];
        while (i < lines.length && (/^ {4}/.test(lines[i]) || (isBlank(lines[i]) && /^ {4}/.test(lines[i + 1] ?? '')))) { code.push(lines[i].slice(4)); i++; }
        parts.push(codeBlock('', code.join('\n')));
        continue;
      }
      // ATX heading
      if ((m = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/.exec(t))) {
        parts.push(heading(m[1].length, m[2]));
        i++; continue;
      }
      // Setext heading
      if (!isBlank(lines[i + 1]) && /^\s*(=+|-+)\s*$/.test(lines[i + 1] ?? '') && !listMarker.test(t) && indentOf(lines[i + 1]) >= indent) {
        parts.push(heading(/=/.test(lines[i + 1]) ? 1 : 2, t.trim()));
        i += 2; continue;
      }
      // Rule
      if (/^ {0,3}([-*_])(\s*\1){2,}\s*$/.test(t)) { parts.push('<hr>'); i++; continue; }
      // Block quote
      if (/^ {0,3}>/.test(t)) {
        const quote = [];
        while (i < lines.length && !isBlank(lines[i]) && (/^\s*>/.test(lines[i].slice(indent)) || quote.length)) {
          const l = lines[i].slice(indent);
          if (!/^\s*>/.test(l) && (listMarker.test(l) || /^\s*(#|```)/.test(l))) break;
          quote.push(l.replace(/^\s*>\s?/, ''));
          i++;
        }
        const inner = markdown(quote.join('\n'), ctx);
        parts.push(`<blockquote>${inner.html}</blockquote>`);
        continue;
      }
      // Table
      if (/\|/.test(t) && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test((lines[i + 1] ?? '').slice(indent))) {
        const cells = (row) => {
          const cut = row.trim().replace(/^\|/, '').replace(/(?<!\\)\|$/, '');
          const outCells = []; let cur = ''; let tick = false;
          for (let k = 0; k < cut.length; k++) {
            const ch = cut[k];
            if (ch === '`') tick = !tick;
            if (ch === '\\' && cut[k + 1] === '|') { cur += '|'; k++; continue; }
            if (ch === '|' && !tick) { outCells.push(cur.trim()); cur = ''; continue; }
            cur += ch;
          }
          outCells.push(cur.trim());
          return outCells;
        };
        const aligns = cells(lines[i + 1].slice(indent)).map((c) => (/^:-+:$/.test(c) ? 'center' : /-+:$/.test(c) ? 'right' : ''));
        const head = cells(t);
        i += 2;
        const rows = [];
        while (i < lines.length && !isBlank(lines[i]) && /\|/.test(lines[i])) { rows.push(cells(lines[i].slice(indent))); i++; }
        const al = (k) => (aligns[k] ? ` style="text-align:${aligns[k]}"` : '');
        parts.push(`<div class="table" role="region" tabindex="0" aria-label="Table"><table><thead><tr>${head.map((c, k) => `<th${al(k)}>${inline(c)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${head.map((_, k) => `<td${al(k)}>${inline(r[k] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
        continue;
      }
      // List
      if ((m = listMarker.exec(t)) && indentOf(t) <= 3) {
        const ordered = /\d/.test(m[2]);
        const start = ordered ? parseInt(m[2], 10) : 1;
        const items = [];
        let loose = false;
        const markerIndent = indent + m[1].length;
        while (i < lines.length) {
          const l = lines[i];
          if (isBlank(l)) break;
          const lm = listMarker.exec(l.slice(indent));
          if (!lm || indent + lm[1].length !== markerIndent || /\d/.test(lm[2]) !== ordered) break;
          const contentIndent = markerIndent + lm[2].length + Math.max(1, Math.min(lm[3].length, 4));
          // Rewrite the first line so the item's body is uniformly indented.
          lines[i] = ' '.repeat(contentIndent) + l.slice(indent + lm[0].length);
          const itemParts = [];
          const first = i;
          // Collect the item: its first paragraph may continue lazily; nested blocks are indented.
          let body = blocksForItem(contentIndent);
          if (body.hadBlank) loose = true;
          itemParts.push(body.html);
          items.push({ html: itemParts.join(''), first });
          // Blank lines between items make the list loose.
          let j = i;
          while (j < lines.length && isBlank(lines[j])) j++;
          if (j > i && j < lines.length) {
            const nl = listMarker.exec(lines[j].slice(indent));
            if (nl && indent + nl[1].length === markerIndent && /\d/.test(nl[2]) === ordered) { loose = true; i = j; continue; }
          }
        }
        const tag = ordered ? 'ol' : 'ul';
        const lis = items.map((it) => `<li>${loose ? it.html : it.html.replace(/^<p>([\s\S]*?)<\/p>/, '$1')}</li>`).join('');
        parts.push(`<${tag}${ordered && start !== 1 ? ` start="${start}"` : ''}>${lis}</${tag}>`);
        continue;
      }
      // Raw HTML block
      if (/^ {0,3}<(?:[a-zA-Z][\w-]*[\s>/]|!--|\/[a-zA-Z])/.test(t)) {
        const raw = [];
        while (i < lines.length && !isBlank(lines[i])) { raw.push(lines[i].slice(Math.min(indent, indentOf(lines[i])))); i++; }
        parts.push(raw.join('\n')
          .replace(/(href|src)="([^"]+)"/g, (_, attr, v) => `${attr}="${escape(attr === 'src' ? ctx.asset(v) : ctx.link(v))}"`));
        continue;
      }
      // Paragraph
      const para = [];
      while (i < lines.length) {
        const l = lines[i];
        if (isBlank(l)) break;
        if (indentOf(l) < indent) break;
        const lt = l.slice(indent);
        if (para.length && (/^ {0,3}(```|~~~|#{1,6}\s|>)/.test(lt) || (listMarker.test(lt) && indentOf(lt) <= 3 && !/^\s*\d/.test(lt)) || /^ {0,3}([-*_])(\s*\1){2,}\s*$/.test(lt))) break;
        if (para.length && /\|/.test(lt) && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] ?? '')) break;
        para.push(lt.trim() + (/ {2,}$/.test(lt) ? '  ' : ''));
        i++;
      }
      parts.push(`<p>${inline(para.join('\n'))}</p>`);
    }
    return groupTabs(parts).join('\n');
  }

  // An item's body: everything indented at least `inner`, plus lazy paragraph lines.
  function blocksForItem(inner) {
    const startLine = i;
    const collected = [];
    let hadBlank = false;
    // First line (already re-indented) and its lazy continuation lines.
    collected.push(lines[i]); i++;
    while (i < lines.length) {
      const l = lines[i];
      if (isBlank(l)) {
        // Continue the item only if the next non-blank line is indented into it.
        let j = i; while (j < lines.length && isBlank(lines[j])) j++;
        if (j < lines.length && indentOf(lines[j]) >= inner) {
          for (let k = i; k < j; k++) collected.push('');
          hadBlank = true;
          i = j; continue;
        }
        break;
      }
      if (indentOf(l) >= inner) { collected.push(l); i++; continue; }
      // Lazy continuation: a plain text line that is not a new block or list item.
      const prev = collected[collected.length - 1];
      if (!isBlank(prev) && !listMarker.test(l) && !/^\s*(```|~~~|#{1,6}\s|>|\|)/.test(l) && !/^\s*([-*_])(\s*\1){2,}\s*$/.test(l)) {
        collected.push(' '.repeat(inner) + l.trim()); i++; continue;
      }
      break;
    }
    const after = i;
    // Render the collected lines in isolation.
    const saved = lines.splice(0, lines.length, ...collected.map((l) => (isBlank(l) ? '' : l.slice(Math.min(inner, indentOf(l))))));
    const savedI = i; i = 0;
    const html = blocks(0);
    lines.splice(0, lines.length, ...saved);
    i = savedI;
    void startLine; void after;
    return { html, hadBlank: hadBlank && /<\/p>[\s\S]*<p>/.test(html) };
  }

  const html = blocks(0);
  return { html, headings };
}

// ---------------------------------------------------------------------------------
// Collect pages
// ---------------------------------------------------------------------------------

const allPages = []; // every page on the site (for links, sitemap, checks)

function slugFor(file) {
  const base = posix.basename(file, '.md');
  if (base.toLowerCase() === 'readme') return posix.basename(posix.dirname(file)) || 'readme';
  return base;
}

for (const lang of languages) {
  const base = join(root, lang.dir);
  lang.meta = versionOf(lang);
  lang.present = existsSync(base);
  const listed = new Set();
  for (const section of lang.sections) {
    section.pages = section.pages.filter((p) => {
      listed.add(p.file);
      return existsSync(join(base, p.file));
    });
  }
  // Guides nobody listed are still published, under "More".
  const docsDir = join(base, 'docs');
  if (existsSync(docsDir)) {
    const extra = readdirSync(docsDir)
      .filter((n) => n.endsWith('.md') && statSync(join(docsDir, n)).isFile() && !listed.has(`docs/${n}`))
      .sort()
      .map((n) => ({ file: `docs/${n}`, slug: slugFor(`docs/${n}`) }));
    if (extra.length) lang.sections.push({ title: 'More', pages: extra });
  }
  // A language with no README still gets an index page.
  const hasIndex = lang.sections.some((s) => s.pages.some((p) => p.slug === ''));
  if (!hasIndex) {
    lang.sections.unshift({ title: 'Start here', pages: [{ slug: '', title: 'Introduction', generated: true }] });
  }
  lang.sections = lang.sections.filter((s) => s.pages.length);
  const slugs = new Set();
  for (const section of lang.sections) {
    for (const page of section.pages) {
      page.slug ??= slugFor(page.file);
      while (slugs.has(page.slug)) page.slug += '-2';
      slugs.add(page.slug);
      page.lang = lang;
      page.section = section;
      page.url = `/docs/${lang.id}/${page.slug ? page.slug + '/' : ''}`;
      if (page.generated) {
        page.source = [
          `# TensorCode for ${lang.name}`,
          '',
          `The ${lang.name} documentation is being written. Until it lands here, the source and`,
          `its README are on [GitHub](${lang.repo}).`,
          '',
        ].join('\n');
      } else {
        page.source = readFileSync(join(base, page.file), 'utf8');
      }
      const h1 = /^\s{0,3}#\s+(.+?)\s*#*\s*$/m.exec(page.source)?.[1];
      page.heading = h1 ? plain(h1) : lang.name;
      page.title ??= page.heading;
      allPages.push(page);
    }
  }
  lang.flat = lang.sections.flatMap((s) => s.pages);
}

for (const page of shared) {
  page.lang = null;
  page.url = `/docs/${page.slug}/`;
  page.source = readFileSync(join(sharedDir, page.file), 'utf8');
  const h1 = /^\s{0,3}#\s+(.+?)\s*#*\s*$/m.exec(page.source)?.[1];
  page.heading = h1 ? plain(h1) : page.title;
  page.title ??= page.heading;
  allPages.push(page);
}

// ---------------------------------------------------------------------------------
// Links
// ---------------------------------------------------------------------------------

// A link written as https://tensorcode.dev/... becomes a site path, so it also works on
// previews and the workers.dev address.
const sitePath = (href) => (href === origin || href.startsWith(`${origin}/`) ? href.slice(origin.length) || '/' : null);

function contextFor(page) {
  const lang = page.lang;
  const tabs = { count: 0 };
  if (!lang) {
    // Shared pages link with site paths (/docs/python/quickstart/) or full URLs.
    return { tabs, link: (href) => sitePath(href) ?? href, asset: (src) => src };
  }
  const byFile = new Map(lang.flat.filter((p) => p.file).map((p) => [p.file, p]));
  const fileDir = page.file ? posix.dirname(page.file) : '.';
  const resolveRel = (path) => posix.normalize(posix.join(fileDir, path)).replace(/^\.\//, '');
  return {
    tabs,
    link(href) {
      const local = sitePath(href);
      if (local) return local;
      if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(href)) return href;
      if (href.startsWith('#')) return href;
      if (href.startsWith('/')) return `${lang.repo}/blob/${lang.branch}${href}`;
      const [path, hash] = href.split('#');
      const target = resolveRel(decodeURI(path));
      const tail = hash ? `#${hash}` : '';
      if (target.startsWith('../')) {
        // Outside the implementation's repository: the monorepo.
        return `https://github.com/TensaCo/tensacode/tree/develop/${posix.normalize(posix.join(lang.dir, target))}${tail}`;
      }
      const hit = byFile.get(target) ?? byFile.get(posix.join(target, 'README.md'));
      if (hit) return hit.url + tail;
      const onDisk = join(root, lang.dir, target);
      const isDir = target.endsWith('/') || target === '.' || (existsSync(onDisk) ? statSync(onDisk).isDirectory() : !/\.[A-Za-z0-9]+$/.test(target));
      return `${lang.repo}/${isDir ? 'tree' : 'blob'}/${lang.branch}/${target.replace(/\/$/, '').replace(/^\.$/, '')}${tail}`;
    },
    asset(src) {
      if (/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(src)) return src;
      const target = resolveRel(src);
      return `${lang.repo.replace('https://github.com/', 'https://raw.githubusercontent.com/')}/${lang.branch}/${target}`;
    },
  };
}

// ---------------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------------

const icons = {
  docs: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 2.5h7l3 3v8H3z"/><path d="M10 2.5v3h3M5.5 8.5h5M5.5 11h5"/></svg>',
  github: '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38l-.01-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.93-.89-1.18-.89-1.18-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.72 1.23 1.88.88 2.34.67.07-.52.28-.88.51-1.08-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48l-.01 2.19c0 .21.14.46.55.38A8 8 0 0 0 8 0Z"/></svg>',
  moon: '<svg class="moon" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M13.6 10.4A6 6 0 0 1 5.6 2.4a6 6 0 1 0 8 8Z"/></svg>',
  sun: '<svg class="sun" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="3.1"/><path d="M8 1.1v1.7M8 13.2v1.7M1.1 8h1.7M13.2 8h1.7M3.15 3.15l1.2 1.2M11.65 11.65l1.2 1.2M12.85 3.15l-1.2 1.2M4.35 11.65l-1.2 1.2"/></svg>',
  bars: '<svg class="bars" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M2 4h12M2 8h12M2 12h12"/></svg>',
  cross: '<svg class="cross" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M3.5 3.5l9 9M12.5 3.5l-9 9"/></svg>',
  mark: '<svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><path class="loop" d="M25.4 12.6A10 10 0 1 1 17.7 6.2"/><path class="loop" d="M15.3 2.4l3.3 3.9-3.9 3.1"/><path class="tick" d="M11.6 16.4l3.1 3.1 5.9-6.4"/></svg>',
};
const favicon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%2312151a'/%3E%3Cg fill='none' stroke-linecap='round' stroke-linejoin='round' stroke-width='3'%3E%3Cpath stroke='%233fd1b8' d='M25.4 12.6A10 10 0 1 1 17.7 6.2M15.3 2.4l3.3 3.9-3.9 3.1'/%3E%3Cpath stroke='%23f7f6f2' d='M11.6 16.4l3.1 3.1 5.9-6.4'/%3E%3C/g%3E%3C/svg%3E";
const themeSnippet = "<script>(function(){try{var t=localStorage.getItem('tc-theme');if(t!=='light'&&t!=='dark')t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';document.documentElement.dataset.theme=t}catch(e){}})();</script>";

function head({ title, description, url }) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escape(title)}</title>
<meta name="description" content="${escape(description)}">
<link rel="canonical" href="${origin}${url}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="TensorCode">
<meta property="og:url" content="${origin}${url}">
<meta property="og:title" content="${escape(title)}">
<meta property="og:description" content="${escape(description)}">
<meta property="og:image" content="${origin}/media/og.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="theme-color" content="#f7f6f2" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0b0e12" media="(prefers-color-scheme: dark)">
<link rel="icon" href="${favicon}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap">
<link rel="stylesheet" href="/assets/base.css">
<link rel="stylesheet" href="/assets/docs.css">
${themeSnippet}
</head>`;
}

function topBar({ lang, page, menu }) {
  const switcher = languages.map((l) => {
    const same = page && l.flat.find((p) => p.slug === page.slug);
    const href = same ? same.url : `/docs/${l.id}/`;
    const current = lang && l.id === lang.id;
    return `<a href="${href}"${current ? ' aria-current="true"' : ''}>${escape(l.name)}</a>`;
  }).join('');
  return `<header class="site-top is-docs">
  <div class="bar">
    <a class="brand" href="/" aria-label="TensorCode home">${icons.mark}<span>TensorCode</span></a>
    <a class="docs-tag" href="/docs/" style="text-decoration:none">Docs</a>
    <nav class="lang-switch" aria-label="Language">${switcher}</nav>
    <nav class="corner" aria-label="Resources">
      <a class="docs-home" href="/docs/overview/"${page && page.slug === 'overview' && !page.lang ? ' aria-current="page"' : ''}>Overview</a>
      <a class="gh" href="${lang ? lang.repo : 'https://github.com/TensaCo/tensacode'}" aria-label="${lang ? `TensorCode for ${escape(lang.name)} on GitHub` : 'TensorCode on GitHub'}">${icons.github}<span>GitHub</span></a>
      <button class="icon-btn" type="button" data-theme-toggle hidden aria-label="Switch theme">${icons.moon}${icons.sun}</button>
      ${menu ? `<button class="icon-btn menu-btn" id="menu" type="button" aria-expanded="false" aria-controls="side" aria-label="Show the documentation menu">${icons.bars}${icons.cross}</button>` : ''}
    </nav>
  </div>
</header>`;
}

const pageScript = `<script>
(function(){
  var menu=document.getElementById('menu'),side=document.getElementById('side');
  if(menu&&side){
    var set=function(open){side.classList.toggle('open',open);menu.setAttribute('aria-expanded',String(open));menu.setAttribute('aria-label',open?'Hide the documentation menu':'Show the documentation menu');if(open)side.scrollTop=0;};
    menu.addEventListener('click',function(){set(!side.classList.contains('open'));});
    side.addEventListener('click',function(e){if(e.target.closest('a'))set(false);});
    document.addEventListener('keydown',function(e){if(e.key==='Escape'&&side.classList.contains('open')){set(false);menu.focus();}});
    matchMedia('(min-width: 861px)').addEventListener('change',function(e){if(e.matches)set(false);});
  }
  document.querySelectorAll('.code').forEach(function(block){
    var b=document.createElement('button');b.type='button';b.className='copy';b.textContent='Copy';
    b.addEventListener('click',function(){
      var text=block.querySelector('pre').innerText;
      var done=function(ok){b.textContent=ok?'Copied':'Press Ctrl+C';b.classList.toggle('done',ok);setTimeout(function(){b.textContent='Copy';b.classList.remove('done');},1800);};
      if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(text).then(function(){done(true);},function(){done(false);});}
      else{var r=document.createRange();r.selectNodeContents(block.querySelector('pre'));var s=getSelection();s.removeAllRanges();s.addRange(r);var ok=false;try{ok=document.execCommand('copy');}catch(e){}done(ok);}
    });
    block.appendChild(b);
  });
  var LANG='tc-lang',pref=null;
  try{pref=localStorage.getItem(LANG);}catch(e){}
  var here=document.body.getAttribute('data-lang');
  if(here){pref=here;try{localStorage.setItem(LANG,here);}catch(e){}}
  var groups=[].slice.call(document.querySelectorAll('[data-code-tabs]'));
  var choose=function(g,key,focus){
    var tabs=[].slice.call(g.querySelectorAll('[role="tab"]'));
    if(!tabs.some(function(t){return t.getAttribute('data-tab')===key;}))key=tabs[0].getAttribute('data-tab');
    tabs.forEach(function(t){var on=t.getAttribute('data-tab')===key;t.setAttribute('aria-selected',String(on));t.tabIndex=on?0:-1;if(on&&focus)t.focus();});
    g.querySelectorAll('[role="tabpanel"]').forEach(function(p){p.hidden=p.getAttribute('data-tab')!==key;});
  };
  var chooseAll=function(key,from){groups.forEach(function(g){choose(g,key,g===from);});try{localStorage.setItem(LANG,key);}catch(e){}};
  groups.forEach(function(g){
    g.querySelector('[role="tablist"]').hidden=false;g.classList.add('ready');
    choose(g,pref,false);
    g.addEventListener('click',function(e){var t=e.target.closest('[role="tab"]');if(t){var y=t.getBoundingClientRect().top;chooseAll(t.getAttribute('data-tab'),null);window.scrollBy(0,t.getBoundingClientRect().top-y);}});
    g.addEventListener('keydown',function(e){
      var tabs=[].slice.call(g.querySelectorAll('[role="tab"]')),i=tabs.indexOf(document.activeElement);if(i<0)return;
      var j=e.key==='ArrowRight'?(i+1)%tabs.length:e.key==='ArrowLeft'?(i-1+tabs.length)%tabs.length:e.key==='Home'?0:e.key==='End'?tabs.length-1:-1;
      if(j<0)return;e.preventDefault();chooseAll(tabs[j].getAttribute('data-tab'),g);
    });
  });
  var links=[].slice.call(document.querySelectorAll('.toc a'));
  if(links.length&&'IntersectionObserver' in window){
    var map=new Map(links.map(function(a){return [decodeURIComponent(a.hash.slice(1)),a];}));
    var io=new IntersectionObserver(function(entries){entries.forEach(function(en){if(en.isIntersecting){links.forEach(function(a){a.classList.remove('active');});var a=map.get(en.target.id);if(a)a.classList.add('active');}});},{rootMargin:'-70px 0px -70% 0px'});
    map.forEach(function(_,id){var h=document.getElementById(id);if(h)io.observe(h);});
  }
})();
</script>`;

function sidebar(lang, page) {
  const v = lang.meta.version ? `<small>${escape(lang.meta.name ?? '')} ${escape(lang.meta.version)}</small>` : '';
  const groups = lang.sections.map((s) => `<div class="group"><p class="group-title">${escape(s.title)}</p>${s.pages.map((p) =>
    `<a href="${p.url}"${p === page ? ' aria-current="page"' : ''}>${escape(p.title)}</a>`).join('')}</div>`).join('');
  return `<nav class="side" id="side" aria-label="${escape(lang.name)} documentation"><p class="lang-title">${escape(lang.name)} ${v}</p>${groups}${sharedLinks(null)}</nav>`;
}

// Links every sidebar ends with: the shared pages, then the way out.
function sharedLinks(current) {
  const pages = shared.map((p) => `<a href="${p.url}"${p === current ? ' aria-current="page"' : ''}>${escape(p.title)}</a>`).join('');
  return `<div class="group back"><p class="group-title">All languages</p>${pages}<a href="/docs/">&larr; Docs home</a><a href="/">&larr; tensorcode.dev</a></div>`;
}

function sharedSidebar(page) {
  const langs = languages.map((l) => {
    const pages = l.flat.filter((p) => ['', 'quickstart', 'parity'].includes(p.slug));
    return `<div class="group"><p class="group-title">${escape(l.name)}</p>${pages.map((p) => `<a href="${p.url}">${escape(p.slug === '' ? `${l.name} introduction` : p.title)}</a>`).join('')}<a href="/docs/${l.id}/">All ${escape(l.name)} guides &rarr;</a></div>`;
  }).join('');
  const start = `<div class="group"><p class="group-title">Start here</p>${shared.map((p) => `<a href="${p.url}"${p === page ? ' aria-current="page"' : ''}>${escape(p.title)}</a>`).join('')}</div>`;
  return `<nav class="side" id="side" aria-label="Documentation"><p class="lang-title">Python and TypeScript</p>${start}${langs}<div class="group back"><a href="/docs/">&larr; Docs home</a><a href="/">&larr; tensorcode.dev</a></div></nav>`;
}

function renderPage(page) {
  const lang = page.lang;
  const { html, headings } = markdown(page.source, contextFor(page));
  page.ids = new Set(headings.map((h) => h.id));
  page.html = html;
  const toc = headings.filter((h) => h.level === 2);
  const order = lang ? lang.flat : shared;
  const at = order.indexOf(page);
  const prev = at > 0 ? order[at - 1] : null;
  const next = at < order.length - 1 ? order[at + 1] : null;
  const firstPara = /<p>([\s\S]*?)<\/p>/.exec(html)?.[1];
  const scope = lang ? `TensorCode for ${lang.name}` : 'TensorCode for Python and TypeScript';
  const description = firstPara ? plain(firstPara).replace(/\s+/g, ' ').slice(0, 180) : `${page.title}: ${scope}.`;
  const repoRoot = 'https://github.com/TensaCo/tensacode';
  const editUrl = lang
    ? (page.file ? `${lang.repo}/blob/${lang.branch}/${page.file}` : lang.repo)
    : `${repoRoot}/blob/develop/scripts/site/pages/${page.file}`;
  const crumb = lang
    ? `<a href="/docs/">Docs</a><span aria-hidden="true">/</span><a href="/docs/${lang.id}/">${escape(lang.name)}</a><span aria-hidden="true">/</span><span>${escape(page.section.title)}</span>`
    : `<a href="/docs/">Docs</a><span aria-hidden="true">/</span><span>${escape(page.title)}</span>`;
  return `${head({ title: `${page.title} | TensorCode ${lang ? `${lang.name} ` : ''}docs`, description, url: page.url })}
<body class="docs"${lang ? ` data-lang="${lang.id}"` : ''}>
<a class="skip" href="#content">Skip to content</a>
${topBar({ lang, page, menu: true })}
<div class="shell${toc.length ? '' : ' no-toc'}">
  ${lang ? sidebar(lang, page) : sharedSidebar(page)}
  <main class="doc-main" id="content">
    <p class="crumb">${crumb}</p>
    <article class="prose">
${html}
    </article>
    ${(prev || next) ? `<nav class="pager" aria-label="Previous and next">${prev ? `<a class="prev" href="${prev.url}"><small>&larr; Previous</small>${escape(prev.title)}</a>` : ''}${next ? `<a class="next" href="${next.url}"><small>Next &rarr;</small>${escape(next.title)}</a>` : ''}</nav>` : ''}
    <footer class="doc-foot"><span>&copy; 2026 TensorCode &middot; MIT licensed &middot; by TensaCo</span><a href="${editUrl}">${page.file ? 'Edit this page on GitHub' : 'Source on GitHub'}</a></footer>
  </main>
  ${toc.length ? `<nav class="toc" aria-label="On this page"><p class="group-title">On this page</p>${toc.map((h) => `<a href="#${h.id}">${plain(h.text)}</a>`).join('')}</nav>` : ''}
</div>
<script src="/assets/theme.js" defer></script>
${pageScript}
</body>
</html>
`;
}

// ---------------------------------------------------------------------------------
// The technical landing: /docs/
// ---------------------------------------------------------------------------------

function firstCodeBlock(lang, fences) {
  const readme = lang.flat.find((p) => p.slug === '' && !p.generated);
  if (!readme) return null;
  const re = /^```([\w-]*)[^\n]*\n([\s\S]*?)^```\s*$/gm;
  let m;
  while ((m = re.exec(readme.source))) if (fences.includes(m[1].toLowerCase())) return { lang: m[1], code: m[2].replace(/\n$/, '') };
  return null;
}

const archSvg = `<svg viewBox="0 -28 900 388" role="img" aria-labelledby="arch-t arch-d">
<title id="arch-t">TensorCode architecture</title>
<desc id="arch-d">Operations (vec, text, graph) are composed into tools (Chatbot, Investigator, Planner, Decision, Scene). A tool owns its model parameters and creates sessions that hold sourced evidence, revisions and memory. Sessions emit receipts and experience records; the trainer fits the tool's parameters from experience; artifacts are saved with save_pretrained and restored with from_pretrained, locally or from the Hugging Face Hub.</desc>
<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0 0L10 5L0 10z"/></marker>
  <marker id="ah-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head-a" d="M0 0L10 5L0 10z"/></marker>
  <marker id="ah-w" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head-w" d="M0 0L10 5L0 10z"/></marker>
</defs>
<text class="t-l" x="20" y="28">OPERATIONS</text>
<rect class="box" x="20" y="40" width="190" height="62" rx="10"/><text class="t-h mono" x="36" y="66">ops.vec</text><text class="t-s" x="36" y="86">encode, decode, classify, retrieve</text>
<rect class="box" x="20" y="114" width="190" height="62" rx="10"/><text class="t-h mono" x="36" y="140">ops.text</text><text class="t-s" x="36" y="160">messages, classify, decide, retrieve</text>
<rect class="box-dash" x="20" y="188" width="190" height="62" rx="10"/><text class="t-h mono" x="36" y="214">ops.graph</text><text class="t-s" x="36" y="234">symbolic interfaces (stubs)</text>

<text class="t-l" x="270" y="28">TOOLS</text>
<rect class="box-accent" x="270" y="40" width="200" height="210" rx="12"/>
<text class="t-h mono" x="288" y="70">Chatbot</text><text class="t-h mono" x="288" y="98">Investigator</text><text class="t-h mono" x="288" y="126">Planner</text><text class="t-h mono" x="288" y="154">Decision</text><text class="t-h mono" x="288" y="182">Scene</text>
<text class="t-s" x="288" y="214">own encoders, workspace, decoders</text><text class="t-s" x="288" y="232">configured from JSON</text>

<text class="t-l" x="530" y="28">RUNTIME</text>
<rect class="box" x="530" y="40" width="170" height="96" rx="10"/><text class="t-h" x="546" y="66">Sessions</text><text class="t-s" x="546" y="86">sourced evidence</text><text class="t-s" x="546" y="103">immutable revisions</text><text class="t-s" x="546" y="120">episodic memory</text>
<rect class="box" x="530" y="154" width="170" height="96" rx="10"/><text class="t-h" x="546" y="180">Receipts</text><text class="t-s" x="546" y="200">candidates + sources</text><text class="t-s" x="546" y="217">support / contradiction</text><text class="t-s" x="546" y="234">abstentions</text>

<text class="t-l" x="760" y="28">PERSISTENCE</text>
<rect class="box" x="740" y="40" width="140" height="96" rx="10"/><text class="t-h" x="756" y="66">Artifacts</text><text class="t-s mono" x="756" y="88">save_pretrained</text><text class="t-s mono" x="756" y="106">from_pretrained</text><text class="t-s" x="756" y="124">local or HF Hub</text>
<rect class="box-warm" x="740" y="154" width="140" height="96" rx="10"/><text class="t-h" x="756" y="180">Experience</text><text class="t-s" x="756" y="200">reviewed targets</text><text class="t-s" x="756" y="217">action outcomes</text><text class="t-s" x="756" y="234">source provenance</text>

<path class="edge" d="M210 71H262" marker-end="url(#ah)"/><path class="edge" d="M210 145H262" marker-end="url(#ah)"/><path class="edge" d="M210 219H262" marker-end="url(#ah)" stroke-dasharray="4 4"/>
<path class="edge" d="M470 88H522" marker-end="url(#ah)"/>
<path class="edge" d="M615 136V146" marker-end="url(#ah)"/>
<path class="edge" d="M700 202H732" marker-end="url(#ah-w)"/>
<path class="edge-a" d="M866 40V-12H370V32" marker-end="url(#ah-a)"/>
<text class="t-s" x="500" y="-18">from_pretrained restores exact weights and configuration</text>

<rect class="box-accent" x="330" y="290" width="330" height="54" rx="12"/>
<text class="t-h mono" x="348" y="314">training.Trainer.from_tool(model)</text><text class="t-s" x="348" y="332">capture &#8594; save experience &#8594; fit &#8594; checkpoint</text>
<path class="edge-w" d="M810 250V317H668" marker-end="url(#ah-w)"/>
<path class="edge-a" d="M370 290V258" marker-end="url(#ah-a)"/>
<text class="t-s" x="378" y="276">updates parameters</text>
</svg>`;

function renderLanding() {
  const cards = languages.map((lang) => {
    const pages = lang.flat.filter((p) => p.slug !== '').slice(0, 7);
    const sample = firstCodeBlock(lang, lang.id === 'python' ? ['python', 'py'] : ['ts', 'typescript', 'js', 'javascript']);
    const ver = lang.meta.version ? `${lang.meta.name ?? ''} ${lang.meta.version}` : '';
    const install = lang.install.join('\n');
    return `<article class="lang-card">
  <header><h3><a href="/docs/${lang.id}/">${escape(lang.name)}</a></h3><span class="ver">${escape(ver)}</span></header>
  <p class="dl-sub" style="margin:0">${escape(lang.blurb)}</p>
  <div class="code"><span class="lang" aria-hidden="true">bash</span><pre><code class="language-bash">${highlight(install, 'bash')}</code></pre></div>
  ${sample ? `<div class="code"><span class="lang" aria-hidden="true">${escape(sample.lang)}</span><pre><code>${highlight(sample.code, sample.lang)}</code></pre></div>` : ''}
  ${pages.length ? `<ul>${pages.map((p) => `<li><a href="${p.url}"><span>${escape(p.title)}</span><span>${escape(p.section.title)}</span></a></li>`).join('')}</ul>` : `<p class="dl-sub" style="margin:0">Guides for ${escape(lang.name)} are being written. Start with the introduction and the source on <a href="${lang.repo}">GitHub</a>.</p>`}
  <a class="more" href="/docs/${lang.id}/">All ${escape(lang.name)} docs &rarr;</a>
</article>`;
  }).join('\n');
  const py = languages.find((l) => l.id === 'python');
  const validation = py.flat.find((p) => p.slug === 'validation');
  return `${head({ title: 'TensorCode documentation', description: 'Developer documentation for TensorCode: callable operations, tools that own their trainable models, sourced evidence and revision, experience capture, training and portable artifacts, in Python and TypeScript.', url: '/docs/' })}
<body class="docs">
<a class="skip" href="#content">Skip to content</a>
${topBar({ lang: null, page: null, menu: false })}
<main id="content">
  <section class="dl-hero">
    <div class="container">
      <p class="kicker">tensorcode / developer documentation</p>
      <h1>Trainable tools with sourced evidence and owned weights.</h1>
      <p>TensorCode composes callable <b>operations</b> (<code>ops.vec</code>, <code>ops.text</code>, <code>ops.graph</code>) into <b>tools</b> that own their encoders, workspace and decoders. Tools create <b>sessions</b> that keep source evidence, generated hypotheses, model assessments and observed outcomes separate, and emit receipts you can audit.</p>
      <p>Reviewed targets and action outcomes become data-only <b>experience</b>; a <code>Trainer</code> fits the tool from it; <code>save_pretrained</code> / <code>from_pretrained</code> restore exact configuration and weights, offline or from the Hugging Face Hub. Importing the core package loads no ML framework and makes no network calls.</p>
      <p class="dl-actions"><a class="btn btn-primary" href="/docs/install/">Install</a><a class="btn" href="/docs/overview/">Architecture overview</a></p>
    </div>
  </section>

  <section class="dl-section" aria-labelledby="lang-h">
    <div class="container">
      <h2 id="lang-h" class="dl-h2">Pick an implementation</h2>
      <p class="dl-sub">Both implement the same public contracts: operations, tools, sessions, tracing, experience and artifacts. The Python package is the reference. The TypeScript port matches it and reads and writes the same files, so a model trained in one language loads in the other. <a href="/docs/typescript/parity/">What differs</a>.</p>
      <div class="lang-cards">
${cards}
      </div>
    </div>
  </section>

  <section class="dl-section" aria-labelledby="arch-h">
    <div class="container">
      <h2 id="arch-h" class="dl-h2">How the pieces fit</h2>
      <p class="dl-sub">Operations are the callable units. Tools compose them and own every parameter. Sessions are runtime state, never saved into model weights. Experience is the only path from runtime back into training.</p>
      <div class="arch">${archSvg}</div>
    </div>
  </section>

  <section class="dl-section" aria-labelledby="concepts-h">
    <div class="container">
      <h2 id="concepts-h" class="dl-h2">Core concepts</h2>
      <p class="dl-sub">The vocabulary the guides use.</p>
      <dl class="concepts">
        <div><dt>Operation</dt><dd>A callable <code>operation(value, *, context=None)</code>. Learned operations take JSON configuration and own their weights; weightless ones are pure transforms.</dd></div>
        <div><dt>Tool</dt><dd>An owned model (Chatbot, Investigator, Planner, Decision, Scene) with public interaction contracts. Constructing one initializes weights and downloads nothing.</dd></div>
        <div><dt>Evidence</dt><dd>Source-identified observations. Generated hypotheses are interpretations, never evidence; revisions keep what a source originally said.</dd></div>
        <div><dt>Receipt</dt><dd>Per-candidate support / contradiction / unknown assessments per source, with provenance, truncation and abstention status.</dd></div>
        <div><dt>Experience</dt><dd>Data-only training records: inputs, reviewed targets or observed outcomes, and who supplied them.</dd></div>
        <div><dt>Artifact</dt><dd><code>tensorcode_config.json</code> + <code>model.safetensors</code> + model card. Loading rejects incompatible artifacts instead of executing code.</dd></div>
      </dl>
    </div>
  </section>

  <section class="dl-section" aria-labelledby="scope-h">
    <div class="container">
      <h2 id="scope-h" class="dl-h2">Status and measured scope</h2>
      <div class="callout"><b>Alpha.</b> The APIs are pre-1.0 and change between releases. Published checkpoints are small, fixed-split experiments, and they do not yet establish a consistent benefit from the recurrent workspace or general cognitive competence. Model probabilities are uncalibrated and verifier approval is not truth. ${validation ? `Read <a href="${validation.url}">validation and scope</a> before choosing a model.` : ''}</div>
    </div>
  </section>
</main>
<footer class="site-foot"><div class="container"><p class="fine" style="margin-top:0;border-top:0;padding-top:0"><span>&copy; 2026 TensorCode &middot; MIT licensed &middot; by TensaCo</span><span><a href="/">tensorcode.dev</a> &middot; <a href="https://github.com/TensaCo/tensacode-py">Python on GitHub</a> &middot; <a href="https://github.com/TensaCo/tensacode-ts">TypeScript on GitHub</a></span></p></div></footer>
<script src="/assets/theme.js" defer></script>
${pageScript}
</body>
</html>
`;
}

// ---------------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------------

const files = new Map(); // site-relative path -> html
for (const page of allPages) {
  files.set(`${page.url.slice(1)}index.html`, renderPage(page));
}
files.set('docs/index.html', renderLanding());

// Check every internal /docs/ link (and its #fragment) resolves.
const problems = [];
const idsByUrl = new Map();
for (const [path, html] of files) {
  idsByUrl.set('/' + path.replace(/index\.html$/, ''), new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
}
for (const [path, html] of files) {
  const pageUrl = '/' + path.replace(/index\.html$/, '');
  for (const m of html.matchAll(/href="([^"]+)"/g)) {
    const href = m[1].replace(/&amp;/g, '&');
    if (href.startsWith('#')) {
      const ids = idsByUrl.get(pageUrl);
      if (ids && !ids.has(decodeURIComponent(href.slice(1)))) problems.push(`${pageUrl}: missing anchor ${href}`);
      continue;
    }
    if (!href.startsWith('/docs/')) continue;
    const [u, hash] = href.split('#');
    if (!idsByUrl.has(u)) { problems.push(`${pageUrl}: broken link ${href}`); continue; }
    const ids = idsByUrl.get(u);
    if (hash && ids && !ids.has(decodeURIComponent(hash))) problems.push(`${pageUrl}: missing anchor ${href}`);
  }
}

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${['/docs/', ...allPages.map((p) => p.url)].map((u) => `  <url><loc>${origin}${u}</loc></url>`).join('\n')}
</urlset>
`;

if (!check) {
  rmSync(out, { recursive: true, force: true });
  for (const [path, html] of files) {
    const dest = join(siteDir, path);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, html);
  }
  writeFileSync(join(out, 'sitemap.xml'), sitemap);
}

for (const lang of languages) {
  const n = lang.flat.length;
  const note = lang.flat.some((p) => p.generated) ? ' (placeholder index: no README found)' : '';
  console.log(`${lang.name.padEnd(11)} ${String(n).padStart(2)} page${n === 1 ? '' : 's'}${note}`);
}
if (problems.length) {
  console.warn(`\n${problems.length} link problem${problems.length === 1 ? '' : 's'}:`);
  for (const p of problems) console.warn(`  ${p}`);
}
console.log(`${check ? 'checked' : 'wrote'} ${files.size} pages${check ? '' : ` to ${posix.relative(root, out) || out}`}`);
if (check && problems.length) process.exit(1);

#!/usr/bin/env node
// Checks the built site (run build-docs.mjs first). No dependencies.
//
//   node scripts/site/check-links.mjs              # internal links, anchors, code blocks
//   node scripts/site/check-links.mjs --external   # also request every external link once
//
// Internal: every href/src on every page under site/ that points inside the site
// must reach a file (with Cloudflare's auto-trailing-slash rules) and, if it has a
// #fragment, an element with that id. Code: every <pre> holds a <code>, no Markdown
// fence leaked into the HTML, and every tab group has one panel per tab.
// Exits 1 on any internal problem. External failures are reported, not fatal.

import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const site = join(root, 'site');
const external = process.argv.includes('--external');

const walk = (dir) => readdirSync(dir).flatMap((n) => {
  const p = join(dir, n);
  return statSync(p).isDirectory() ? walk(p) : [p];
});
const pages = walk(site).filter((p) => p.endsWith('.html'));
const urlOf = (file) => '/' + relative(site, file).split(sep).join('/').replace(/(^|\/)index\.html$/, '$1');

// The file a site path serves, following html_handling "auto-trailing-slash".
function fileFor(path) {
  const clean = decodeURIComponent(path);
  const direct = join(site, clean);
  if (clean.endsWith('/')) return existsSync(join(direct, 'index.html')) ? join(direct, 'index.html') : null;
  if (existsSync(direct) && statSync(direct).isFile()) return direct;
  if (existsSync(`${direct}.html`)) return `${direct}.html`;
  if (existsSync(join(direct, 'index.html'))) return join(direct, 'index.html');
  return null;
}

const idCache = new Map();
const idsOf = (file) => {
  if (!idCache.has(file)) {
    const html = readFileSync(file, 'utf8');
    idCache.set(file, new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
  }
  return idCache.get(file);
};

const problems = [];
const externals = new Map(); // url -> first page that links it
let checked = 0;

for (const file of pages) {
  const html = readFileSync(file, 'utf8');
  const pageUrl = urlOf(file);
  const base = new URL(pageUrl, 'https://site.invalid');

  // Links and resources.
  for (const m of html.matchAll(/<(\w+)\b[^>]*?\s(href|src)="([^"]*)"[^>]*>/g)) {
    const tag = m[0]; const attr = m[2];
    const raw = m[3].replace(/&amp;/g, '&');
    if (!raw || /^(?:data|mailto|tel|javascript):/i.test(raw)) continue;
    if (/^(?:https?:)?\/\//i.test(raw)) {
      const canonical = m[1] === 'link' && /rel="(?:canonical|alternate)"/.test(tag);
      if (raw.startsWith('https://tensorcode.dev') && !canonical) problems.push(`${pageUrl}: absolute link to its own site ${raw} (use a site path)`);
      if (!externals.has(raw)) externals.set(raw, pageUrl);
      continue;
    }
    checked++;
    const target = new URL(raw, base);
    const hash = target.hash ? decodeURIComponent(target.hash.slice(1)) : '';
    const dest = fileFor(target.pathname);
    if (!dest) { problems.push(`${pageUrl}: broken ${attr} ${raw}`); continue; }
    if (hash && dest.endsWith('.html') && !idsOf(dest).has(hash)) problems.push(`${pageUrl}: missing anchor ${raw}`);
  }

  // Code blocks.
  // Docs code blocks are <div class="code"><pre><code>; marketing pages use bare <pre>
  // for image prompts.
  if (pageUrl.startsWith('/docs/')) {
    for (const [, inner] of html.matchAll(/<pre\b[^>]*>([\s\S]*?)<\/pre>/g)) {
      if (!/^<code\b[\s\S]*<\/code>$/.test(inner.trim())) problems.push(`${pageUrl}: <pre> without a <code> block`);
    }
  }
  const text = html.replace(/<pre\b[\s\S]*?<\/pre>/g, '').replace(/<script\b[\s\S]*?<\/script>/g, '').replace(/<code\b[\s\S]*?<\/code>/g, '');
  if (/```|~~~/.test(text)) problems.push(`${pageUrl}: a Markdown code fence leaked into the page`);
  for (const g of html.matchAll(/<div class="code-tabs"[\s\S]*?<\/div><\/div><\/div>(?=\n|<)/g)) {
    const tabs = (g[0].match(/role="tab"/g) ?? []).length;
    const panels = (g[0].match(/role="tabpanel"/g) ?? []).length;
    if (tabs < 2 || tabs !== panels) problems.push(`${pageUrl}: tab group with ${tabs} tabs and ${panels} panels`);
  }
}

console.log(`${pages.length} pages, ${checked} internal links checked, ${externals.size} distinct external links`);
if (problems.length) {
  console.log(`\n${problems.length} problem${problems.length === 1 ? '' : 's'}:`);
  for (const p of problems) console.log(`  ${p}`);
}

if (external) {
  const failed = [];
  const urls = [...externals.keys()].filter((u) => !/^\/\//.test(u) && !/fonts\.(googleapis|gstatic)\.com\/?$/.test(u));
  const status = async (url) => {
    for (const method of ['HEAD', 'GET']) {
      try {
        const res = await fetch(url, { method, redirect: 'follow', signal: AbortSignal.timeout(20000), headers: { 'user-agent': 'tensorcode-link-check' } });
        if (res.ok || method === 'GET') return res.status;
      } catch (e) {
        if (method === 'GET') return e.name === 'TimeoutError' ? 'timeout' : e.code ?? e.message;
      }
    }
  };
  for (let k = 0; k < urls.length; k += 8) {
    const batch = urls.slice(k, k + 8);
    const codes = await Promise.all(batch.map(status));
    batch.forEach((u, j) => { if (codes[j] !== 200) failed.push(`${codes[j]} ${u} (from ${externals.get(u)})`); });
  }
  console.log(`\nexternal: ${urls.length - failed.length}/${urls.length} returned 200`);
  for (const f of failed) console.log(`  ${f}`);
}

if (problems.length) process.exit(1);

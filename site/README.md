# tensorcode.dev

The product launch site for TensorCode, the `tensorcode` libraries for Python and
TypeScript. It is a site for this one product, not a company site: brand everything as
TensorCode, and mention TensaCo only as the plain-text "by TensaCo" in footers.

It's static: no framework, and no build step for the landing page. Cloudflare Workers
serves it as static assets (`../wrangler.jsonc`, Worker `tensorcode-site`). A few lines
in `../scripts/site/worker.js` run first: they redirect `www.tensorcode.dev` and plain
`http` to `https://tensorcode.dev`, and the retired `/how-it-works/` and `/why/` to `/`.

| Path | What it is |
|---|---|
| `index.html` | The landing page: hero, the two problems, the programming model, four live demos, start |
| `DESIGN.md` | The visual system (written = solid, learned = halftone, `--signal` = training). Read it before changing the look |
| `docs/` | **Generated.** Technical landing, shared overview and install pages, Python and TypeScript guides |
| `404.html` | Served for any missing path (`not_found_handling: "404-page"`) |
| `examples/*.py` | The Python programs the demos show. Each runs offline on CPU with `tensorcode[vec]` |
| `assets/base.css` | Tokens (light and dark), reset, top bar, controls, footer: shared by every page |
| `assets/home.css` | Landing page components and the phase grammar |
| `assets/docs.css` | Docs layout, article typography, code highlighting |
| `assets/theme.js` | The light/dark toggle (key `tc-theme`, shared with the docs) |
| `assets/home/*.js` | Landing page behaviour: `code.js` (highlighting, phase gutter, melt/condense, superposed values), one module per section and demo |
| `assets/programs/*.js` | JavaScript twins of `examples/*.py`, run on the bundled library |
| `assets/lib/tensorcode.js` | **Generated, committed.** TensorCode for TypeScript bundled for the browser |
| `media/og.png` | Social preview image (1200×630) |
| `sitemap.xml`, `sitemap-pages.xml`, `robots.txt` | Sitemap index (pages + generated `docs/sitemap.xml`) |

Keep the landing page short: one idea per section, lines people can repeat. Every claim
has to be something the library actually does. Check it against `tensacode/python/docs/`
(especially `validation.md`) before you add it. Technical detail belongs in `/docs/`.

## The live demos run TensorCode

Every demo imports `assets/lib/tensorcode.js`, the TypeScript library itself, and
trains in the visitor's tab. Each demo's program is written twice: `examples/<name>.py`
(the code the page shows, downloadable) and `assets/programs/<name>.js` (the same program
for the browser). The two print the same report, and a check compares them:

```sh
node scripts/site/build-lib.mjs     # rebuild assets/lib/tensorcode.js after the typescript submodule moves
PYTHON=tensacode/python/.venv/bin/python node scripts/site/check-examples.mjs
```

`build-lib.mjs` needs `npm install` in `tensacode/typescript` once (it uses that
checkout's `dist/` and its rolldown). `check-examples.mjs` needs a Python with
`tensorcode[vec]`. Numbers must match exactly, except where float32 rounding compounds
over training (`rules` agrees to within 1%). Run both after changing a demo or its
program, and after moving either submodule.

## Build the docs

`site/docs/` is generated and git-ignored. Build it before you preview or deploy:

```sh
node scripts/site/build-docs.mjs           # writes site/docs/
node scripts/site/build-docs.mjs --check   # writes nothing; exits 1 on a broken internal link or anchor
```

The script has no dependencies. It reads each implementation's README and `docs/*.md`
from the submodules (`tensacode/python`, `tensacode/typescript`) in the reading order
listed at the top of the script. Missing files are skipped, and unlisted `docs/*.md`
files go under "More". A language with no README gets a placeholder introduction.
Relative `.md` links become site URLs, and links written as `https://tensorcode.dev/...`
become site paths. Any other repository link (examples, JSON records, source) points to
the file on GitHub. Code is highlighted at build time.

The pages about both languages, `/docs/overview/` (architecture) and `/docs/install/`
(pip and npm), are Markdown in `scripts/site/pages/`. Their links use site paths such
as `/docs/python/quickstart/`. Consecutive code blocks whose info string has `tab=...`
(```` ```python tab=Python ```` then ```` ```ts tab=TypeScript ````) render as one
tabbed sample. The reader's language is remembered (`tc-lang`), and visiting a
language's guides sets it.

To add a guide, add its `file`/`slug` to that language's `sections` in
`scripts/site/build-docs.mjs`. To add a shared page, add it to `shared`.

After a build, check the whole site:

```sh
node scripts/site/check-links.mjs              # internal links, anchors, code blocks; exits 1 on a problem
node scripts/site/check-links.mjs --external   # also requests every external link once
```

## Preview

```sh
node scripts/site/build-docs.mjs
npx wrangler dev          # http://localhost:8787, with the same 404 and trailing-slash rules as production
```

## Deploy

```sh
node scripts/site/build-docs.mjs && npx wrangler deploy
```

`wrangler.jsonc` attaches the custom domains `tensorcode.dev` and `www.tensorcode.dev`
(zone `tensorcode.dev` in the Cloudflare account, which Cloudflare's nameservers serve)
and keeps the `workers.dev` address. Cloudflare creates and manages the DNS records for
custom domains, but it refuses a hostname that already has an A, AAAA or CNAME record
(error 100117). Delete those records for `tensorcode.dev` and `www` first (never MX or
TXT), then deploy again. `.assetsignore` keeps this README and `DESIGN.md` out of the
upload.

## Checks before shipping

- `node scripts/site/build-docs.mjs --check` passes.
- `check-examples.mjs` passes, if you changed a demo, a program or a submodule.
- No horizontal page scroll at 360px wide (code blocks may scroll inside themselves).
- Both themes: toggle in the corner, or change the OS setting with no stored choice.
- `prefers-reduced-motion`: the hero shows its final state, and demos compute without
  animating. Controls still work.
- `--signal` appears only while something is training.

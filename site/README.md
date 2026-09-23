# tensorcode.dev

The product launch site for TensorCode, the `tensorcode` libraries for Python and
TypeScript. It is a site for this one product, not a company site: brand everything as
TensorCode, and mention TensaCo only as the plain-text "by TensaCo" in footers.

It's static: no framework, and no build step for the marketing pages. Cloudflare Workers
serves it as static assets (`../wrangler.jsonc`, Worker `tensorcode-site`).

| Path | What it is | Audience |
|---|---|---|
| `index.html` | Landing page: hero, three value sections, who it's for, trust strip, CTA | Business readers |
| `how-it-works/` | Four plain steps, what stays in your control, limits, FAQ | Business readers |
| `why/` | Principles, a side-by-side comparison, open-source stance | Business readers |
| `docs/` | **Generated.** Technical landing plus Python and TypeScript guides | Engineers |
| `404.html` | Served for any missing path (`not_found_handling: "404-page"`) | Everyone |
| `assets/base.css` | Tokens (light and dark), reset, top bar, buttons, footer: shared by every page | |
| `assets/site.css` | Marketing components: hero, diagrams, image placeholders, strips | |
| `assets/docs.css` | Docs layout, article typography, code highlighting colours | |
| `assets/theme.js` | The light/dark toggle (key `tc-theme`, shared with the docs) | |
| `assets/site.js` | Copy-prompt buttons, header state, reveal on scroll | |
| `media/` | Generated images; `PROMPTS.md` has the prompts | |
| `sitemap.xml`, `sitemap-pages.xml`, `robots.txt` | Sitemap index (pages + generated `docs/sitemap.xml`) | |

Keep the marketing pages in plain language: short sentences, concrete outcomes, no
jargon. Every claim has to be something the library actually does. Check it against
`tensacode/python/docs/` (especially `validation.md`) before you add it. Technical detail
belongs in `/docs/`.

The header, footer and favicon are repeated in each marketing page (there is no
template step). When you change one, change all four: `index.html`, `how-it-works/`,
`why/` and `404.html`. The docs generator carries its own copy of the header, the mark
and the favicon.

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
Relative `.md` links become site URLs. Any other repository link (examples, JSON
records, source) points to the file on GitHub. Code is highlighted at build time.

To add a guide, add its `file`/`slug` to that language's `sections` in
`scripts/site/build-docs.mjs`.

## Preview

```sh
node scripts/site/build-docs.mjs
npx wrangler dev          # http://localhost:8787, with the same 404 and trailing-slash rules as production
```

## Deploy

```sh
node scripts/site/build-docs.mjs && npx wrangler deploy
```

Attach the `tensorcode.dev` custom domain to the `tensorcode-site` Worker in the
Cloudflare dashboard, or add a `routes` entry to `wrangler.jsonc`. `.assetsignore` keeps
this README and `media/PROMPTS.md` out of the upload.

## Images

Each placeholder box shows its image-generation prompt and a **Copy prompt** button, and
already contains an `<img>` that points at the final path. Generate the image and save it
at that path (for example `site/media/value-3.webp`). On the next load the image
replaces the box: `onload` adds `.has-image`, and a missing file triggers `onerror`, which
keeps the box. All prompts, sizes and paths are listed in [`media/PROMPTS.md`](media/PROMPTS.md).
`media/og.png` (1200×630) is the social preview image and has no placeholder box.

## Checks before shipping

- `node scripts/site/build-docs.mjs --check` passes.
- No horizontal scroll at 360px wide. Diagram cards go full-bleed on phones.
- Both themes: toggle in the corner, or change the OS setting with no stored choice.
- `prefers-reduced-motion`: diagram animations, floating cards and scroll reveals all turn off.

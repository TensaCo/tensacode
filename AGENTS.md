# TensorCode umbrella repository

Guidance for coding agents working in this repository.

## Layout

- `tensacode/<language>/`: git submodules, one repository per language.
  `python` (tensacode-py, the reference implementation) and `typescript`
  (tensacode-ts, a port that shares its file formats) are implemented; `cpp`,
  `csharp`, `dart`, `java`, `kotlin` and `swift` are private placeholder
  repositories, so CI checks out only `python` and `typescript`.
- `site/`: the tensorcode.dev product site, static HTML/CSS/JS with no framework.
  `site/README.md` has the page map, writing rules and checklist.
- `site/docs/`: generated and git-ignored. Never edit it by hand.
- `scripts/site/build-docs.mjs`: builds `site/docs/` from the submodules' `README.md`
  and `docs/*.md`, plus the shared pages in `scripts/site/pages/`.
- `scripts/site/check-links.mjs`: checks internal links, anchors and code blocks.
- `wrangler.jsonc`, `scripts/site/worker.js`: the Cloudflare Worker `tensorcode-site`.
- `.github/workflows/`: `site.yml` (build and link check) and `deploy.yml`
  (deploy on push to `develop`).

## Rules

- Library code and library docs live in the submodule repositories. Change them there,
  not through this repository. Don't run `git submodule update` or commit a moved
  submodule pointer unless the task is to do exactly that.
- The product is TensorCode (packages `tensorcode` on PyPI and npm), at
  https://tensorcode.dev. Brand everything as TensorCode; TensaCo appears only as the
  plain-text "by TensaCo" in site footers.
- Never mention or link `tensaco.ai`. It is reserved for a future company site.
- Every claim on the site must be something the libraries do; check it against
  `tensacode/python/docs/` (especially `validation.md`).
- The default branch is `develop`.

## Build, check, deploy

```bash
node scripts/site/build-docs.mjs      # writes site/docs/
node scripts/site/check-links.mjs     # exits 1 on a broken internal link
npx wrangler dev                      # preview at http://localhost:8787
```

Both checks must pass before a commit that touches the site or the scripts. Deployment
is automatic on push to `develop` (`deploy.yml`); by hand it is
`node scripts/site/build-docs.mjs && npx wrangler deploy`. Don't deploy unless asked.

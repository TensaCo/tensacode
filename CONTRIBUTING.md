# Contributing to TensorCode

Thanks for your interest. TensorCode is split across several repositories, so the
first step is finding the right one.

## Where changes go

| Change | Repository |
|---|---|
| Python library (`pip install tensorcode`) | [TensaCo/tensacode-py](https://github.com/TensaCo/tensacode-py) |
| TypeScript library (`npm install tensorcode`) | [TensaCo/tensacode-ts](https://github.com/TensaCo/tensacode-ts) |
| Library guides shown on tensorcode.dev/docs | The library's own repository (`README.md` and `docs/*.md`) |
| The tensorcode.dev site, the shared docs pages, this README | This repository |

Open issues and pull requests in the repository the change belongs to. Each library
repository explains its own development setup and tests. The TypeScript port follows
the Python reference implementation and shares its file formats, so a behavior change
usually needs both.

The C++, C#, Dart, Java, Kotlin and Swift submodules are placeholders. If you want to
start one, open an issue here first.

## Working on the site

```bash
git clone https://github.com/TensaCo/tensacode
cd tensacode
git submodule update --init tensacode/python tensacode/typescript
node scripts/site/build-docs.mjs      # writes site/docs/ (git-ignored)
node scripts/site/check-links.mjs     # must pass
npx wrangler dev                      # preview at http://localhost:8787
```

Node.js 20 or newer is enough; the scripts have no dependencies. See
[site/README.md](site/README.md) for the layout, the writing rules for the marketing
pages and the pre-ship checklist. Every claim on the site must be something the
libraries actually do.

Pull requests run the same build and link check in CI. Merges to `develop` deploy the
site automatically; please don't deploy by hand.

## Submodule pointers

A pull request here should not move a submodule pointer unless that is its purpose
(for example, publishing new library docs to the site). Say so in the description when
it does.

## License

By contributing you agree that your contributions are licensed under the MIT License
of the repository you contribute to.

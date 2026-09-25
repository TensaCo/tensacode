# tensorcode.dev visual system

One idea runs the whole site: **written and learned computation are two phases of
the same material.** Everything is set in type on a character grid. Written code is
the solid phase. Learned code is the same type in its liquid phase. Training is the
phase change between them, and the site's motion mostly shows that change.

## Phases

| | Written (solid) | Learned (liquid) | Training (signal) |
|---|---|---|---|
| Mark | Solid ink: a 2px bar, a filled square, a hairline | Halftone: a dot field printed in `--learn` (riso blue) | `--signal` vermilion, and nothing else |
| Values | One typeset value | Every possible value, overlaid with opacity equal to its probability. Uncertainty adds blur and offset (`--u`) | Loss figures, gradient pulses, "supervise" marks |
| Motion | Discrete. It snaps, types one character per frame, or steps | Continuous. It eases (`--ease-liquid`), settles, and stops moving once it converges | Travels backward along the gutter, only through lines that can learn |
| Code gutter | `│` solid bar | `┊` dotted bar | a pulse running up the bar |

- Form tells written from learned; colour only reinforces it. The palette is
  risograph spot inks: black for written, blue for learned, fluorescent pink only as
  misregistration in liquid type, and vermilion only while training runs.
- A value that has converged is still. Nothing moves unless it is computing.
- A transition shows the boundary moving: solid text melts into superposition, or
  a superposition collapses into one solid value.

## Type

- **Archivo** (variable width and weight) for display and text. Headlines are
  uppercase, width 112 to 125, weight 800, leading 0.88, tracking -0.01em.
  Running text is width 100, weight 400, 18/1.55.
- **IBM Plex Mono** for code, labels, numbers and controls. Code is 15/1.7
  (13/1.65 on phones). Comments are italic. Numbers are tabular.
- Section labels are mono, uppercase, tracked +0.08em: `01 — EXAMPLES`.

## Code as image

- Code sits on the paper. There are no boxes, shadows or rounded panels, only a
  gutter holding line numbers and the phase bar.
- Highlighting is typographic, not coloured. Keywords are weight 600, strings are
  `--ink-2`, comments are italic `--faint`, and numbers are tabular. Names bound to
  learned operations get a dotted underline, the halftone in miniature.
- Live values print at the end of their line in mono, set off by `→`.

## Grid and space

- 8px base unit. Content is at most 1240px wide on a 12-column grid with 24px
  gutters, and the side gutter is at least 16px.
- Sections are separated by a full-width hairline and 160px of air (96px on
  phones), and each opens with its mono label.

## Colour

Light is warm paper with near-black ink; dark is carbon with bone ink. Three spot
inks sit on top, each with one job:

| Token | Ink | Used for |
|---|---|---|
| `--learn` | riso blue | halftone fields, learned-name underlines, uncertain candidates, the first ghost of liquid type |
| `--ghost-rgb` | fluorescent pink | the second ghost of liquid type (misregistration), nothing else |
| `--signal` | vermilion | training in progress: loss, gradient pulses, `supervise`/`fit` lines |

A converged value sets in black. The tokens are in `assets/base.css`. There are no
gradients, glows or shadows.

## Motion timing

| Event | Timing |
|---|---|
| Written value appears | Instant, or typed at 18ms per character |
| Melt (solid → liquid) | 520ms, `--ease-liquid` |
| Condense (liquid → solid) | 640ms, `--ease-liquid`, with a final 1-frame snap |
| Value moving down a program | 160ms per line |
| Gradient pulse | 420ms per hop, moving upward |
| Training frame | One real optimizer epoch per animation frame, at most 60 per second |

`prefers-reduced-motion`: every demo renders its final state without animating,
and the controls still work.

## Interaction

- Controls look like code: `[ train ]`, a line you can comment out, a value you can
  drag. Hover inverts them (ink ground, paper text).
- The phase slider's thumb goes from a solid square at WRITTEN to a halftone disc
  at LEARNED.
- Each interaction must teach something about the programming model.

## Diagrams

No node graphs. A diagram is made of type, bars and lines. Distributions are the
labels themselves at varying opacity or as halftone bars. Observations are solid
dots, and learned predictions are halftone fields.

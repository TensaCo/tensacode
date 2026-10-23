"""The robots reel behind section 02 of tensorcode.dev: four shots of one recorded run.

The run is the dexterous Dex3-1 controller in the shared MuJoCo warehouse, rendered in Cycles
(sources/dexterous-cinematic.render.json holds the render's settings and the hashes of the recorded
geometry and poses). Times are seconds; `in_` and `out` are times in the source. The video is silent
and loops, so it ends on the placement and starts again on the wide shot.

Open it in the CommandAGI video editor, or rebuild the site's copy with
`node scripts/site/build-media.mjs` (it writes site/media/robots.mp4 and the poster).
"""
from commandagi.design.media import clip, marker, track, transition, video

CINEMATIC = "sources/dexterous-cinematic.mp4"
FADE = 0.6

# (in, out) in the cinematic, in order.
SHOTS = [
    (0.0, 4.2),    # the warehouse: two G1s at their workcells, forklifts and people around them
    (6.0, 9.8),    # the hand closes on the parcel, the pressure skin in contact
    (11.4, 14.6),  # the carry: the parcel held and turned toward the receiving tray
    (17.0, 21.6),  # the placement and release on the tray
]


def shots():
    start = 0.0
    for i, (a, b) in enumerate(SHOTS):
        fade = [transition(kind="crossDissolve", duration=FADE)] if i else []
        yield clip(*fade, src=CINEMATIC, start=start, in_=a, out=b, fit_mode="fill", name=f"Shot {i + 1}")
        start += b - a


result = video(
    track(*shots(), name="Picture"),
    marker(name="poster", time=1.0),
    name="Robots", width=1280, height=720, fps=24, background="#0f0f0e",
)

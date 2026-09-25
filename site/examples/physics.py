"""Write the physics, learn the aim: gradients flow back through written code.

The learned `aim` never sees an angle. Its only supervision is "land on the
target"; the written, replayable `Flight` operation turns that into a gradient.
Runs offline on CPU:  pip install 'tensorcode[vec]'  then  python physics.py
"""
import math
import torch
from tensorcode import trace, training
from tensorcode.ops import Operation
from tensorcode.ops.vec import Latent, Space
from tensorcode.ops.vec.decode import Decode

SPEED, GRAVITY, DRAG, DT, STEPS = 22.0, 9.81, 0.12, 0.1, 120  # 12 s of flight


class Flight(Operation):
    """Written: aim -> landing distance (m), with air drag. No closed-form inverse."""
    replayable = True  # pure tensor code, so replay can recompute it with gradients

    def forward(self, aim, *, context=None):
        angle = torch.pi / 2 * torch.sigmoid(aim)  # a rule: launch between 0 and 90 degrees
        vx, vy = SPEED * torch.cos(angle), SPEED * torch.sin(angle)
        x, y = torch.zeros_like(angle), torch.zeros_like(angle)
        landed = torch.zeros_like(angle)
        for _ in range(STEPS):
            nx, ny = x + vx * DT, y + vy * DT
            vx, vy = vx - DRAG * vx * DT, vy - (GRAVITY + DRAG * vy) * DT
            falling = (y >= 0) & (ny < 0)  # the step that crosses the ground
            share = y / (y - ny).clamp_min(1e-6)
            landed = torch.where(falling, x + share * (nx - x), landed)
            x, y = nx, ny
        return landed


torch.manual_seed(3)
space = Space("target distance", 1)
aim = Decode({"architecture": "mlp", "hidden_dimensions": [16], "input_space": space.configuration(),
              "output_dimensions": 1, "output": "aim (before the angle rule)"})
fly = Flight()


def targets(*meters):
    return Latent(torch.tensor([[m / 40] for m in meters]), space)


goals = [8, 14, 20, 26, 32, 38]
with trace() as t:
    landed = fly(aim(targets(*goals)))
t.supervise(landed, [[m] for m in goals], loss="mse", source="the targets themselves")

trainer = training.Trainer.from_ops({"aim": aim, "fly": fly},
                                    optimizer=lambda p: torch.optim.Adam(p, lr=0.03))
losses = trainer.fit([t], epochs=300)
print(f"loss {losses[0]:.1f} -> {losses[-1]:.3f}")
with torch.no_grad():
    unseen = [11, 23, 35]  # distances it never trained on
    raw = aim(targets(*unseen))
    for meters, a, x in zip(unseen, torch.pi / 2 * torch.sigmoid(raw), fly(raw)):
        print(f"target {meters} m: launch at {math.degrees(a.item()):.1f} deg, lands at {x.item():.2f} m")

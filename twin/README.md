# Digital twin (CPU MuJoCo)

Official SO-101 MJCF, same joints as the 12V arms on order.

## Skills

| Skill | Job | Scripted pass rate (50 episodes) |
| --- | --- | --- |
| `reach` | Touch a sampled point | 26/50 — the sampler puts targets behind the arm |
| `place` | Pick at pad A, set down at pad B | 50/50 |
| `barrier` | Pick at A, lift **over a fence**, set down at B | 50/50 |

`barrier` is the demo job. Fence height, fence distance, can start and target
pad are re-rolled every episode, and any contact with the fence ends the
episode as a failure.

```bash
.venv/bin/python -m twin eval --skill barrier --policy scripted --episodes 50
```

### Straight-down working envelope

The wrist can only aim at the table for x in 0.16–0.32 m, up to roughly 9 cm
above it. Every randomised pad and start position stays inside that band; move
them out and a miss stops being the policy's fault. Re-measure by sweeping
`twin.ik.ik_delta` targets over the fence-free scene.

## Demo video

Public hub: https://robots.fimpo.net · classroom: https://robo-twin.fimpo.net

```bash
.venv/bin/python -m twin.demo_reel
# LinkedIn 4:5 1080×1350 → videos/twin/barrier-demo-linkedin.mp4
# LinkedIn 9:16         → videos/twin/barrier-demo-linkedin-9x16.mp4
# LinkedIn 16:9         → videos/twin/barrier-demo-linkedin-16x9.mp4
```

Upload the **4:5** file to the LinkedIn feed. Three passing episodes with
different fence heights, then `naive_barrier_action` dragging the can straight
across and striking the fence. Chrome is FIMPO / 50/50 / promote ≥ 80%.
Rendering is headless (`MUJOCO_GL=egl`); PNG frames are deleted after encode.

## Teach with the 3D window (like dragging in RViz)

You move a coloured marker. The simulated claw follows. That is you teaching
with graphics. It is not a license.

```bash
python -m twin teach --skill place
```

- Double-click the RGB sticks, then Ctrl-drag — or use arrow keys and `[` `]`.
- `G` close claw, `O` open.
- `K` record motion, `Space` save a still pose, `Enter` write the demo file.
## From a Windows PC

Open **https://robo-twin.fimpo.net**, or the bind address printed by `python -m twin web`.
Drag the picture to move the RGB sticks; the claw follows.

Do not start a second `python -m twin web` on the Linux box — two classrooms
share the preview file and the picture stops matching the buttons.

Replay a demo:

```bash
.venv/bin/python -m twin teach --skill place --replay .cache/twin-demos/place/demo-….json
```

## Train / exam (the code half)

```bash
.venv/bin/python -m twin viewer --skill barrier --policy scripted
.venv/bin/python -m twin train --skill barrier --steps 50000
.venv/bin/python -m twin eval --skill barrier --policy ppo --model <zip> --episodes 50
```

A hand-taught demo is homework. A certificate still requires the train gate in `GOVERNANCE.md`.

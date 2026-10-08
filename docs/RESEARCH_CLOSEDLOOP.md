# Closed-loop SmolVLA × LIBERO — research notes (8 Oct 2026)

## What the community / docs agree on

| Knob | Recommended | Source |
|------|-------------|--------|
| Cameras | `agentview_image` → `observation.images.image`, wrist → `image2` | zuoxingdong/smolvla-libero-eval, roboeval |
| Image prep | 180° rotate (`[::-1, ::-1]`) then float `[0,1]` CHW | ActuallyIR/roboeval, HF issue #1316 |
| State | eef_pos(3) + axis-angle(3) + gripper_qpos(2) = **8D** | same |
| Action | 7D OSC relative `[-1,1]`, gripper `-1` open / `+1` close | smolvla_libero convention |
| Control | `--env.control_mode=relative` | LeRobot LIBERO docs |
| Policy | `n_action_steps=1`, often `num_steps=1` (denoising) | HF blog + #1316 (horizon=1 ≫ chunk) |
| MuJoCo | pin **3.3.2** for official SR; avoid ≥3.4.0 init-state bugs | #3264 / #4390 |
| Renderer | `MUJOCO_GL=egl` or `osmesa` on Linux | HF docs |

Reference checkpoint for *LIBERO-trained* closed-loop: `HuggingFaceVLA/smolvla_libero`.

## Implication for *this* repo

Our finetune used ego `camera1` + **6D** state/action (retarget deltas). That is **not** the LIBERO observation/action layout. Therefore:

1. **Personal-data claim** stays strongest as: ego → retarget → SmolVLA → **open-loop** EE deltas in LIBERO (already shipped) + offline ego MSE.
2. **Closed-loop with our ckpt** is a domain-transfer experiment: map `agentview`→`camera1`, pad/truncate state & action. Expect poor task success; still valuable as an honest video.
3. **Optional reference closed-loop** with `smolvla_libero` demonstrates the stack can do proper reactive LIBERO control (not personal-data-driven).

We do **not** claim paper-level LIBERO SR with our ego finetune.

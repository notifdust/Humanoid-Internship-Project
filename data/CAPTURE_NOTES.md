# Capture notes — pick_place_mug

| Field | Value |
|-------|--------|
| Task | Pick fuchsia mug → place on white plate |
| Instruction | `pick up the fuchsia mug and place it on the white plate` |
| Clips | 42 (`pick_place_mug_01.mp4` … `_42.mp4`) |
| Location | `data/raw/ego/` |
| Labels | `data/labels.csv` (copy also in `data/raw/ego/labels.csv`) |
| Viewpoint | Egocentric |
| Hand | Right |
| Mug | Fuchsia |
| Plate | White |
| Table | Black |
| Resolution | 3840×2160 (4K) |
| Frame rate | ~30 fps |
| Total duration | ~337 s (~5.6 min) |
| Capture date | 2026-10-08 |

## Splits (in `labels.csv`)

| Split | Clips |
|-------|--------|
| train | 01–34 |
| val | 35–38 |
| test | 39–42 |

## Pipeline note

Downstream training should **downsample to 1080p (or 512² crops)** before HaMeR / LeRobot — native 4K is unnecessarily heavy for an RTX 4060.

Original phone filenames → new names: see `data/rename_map.csv`.

from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1] / "third_party" / "LIBERO" / "libero" / "libero"
cfg = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
path = Path.home() / ".libero" / "config.yaml"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.dump(cfg), encoding="utf-8")
print("wrote", path)
print(path.read_text(encoding="utf-8"))

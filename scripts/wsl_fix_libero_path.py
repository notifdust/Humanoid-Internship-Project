from pathlib import Path
import site

sp = Path(site.getsitepackages()[0])
print("site", sp)
for p in sorted(sp.glob("*libero*")):
    print("file", p)
    if p.suffix in {".pth", ".txt"} or "editable" in p.name:
        try:
            print(p.read_text(encoding="utf-8")[:500])
        except Exception as e:
            print("read fail", e)

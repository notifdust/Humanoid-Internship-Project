import importlib

for m in ("torch", "lerobot", "libero", "mujoco", "robosuite"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "FAIL", type(e).__name__, str(e)[:80])

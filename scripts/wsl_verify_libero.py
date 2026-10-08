import mujoco
import robosuite
from libero.libero import benchmark

print("mujoco", mujoco.__version__)
print("robosuite", robosuite.__version__)
print("libero suites:", list(benchmark.get_benchmark_dict().keys()))
print("LIBERO_OK")

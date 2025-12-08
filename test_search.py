from tinygrad  import Tensor
from tinygrad import Tensor, Device, GlobalCounters, TinyJit
import torch
import time
import numpy as np

N = 1024
CNT = 8
save_ops, save_mem = 0, 0

def gemm(a, b, c):
    return a @ b + c

def helper_test_speed(f1, *args):
  global save_ops, save_mem
  ets = []
  ret = None
  for _ in range(CNT):
    del ret
    # operation cache defeats
    args = [(x+1).realize() if isinstance(x, Tensor) else (None if x is None else (x+1)) for x in args]
    args = [(x-1).realize() if isinstance(x, Tensor) else (None if x is None else (x-1)) for x in args]

    # force syncing
    [x.numpy() for x in args if x is not None]

    # manual pre sync
    local_device = Device[args[0].device]
    local_device.synchronize()

    GlobalCounters.global_ops = 0
    GlobalCounters.global_mem = 0
    st = time.perf_counter()
    ret: Tensor = f1(*args)
    local_device.synchronize()
    et = (time.perf_counter() - st) * 1000
    ets.append(et)
    if GlobalCounters.global_ops:
      save_ops, save_mem = GlobalCounters.global_ops, GlobalCounters.global_mem
  return ret.numpy(), np.min(ets)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch_dt = torch.float32
    torch_a = torch.rand(N, N, dtype=torch_dt)
    torch_b = torch.rand(N, N, dtype=torch_dt)
    torch_c = torch.rand(N, N, dtype=torch_dt)
    a = Tensor(torch_a.numpy())
    b = Tensor(torch_b.numpy())
    c = Tensor(torch_c.numpy())
    val_tinygrad, et_tinygrad = helper_test_speed(TinyJit(gemm), *(a,b,c))
    flops = save_ops*1e-6
    mem = save_mem*1e-6
    print(f"{et_tinygrad:7.2f} ms ({flops/et_tinygrad:9.2f} GFLOPS {mem/et_tinygrad:7.2f} GB/s) in tinygrad, {flops:10.2f} MFLOPS {mem:8.2f} MB")  # noqa: E501
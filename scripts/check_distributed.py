"""Two-process CPU check for unequal local backward counts and gradient averaging."""
import torch
from accelerate import Accelerator

from memsearch.training import average_gradients, synchronize

accelerator = Accelerator(cpu=True)
if accelerator.num_processes != 2:
    raise RuntimeError("Run with torchrun --standalone --nproc-per-node=2")
model = torch.nn.Linear(1, 1, bias=False).to(accelerator.device)
model.weight.data.fill_(1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# Rank 0 has one contributing action, rank 1 none. Both must take the same update.
if accelerator.process_index == 0:
    (2 * model(torch.ones(1, 1, device=accelerator.device)).sum()).backward()
average_gradients(model, accelerator)
assert torch.allclose(model.weight.grad, torch.ones_like(model.weight))
optimizer.step()
assert torch.allclose(model.weight, torch.full_like(model.weight, 0.9))
gathered = accelerator.gather(model.weight.detach())
assert torch.allclose(gathered, torch.full_like(gathered, 0.9))
if accelerator.is_main_process:
    print("PASS: 2 CPU ranks, unequal local backward counts, identical averaged update")
synchronize(accelerator)
torch.distributed.destroy_process_group()

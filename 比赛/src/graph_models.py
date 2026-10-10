"""Shared message-passing primitive for the supply-chain graph models."""
import torch


def mean_messages(x,edges):
    result=torch.zeros_like(x)
    if edges.numel()==0:return result
    source,target=edges[:,0],edges[:,1]
    result.index_add_(0,target,x[source])
    degree=torch.zeros((len(x),1),dtype=x.dtype,device=x.device)
    degree.index_add_(0,target,torch.ones((len(target),1),dtype=x.dtype,device=x.device))
    return result/degree.clamp_min(1)


def configure_cpu_runtime():
    # Two loaded OpenMP runtimes hung at a parallel allocation barrier on macOS.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

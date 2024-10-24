# -*- coding: utf-8 -*-
"""Flop_Count_Example.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/facebookresearch/NeuralCompression/blob/main/tutorials/Flop_Count_Example.ipynb

# NeuralCompression Flop Counter Example

Welcome! In this notebook we'll walkthrough using `neuralcompression`'s flop counter to calculate the computational complexity of PyTorch models.

## Setup

First, if you haven't yet intalled `neuralcompression`, do so now with:
"""

# !pip install git+https://github.com/facebookresearch/NeuralCompression/

import torch
from torch import nn

import neuralcompression.functional as ncF
from neuralcompression.models import ScaleHyperprior

"""## Basic Usage

The `neuralcompression` flop counter can be found at `neuralcompression.functional.count_flops`. To get started with the flop counter, two arguments need to be passed:

- The model (an `nn.Module`) that you want to evaluate.
- A list of inputs that the model should be evaluated on.

As an example:
"""

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 32, stride = 2, kernel_size = 5, padding = 2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 10)
)

inputs = [torch.randn(5, 3, 32, 32)]

results = ncF.count_flops(model, inputs)

results

"""The result returned by the flop counter is a 3-tuple:

1. The first element records the total number of flops performed by the model.
2. The second element in the return tuple breaks down this count by operation.
3. The third item returned by the counter records all the operations that the counter didn't know how to count flops for (more detail about these ops below). This dicitionary (or more specifically a [collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter)) maps unknown op names to the number of times those ops were called in the model. In our example, all of the model's operations are supported by the counter, so this counter is empty.

Most common ML operations (e.g. matrix multiplications, convolutions, elementwise arithmetic operations, etc.) already have counter functions registered by either `neuralcompression` or [fvcore](https://github.com/facebookresearch/fvcore), whose flop counter is used under the hood in `neuralcompression`. Other ops, like sigmoid and tanh, whose flop count can vary by platform, are by default ignored, but have estimated counter functions that can be optionally turned on (see the Single-Flop Estimate section for details).

If you need to add support for more ops or override the counter's default implementation, read on!

## Advanced Usage

### How the Counter Works

The flop counter in `neuralcompression` makes heavy use of the counting utilities in [fvcore](https://github.com/facebookresearch/fvcore). Counting a model's flops is a two-step process:

1. Using PyTorch's [TorchScript](https://pytorch.org/docs/stable/jit.html) capabilities, the model is first JIT-traced into a computational graph. Each node in the graph corresponds to an ATen (linear algebra) operation, like matrix multiplications, convolutions, and elementwise operations like additions/subtractions.

2. Every node in the traced graph is iterated over, and if a node is associated with a registered flop-counting function, that function is invoked and the counted flops are added to the model's total.


In this second step, the counter's registered flop functions are stored as a dictionary mapping operator names (e.g. `aten::add`, `aten::matmul`) to counter functions. These functions have a signature of:

```
def my_counter_function(inputs: List[torch._C.Value], outputs: List[torch._C.Value]) -> float
```

where objects of type `torch._C.Value` represent the symbolic inputs and outputs for the node in the graph (i.e. they're not actual concrete `Tensor`s). Check out the `fvcore` codebase for [examples of how to write counter functions](https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/jit_handles.py).

### What is Counted as a Flop

Note that all of the built-in counter functions consider one fused multiply-accumulate (MAC) to be one flop. This means that a matrix multiplication between an `N x K` and a `K x M` matrix is recorded as `N * K * M` flops (since `N * K * M` MACs are performed). For non-fused elementwise arithmetic operations such as addition, subtraction, absolute value, etc., one flop is counted for each element of the output tensor. For example, adding two `N x K` matrices counts as `N * K` flops.

### Single-Flop Estimates

As mentioned in the previous section, simple elementwise operations like addition and multiplication are counted as one flop per output tensor element. More complicated elementwise operations, such as logarithms, exponentiation, and the sigmoid function, are by default unregistered by the flop counter (i.e. these ops do not contribute to the total flop count and are returned as unrecognized ops), since the number of flops performed by these operations can vary by platform and implementation.

If your model contains lots of these kinds of ops and you wish to obtain a rough estimate of their contribution to model complexity, for convenience the counter exposes the flag `use_single_flop_estimates`, which defaults to `False`. If `True`, this flag treats these more complicated elementwise operations like simple arithmetic ops and registers counter functions that count one flop per element of the output tensor. This is likely an undercount of the contribution of these operations, but this undercount may be more informative when analyzing a model than simply ignoring the operations altogether.

If you know exactly how many flops some of these ops should have on your platform, use the `counter_overrides` argument described in the next section.

Using the counter's default supported ops and the added estimate counters, complicated models should be able to pass through the flop counter with few to no unregistered ops, such as the [Scale Hyperprior Model](https://arxiv.org/abs/1802.01436):
"""

big_model = ScaleHyperprior(network_channels=32, compression_channels=64)
_,_, unsupported_ops = ncF.count_flops(big_model, (torch.randn(1, 3, 64, 64),), use_single_flop_estimates=True)

len(unsupported_ops) == 0 # Verifying no unsupported operations

"""### Customizing the Counter

To register additional counter functions or override the counter's default implementation for specific operations, use the `counter_overrides` argument. This argument takes the form of a dictionary mapping ATen operator names to the corresponding counter functions you wish to use. As an example, consider the following simple module:
"""

class MyModule(nn.Module):
    def forward(self,x,y,z):
        return (x + y).abs()

"""By default, the elementwise addition of two tensors of shape `N x M` or taking the absolute vaue of an `N x M` tensor will contribute `N x M` flops to the total computational complexity of a model (since each scalar addition/absolute value is counted as 1 flop):"""

N = 5
M = 32

inp = torch.randn(N, M)

flops,_,_ = ncF.count_flops(MyModule(), (inp, inp, inp))
flops == 2 * N * M

"""However, let's say that you wanted to ignore all the flops coming from addition operations. You could do this as follows:"""

new_flops, _, _ = ncF.count_flops(
    MyModule(),
    (inp, inp, inp),
    counter_overrides = {
        "aten::add": lambda inps,outs: 0.0
    }
)

new_flops == N * M

"""and now we see that only the `N x M` flops from the absolute value, and not the additional `N x M` flops from the addition, are counted.

## Conclusion

For more details on the internals of the flop counter and how to write counter functions, check out [fvcore on GitHub](https://github.com/facebookresearch/fvcore/tree/master/fvcore/nn). Happy counting!
"""


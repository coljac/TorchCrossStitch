# TorchCrossStitch

A simple implementation of cross-stitch networks (Misra+ 2016[^misra]) in Torch.

A cross-stitch unit produces the weighted sum of the outputs of two (or more) neural networks. The intuition is that if two networks are learning different but related tasks (for instance, image classification and segmentation), then they may benefit from incorporating some of the features of the sibling network at various points in the forward pass. 

The `crossstitch.FlexibleCrossStitch` class is instantiated with constructor methods for the various pieces of the child networks which are to be joined with cross-stitch units. For example:

```
class SimpleNet(nn.Module):
    def __init__(self, input_size=100, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x, dimcheck=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

pre = lambda: SimpleNet(input_size=50, output_size=32)
post = lambda: SimpleNet(input_size=32, output_size=1)

fcs = FlexibleCrossStitch([pre, post], split=2)
```

This produces the following network:

```
FlexibleCrossStitch(
  (net_1_1): SimpleNet(
    (fc1): Linear(in_features=50, out_features=64, bias=True)
    (fc2): Linear(in_features=64, out_features=64, bias=True)
    (fc3): Linear(in_features=64, out_features=32, bias=True)
  )
  (net_1_2): SimpleNet(
    (fc1): Linear(in_features=50, out_features=64, bias=True)
    (fc2): Linear(in_features=64, out_features=64, bias=True)
    (fc3): Linear(in_features=64, out_features=32, bias=True)
  )
  (cs1): CrossStitchUnit()
  (net_2_1): SimpleNet(
    (fc1): Linear(in_features=32, out_features=64, bias=True)
    (fc2): Linear(in_features=64, out_features=64, bias=True)
    (fc3): Linear(in_features=64, out_features=1, bias=True)
  )
  (net_2_2): SimpleNet(
    (fc1): Linear(in_features=32, out_features=64, bias=True)
    (fc2): Linear(in_features=64, out_features=64, bias=True)
    (fc3): Linear(in_features=64, out_features=1, bias=True)
  )
)

```

At the CrossStitchUnit, the two 32-vectors will be weighted and then passed to the inputs of the second layer of SimpleNets.

CrossStitch units have an `n x n` matrix of weights. The are initialized with 0.9 down the diagonal and equal values in the other positions such that each row sums to 1 (where the identity matrix implies no contribution from other networks).

[^misra]: http://arxiv.org/abs/1604.03539

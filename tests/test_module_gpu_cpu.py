import torch
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn


class TestModule(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x):

        _s = time.time()
        y = x.unsqueeze(1)
        y = y.cpu()
        print("Forward time: ", time.time() - _s)

        return y


data = torch.Tensor(64,).cuda()
layer = TestModule().cuda()
cudnn.benchmark = True

for i in range(10):
    y= layer(data)

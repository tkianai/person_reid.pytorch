

import torch
import time

data = torch.Tensor(64, 751)

counter = 10
while counter:
    _s = time.time()
    data = data.cuda()
    print("cpu->gpu: ", time.time() - _s)

    _s = time.time()
    data = data.cpu()
    print("gpu->cpu: ", time.time() - _s)

    counter -= 1

import time
import torch
import torch.nn as nn


inputs = torch.randn(64, 751).cuda(0)
targets = torch.randint(0, 751, (64,), dtype=torch.int64).cuda(0)
logsoftmax = nn.LogSoftmax(dim=1)

while True:
    inputs = torch.randn(64, 751).cuda(0)
    targets = torch.randint(0, 751, (64,), dtype=torch.int64).cuda(0)
    logsoftmax = nn.LogSoftmax(dim=1)

    s_1 = time.time()
    log_probs = logsoftmax(inputs)
    print("First step: ", time.time() - s_1)

    s_2 = time.time()
    targets = torch.zeros(log_probs.size()).scatter_(
        1, targets.unsqueeze(1).data.cpu(), 1)
    print("Second step: ", time.time() - s_1)

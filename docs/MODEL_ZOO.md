

# Market1501

| backbone | midneck       | loss              | epochs | gpus | rank-1(w/o re-rank) | mAP(w/o re-rank) |
| -------- | ------------- | ----------------- | ------ | ---- | ------------------- | ---------------- |
| resnet50 | single_bnneck | ce+triplet+center | 300    | 8    | 95.9(94.4)          | 94.1(86.5)       |
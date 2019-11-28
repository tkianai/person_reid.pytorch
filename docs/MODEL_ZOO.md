

# Market1501

| backbone       | midneck       | loss              | lr     | epochs            | gpus | rank-1(w/o re-rank) | mAP(w/o re-rank) | download                                                                    | configuration                     |
| -------------- | ------------- | ----------------- | ------ | ----------------- | ---- | ------------------- | ---------------- | --------------------------------------------------------------------------- | --------------------------------- |
| resnet50       | single_bnneck | ce+triplet+center | 0.0007 | 300[30, 100, 190] | 8    | 95.9(94.4)          | 94.1(86.5)       | [model](https://drive.google.com/open?id=1WNi3J18Gb74LkSVol1dlgYfsMSGzNFSA) | resnet50_best_practice.yaml       |
| resnet50_ibn_a | single_bnneck | ce+triplet+center | 0.001  | 300[30, 100, 230] | 8    | 95.7(95.2)          | 94.3(88.1)       | [model](https://drive.google.com/open?id=1hEhgWFdg5GEpaCldmkZGwO_F8a08Ngdq) | resnet50_ibn_a_best_practice.yaml |
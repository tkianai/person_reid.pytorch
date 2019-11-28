
#!/bin/bash

python run.py --config-file configs/market1501/resnet50_best_practice.yaml --name resnet50_best_practice --test True test.rerank True model.load_weights work_dirs/resnet50_best_practice/model.pth.tar-300
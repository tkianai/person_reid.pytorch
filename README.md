
# person_reid.pytorch

> Baseline model[resnet50] rank-1 achieves **95.9** on market1501!!!

Better performance, parallel training, easier usage!


This projects is my own code base for `person re-identification` research and competitions. The codes are heavily borrowed from [1](https://github.com/KaiyangZhou/deep-person-reid). Based on this, I have eliminated the support of video re-id features, and added many useful training skills or structure strategy. Thanks a lot for the authors whose codes I have used.

**Only based on global feature, currently!**

*Local features are coming, stay tuned!Welcome PR!*

Better performace than deep-person-id, while more features than strong-baseline. Here are the newly features...

## Features

- [x] Lots of backbones supported(resnet, resnet_ibn, senet, densenet, mlfn, mudeep, osnet)
- [x] Lots of losses supported(center loss, triplet loss, focal loss, ranked list loss)
- [x] Lots of heads supported(arcface, cosface, sphere face)
- [x] Lots of midnecks supported(single_bnneck, multi_bnneck)
- [x] Training different parts of the model with different solvers and schedulers(for example, backbone, head or loss(center) would have different learning rates)
- [x] Multi-gpu training(fast and robust...)


## Usage

Easy to train and evaluation, as follows:

### Train

- Train with single gpu

```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config-file <your config file>
```

- Train with multi gpu

```sh
python run.py --config-file <your config file>
```

You can use option `--name` to specify the training work directory, under `work_dirs/<name>`, otherwise `name` would be choosen the default name according to your training configuration. 

- Visulizing loss and learning rate

```sh
# install tensorboard first: pip install tensorboard
tensorboard --logdir <log directory> --port <port> --host <0.0.0.0>
```

Try to visit `<server ip>:<port>`

**Adjust your super-parameters according to training details!**


### Test

- Test with groundtruth-available dataset

```sh
python run.py --config-file <your config file> --test True test.rerank False model.load_weights <checkpoint path>
```

If you want to test with `re-rank`, just set the flag `test.rerank` to `True`.


- Test with groundtruth-un-available dataset

For example, you want to get the most probably same identity images, the distance matrix will be saved as `save_for_test.pkl`, you would get the image names by analyzing this matrix. More details refere to `tools/predict_naic2019.py`.

- Visulize the test results

```sh
python run.py --config-file <your config file> --test True test.rerank False model.load_weights <checkpoint path> test.visrank True
```

`visrank` is the flag to save the visrank results to images, you can find them in the `work_dirs/<name>`.


## MODEL_ZOO

Lots of fancy results are listed here~[MODEL_ZOO](./docs/MODEL_ZOO.md)

**I have no time to test all of these features, any excellent config and results PR would be welcomed!**

## ...

Any advice and discussion would be helpful for me ~!

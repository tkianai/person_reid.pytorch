from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.load_weights = ''  # path to model weights
    cfg.model.resume = ''  # path to checkpoint for resume training
    cfg.model.backbone = CN()
    cfg.model.backbone.name = 'resnet50'
    cfg.model.backbone.last_stride = 1
    # automatically load pretrained model weights if available
    cfg.model.backbone.pretrained = ''
    cfg.model.midneck = CN()
    cfg.model.midneck.name = 'bnneck'
    
    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'datasets'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.split_id = 0  # split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.transforms = ['random_flip']  # data augmentation
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = 'log'  # path to save log

    # specific datasets
    cfg.market1501 = CN()
    # add 500k distractors to the gallery set for market1501
    cfg.market1501.use_500k_distractors = False
    cfg.cuhk03 = CN()
    # use labeled images, if False, use detected images
    cfg.cuhk03.labeled_images = False
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # use cuhk03's metric for evaluation
    cfg.naic2019 = CN()
    cfg.naic2019.phase = 'train'
    cfg.naic2019.total = False

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'
    # number of instances per identity for RandomIdentitySampler
    cfg.sampler.num_instances = 4

    # train
    cfg.train = CN()
    cfg.train.optim_weights = [1.0, 1.0 / 0.0005]
    cfg.train.optim_layers = [None, 'loss.center']    # None stands for the rest
    cfg.train.optims = ['adam', 'sgd']
    cfg.train.lrs = [0.0003, 0.5]
    cfg.train.weight_decays = [5e-4, 5e-4]
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    # layers for training while keeping others frozen
    cfg.train.open_layers = ['classifier']

    cfg.train.lr_scheduler = 'single_step'
    cfg.train.stepsize = [20]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.warmup_factor = 0.01
    cfg.train.warmup_epochs = 10
    cfg.train.warmup_method = "linear"

    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 4453  # random seed

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.softmax.loss_weight = 1.0
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.loss_weight = 1.0
    cfg.loss.center = CN()
    cfg.loss.center.loss_weight = 0.0005

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    # distance metric, ['euclidean', 'cosine']
    cfg.test.dist_metric = 'euclidean'
    # normalize feature vectors before computing distance
    cfg.test.normalize_feature = False
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    # evaluation frequency (-1 means to only test after training)
    cfg.test.eval_freq = -1
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank = False
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    cfg.test.visactmap = False  # visualize CNN activation maps
    cfg.test.feat_after_neck = True

    return cfg


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.data.split_id,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        # image
        'cuhk03_labeled': cfg.cuhk03.labeled_images,
        'cuhk03_classic_split': cfg.cuhk03.classic_split,
        'market1501_500k': cfg.market1501.use_500k_distractors,
        'naic2019_phase': cfg.naic2019.phase,
        'naic2019_total': cfg.naic2019.total
    }


def optimizer_kwargs(cfg):
    return {
        'optims': cfg.train.optims,
        'lrs': cfg.train.lrs,
        'weight_decays': cfg.train.weight_decays,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'print_freq': cfg.train.print_freq,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'use_metric_cuhk03': cfg.cuhk03.use_metric_cuhk03,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank,
        'visactmap': cfg.test.visactmap
    }

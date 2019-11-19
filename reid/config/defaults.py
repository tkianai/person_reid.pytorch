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
    cfg.data.root = 'datasets'
    cfg.data.sources = ['market1501']   # source for training
    cfg.data.targets = ['market1501']   # target for testing
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.split_id = 0  # split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.transforms = ['random_flip']  # data augmentation
    cfg.data.padding = 10   # zero padding the origin image
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = 'work_dirs'  # path to save log

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
    cfg.naic2019.phase = 'train'   # train for training and evaluation, test for predict
    cfg.naic2019.total = False   # whether use the whole training set or not

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'   # Using `randomIdentitysampler` in most cases
    # number of instances per identity for RandomIdentitySampler
    cfg.sampler.num_instances = 4

    # train
    cfg.train = CN()
    cfg.train.max_epoch = 120
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 64
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 4453  # random seed

    # solver
    cfg.solver = CN()
    # layers for training while keeping others frozen, Null stands for open the all layers
    cfg.solver.open_layers = []  
    # None stands for the rest
    cfg.solver.optim_layers = [None, 'loss.center']
    cfg.solver.optim_weights = [1.0, 1.0 / 0.0005]   # rescale the grad
    # NOTE the solver number must equal to optim_layers length
    # param: [solver 1, solver 2, ...]
    cfg.solver.optims = ['adam', 'sgd']
    cfg.solver.lrs = [0.00035, 0.5]
    cfg.solver.weight_decays = [5e-4, 5e-4]
    cfg.solver.lr_schedulers = ['warmup_multi_step', None]
    cfg.solver.step_sizes = [[40, 70], None]
    cfg.solver.gammas = [0.1, None]
    cfg.solver.warmup_factors = [0.01, None]
    cfg.solver.warmup_epochs = [10, None]
    cfg.solver.warmup_methods = ['linear', None]

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

    # head
    cfg.head = CN()
    cfg.head.name = 'softmax'  # [softmax, am_softmax, arc, cos, sphere]

    # loss
    cfg.loss = CN()
    cfg.loss.name = ['crossentropy', 'triplet', 'center']
    cfg.loss.crossentropy = CN()
    cfg.loss.crossentropy.label_smooth = True  # use label smoothing regularizer
    cfg.loss.crossentropy.weight = 1.0
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight = 1.0
    cfg.loss.center = CN()
    cfg.loss.center.weight = 0.0005
    # NOTE `softmax` and `focal` should be mutually exclusive
    cfg.loss.focal = CN()
    cfg.loss.focal.label_smooth = True
    cfg.loss.focal.weight = 1.0
    cfg.loss.ranked = CN()
    cfg.loss.ranked.weight = 0.4

    # test
    cfg.test = CN()
    cfg.test.batch_size = 32
    # distance metric, ['euclidean', 'cosine']
    cfg.test.dist_metric = 'euclidean'
    # normalize feature vectors before computing distance
    cfg.test.normalize_feature = False
    cfg.test.feat_after_neck = True   # using the feature after midneck
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

    return cfg


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'padding': cfg.data.padding,
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
        'optim_layers': cfg.solver.optim_layers,
        'optims': cfg.solver.optims,
        'lrs': cfg.solver.lrs,
        'weight_decays': cfg.solver.weight_decays,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
    }


def lr_scheduler_kwargs(cfg):

    list_kwargs = []
    for i, _ in enumerate(cfg.solver.optim_layers):
        kwargs = {
            'lr_scheduler': cfg.solver.lr_schedulers[i],
            'step_size': cfg.solver.step_sizes[i],
            'gamma': cfg.solver.gammas[i],
            'warmup_factor': cfg.solver.warmup_factors[i],
            'warmup_method': cfg.solver.warmup_methods[i],
            'warmup_epoch': cfg.solver.warmup_epochs[i],
            'max_epoch': cfg.train.max_epoch
        }
        list_kwargs.append(kwargs)
    return list_kwargs


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.solver.open_layers,
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


def get_defeault_exp_name(cfg):
    return '{}_{}_{}_{}_{}_{}'.format(
        cfg.model.backbone.name,
        cfg.model.midneck.name,
        cfg.head.name,
        cfg.loss.name,
        cfg.data.sources,
        cfg.data.targets,
    )

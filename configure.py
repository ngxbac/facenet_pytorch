import torch

class Configure(object):
    def __init__(self):
        super(Configure, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.bs = 16
        self.sz = 224
        self.seed = 1234
        self.log_dir = "log"
        self.optimizer = "adagrad"
        self.n_triplets = 10
        self.lr = 0.1
        self.wd = 0
        self.margin = 0.5
        self.embedding_size = 128
        self.dataroot = "../../input/dlib_gen/"
        self.start_epoch = 0
        self.epochs = 10
        self.lr_decay = 1e-4
        self.log_interval = 10
        self.resume = None

import torch
import torch.optim as optim
import torch.nn as nn
from model import ConvexPotentialLayerNetwork, NormalizedModel
import util
import numpy as np
import torch.backends.cudnn as cudnn
import time
import data as data

cudnn.benchmark = True


class Trainer:

    def __init__(self, config):
        self.cuda = True
        self.seed = config.seed
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.wd = config.weight_decay
        self.depth = config.depth
        self.depth_linear = config.depth_linear
        self.save_dir = config.save_dir
        self.conv_size = config.conv_size
        self.num_channels = config.num_channels
        self.n_features = config.n_features
        self.margin = config.margin
        self.lln = config.lln
        self.dataset = config.dataset
        self.norm_input = config.norm_input

    def set_everything(self):
        torch.manual_seed(self.seed)

        # Init dataset
        if self.dataset == "c10":
            self.mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
            self.std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
            num_classes = 10
        else:
            self.mean = (0.5071, 0.4865, 0.4409)
            self.std = (0.2673, 0.2564, 0.2762)
            num_classes = 100

        if not self.norm_input:
            self.std = (1., 1., 1.)

        self.data = data.DataClass(self.dataset, batch_size=self.batch_size)
        self.train_batches, self.test_batches = self.data()

        ## Init model
        self.model = ConvexPotentialLayerNetwork(depth=self.depth,
                                                 depth_linear=self.depth_linear,
                                                 num_classes=num_classes, conv_size=self.conv_size,
                                                 num_channels=self.num_channels,
                                                 n_features=self.n_features,
                                                 use_lln=self.lln)

        self.model = NormalizedModel(self.model, self.mean, self.std)
        self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model = self.model.cuda()
        print(self.model)

        ## Init optimizer
        lr_steps = self.epochs * len(self.train_batches)
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=self.wd, lr=self.lr)
        self.lr_scheduler = util.TriangularLRScheduler(self.optimizer, lr_steps, self.lr)
        self.criterion = lambda yhat, y: util.margin_loss(yhat, y, self.margin)
        print(f"margin loss with param {self.margin} with lr = {self.lr}")
        print(f"number of gpus: {torch.cuda.device_count()}")

    def __call__(self):
        self.set_everything()
        acc_best = 0
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            self.train(epoch)
            print(f'Epoch {epoch}: training 1 epoch -> compute eval')

            self.time_epoch = time.time() - start_time
            print(f"Time epoch: {self.time_epoch}")

            acc, _ = self.test(epoch)
            if (acc > acc_best):
                acc_best = acc
                torch.save(self.model.state_dict(), f"{self.save_dir}/best_model.pt")

        torch.save(self.model.state_dict(), f"{self.save_dir}/last_model.pt")

    def train(self, epoch):
        self.model.train()
        for batch_idx, batch in enumerate(self.train_batches):
            self.lr_scheduler.step(epoch * len(self.train_batches) + (batch_idx + 1))
            images, target = batch['input'], batch['target']
            predictions = self.model(images)
            loss = self.criterion(predictions.cpu(), target.cpu())
            loss = loss.cuda()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                    epoch, batch_idx * len(images), 50000,
                           100. * batch_idx / len(self.train_batches), loss.item()))

    def test(self, epoch):
        if not self.lln:
            acc, cert_acc, _, _, _ = util.certified_accuracy(self.test_batches, self.model,
                                                             lip_cst=1. / np.min(self.std), eps=36. / 255)
        else:
            acc, cert_acc, _, _, _ = util.certified_accuracy_LLN(self.test_batches, self.model,
                                                                 lip_cst=1. / np.min(self.std), eps=36. / 255)
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        print(f"Epoch {epoch}: Accuracy : {acc}, certified accuracy: {cert_acc}, lr = {lr}\n")
        return acc, cert_acc

    def eval_final(self, eps=36. / 255):

        self.model.eval()
        # acc_cert
        lip_cst = 1. / np.min(self.std)
        if not self.lln:
            acc, cert_acc, _, _, _ = util.certified_accuracy(self.test_batches, self.model, lip_cst=lip_cst, eps=eps)
        else:
            acc, cert_acc, _, _, _ = util.certified_accuracy_LLN(self.test_batches, self.model, lip_cst=lip_cst,
                                                                 eps=eps)
        # autoattack
        with torch.no_grad():
            acc_auto = util.test_auto_attack(self.model, self.test_batches, eps=eps)

        # acc pgd
        eps_iter = 2. * eps / 10.
        if eps == 0:
            acc_pgd = acc
        else:
            acc_pgd = util.test_pgd_l2(self.model, self.test_batches, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                       eps=eps, nb_iter=10, eps_iter=eps_iter, rand_init=True, clip_min=0.0,
                                       clip_max=1.0, targeted=False)
        print(f"Final for epsilon={eps}: Accuracy : {acc}, certified accuracy: {cert_acc}, "
              f"autoattack accuracy: {acc_auto}, pgd attack accuracy: {acc_pgd}")

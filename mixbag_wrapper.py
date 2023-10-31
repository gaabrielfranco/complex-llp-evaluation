from llp_learn.base import baseLLPClassifier
import torch
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from scipy.stats import norm

from tqdm import tqdm


def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input + eps)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = cross_entropy_loss(input, target, eps=self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss


class ConfidentialIntervalLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, min_value, max_value):
        mask = torch.where((pred <= max_value) & (pred >= min_value), target, pred)
        loss = cross_entropy_loss(mask, target, eps=self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_rampup_weight(weight, iteration, rampup):
    alpha = weight * sigmoid_rampup(iteration, rampup)
    return alpha


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
        d = torch.randn_like(x)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")

        return lds


class GaussianNoise(nn.Module):
    """add gasussian noise into feature"""

    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        zeros_ = torch.zeros_like(x)
        n = torch.normal(zeros_, std=self.std)
        return x + n


class PiModelLoss(nn.Module):
    def __init__(self, std=0.15):
        super(PiModelLoss, self).__init__()
        self.gn = GaussianNoise(std)

    def forward(self, model, x):
        logits1 = model(x)
        probs1 = F.softmax(logits1, dim=1)
        with torch.no_grad():
            logits2 = model(self.gn(x))
            probs2 = F.softmax(logits2, dim=1)
        loss = F.mse_loss(probs1, probs2, reduction="sum") / x.size(0)
        # return loss, logits1
        return loss

class DynamicWeight(object):
    def __init__(self, lam, K=3, T=1):
        self.num_loss = 3
        self.loss_t1 = [None, None, None]
        self.loss_t2 = [None, None, None]
        self.w = [None, None, None]
        self.e = [None, None, None]

        self.lam = lam

        self.K, self.T = K, T
        for w in self.lam:
            if w == 0:
                self.K -= 1

    def calc_ratio(self):
        for k in range(self.num_loss):
            if self.lam[k] != 0:
                self.w[k] = self.loss_t1[k] / self.loss_t2[k]
                self.e[k] = math.e ** (self.w[k] / self.T)
            else:
                self.e[k] = 0

        for k in range(self.num_loss):
            self.lam[k] = self.K * self.e[k] / sum(self.e)

    def __call__(self, loss_nega, loss_posi, loss_MIL):
        loss = [loss_nega, loss_posi, loss_MIL]
        for k in range(self.num_loss):
            self.loss_t2[k] = self.loss_t1[k]
            self.loss_t1[k] = loss[k]

        # t = 3, ...
        if self.loss_t2[0] is not None:
            self.calc_ratio()

        return self.lam


def consistency_loss_function(
    args, consistency_criterion, model, train_loader, img, epoch
):
    if args.consistency != "none":
        consistency_loss = consistency_criterion(model, img)
        consistency_rampup = 0.4 * epoch * len(train_loader) / args.batch_size
        alpha = get_rampup_weight(0.05, epoch, consistency_rampup)
        consistency_loss = alpha * consistency_loss
    else:
        consistency_loss = torch.tensor(0.0)
    return consistency_loss

def calculate_prop(output, nb, bs):
    output = F.softmax(output, dim=1)
    output = output.reshape(nb, bs, -1)
    lp_pred = output.mean(dim=1)
    return lp_pred

def ci_loss_interval(
    proportion1: list,
    proportion2: list,
    sampling_num1: int,
    sampling_num2: int,
    confidence_interval: float,
):
    a: float = sampling_num1 / (sampling_num1 + sampling_num2)
    b: float = sampling_num2 / (sampling_num1 + sampling_num2)
    t = norm.isf(q=confidence_interval)
    cover1 = t * np.sqrt(proportion1 * (1 - proportion1) / sampling_num1)
    cover2 = t * np.sqrt(proportion2 * (1 - proportion2) / sampling_num2)
    expected_plp = a * proportion1 + b * proportion2
    confidence_area = t * cover1 + b * cover2
    ci_min_value = expected_plp - confidence_area
    ci_max_value = expected_plp + confidence_area
    return ci_min_value, ci_max_value, expected_plp


class MixBag(baseLLPClassifier):
    """
        MixBag wrapper to llp-learn.
    """
    def __init__(self, lr=0.01, epochs=100, consistency="vat", pretrained=True, patience=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.pretraied = pretrained
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_train = ConfidentialIntervalLoss()
        self.loss_val = ProportionLoss()
        self.epochs = epochs

        # dataloader
        # self.train_loader = load_data(args, stage="train")
        # self.val_loader = load_data(args, stage="val")
        # self.test_loader = load_data(args, stage="test")

        # early stopping parameters
        self.val_loss = None
        self.cnt = 0
        self.best_val_loss = float("inf")
        self.break_flag = False
        self.best_path = None
        #self.fold = args.fold
        self.patience = patience
        #self.output_path = args.output_path
        #self.test_acc = None

        # Consistency loss
        if consistency == "none":
            self.consistency_criterion = None
        elif consistency == "vat":
            self.consistency_criterion = VATLoss()
        elif consistency == "pi":
            self.consistency_criterion = PiModelLoss()
        else:
            raise NameError("Unknown consistency criterion")
        
    def model_import(self, pretrained, channels, classes):
        model = resnet18(pretrained=pretrained)
        if model:
            if channels != 3:
                model.conv1 = nn.Conv2d(
                    channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            model.fc = nn.Linear(model.fc.in_features, classes)
            model = model.to(self.device)
        return model 

    def fit(self, X, bags, proportions):
        if len(proportions.shape) == 1:
            n_classes = 2
        else:
            n_classes = proportions.shape[1]
        
        self.model = self.model_import(self.pretraied, X.shape[1], n_classes)
        self.model.train()
        for epoch in range(self.epochs):
            losses = []
            for batch in tqdm(self.train_loader, leave=False):
                # nb: the number of bags, bs: bag size, c: channel, w: width, h: height
                nb, bs, c, w, h = batch["img"].size()
                img = batch["img"].reshape(-1, c, w, h).to(self.device)
                lp_gt = batch["label_prop"].to(self.device)
                ci_min_value, ci_max_value = batch["ci_min_value"], batch["ci_max_value"]

                # Consistency loss
                consistency_loss = consistency_loss_function(
                    self.args,
                    self.consistency_criterion,
                    self.model,
                    self.train_loader,
                    img,
                    epoch,
                )

                output = self.model(img)
                lp_pred = calculate_prop(output, nb, bs)

                loss = self.loss_train(
                    lp_pred,
                    lp_gt,
                    ci_min_value.to(self.device),
                    ci_max_value.to(self.device),
                )
                loss += consistency_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())

            train_loss = np.array(losses).mean()
            print("[Epoch: %d/%d] train loss: %.4f" % (epoch + 1, self.epochs, train_loss))
            break_flag = self.early_stopping()
            if break_flag:
                break


    def predict(self, X):
        pass


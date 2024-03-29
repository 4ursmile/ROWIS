import torch
from torch.nn import functional as F
from torch import nn
import math
from torchvision.ops import DeformConv2d
def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float=0.25, gamma: float=2.0, num_classes: int=81, empty_weight: float = 0.1):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    W = torch.ones(num_classes, dtype=prob.dtype, layout=prob.layout, device=prob.device)
    W[-1] = empty_weight
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=W)
    
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0,1)
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()
        
    def forward(self, x):
        out=self.flatten(x)
        out=self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2
class ProbObjectnessHeadBlock(nn.Module):
    def __init__(self, hidden_dim, activation = None, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten(0,1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        if activation == None:
            activation = nn.ReLU()
        self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*expansion_factor),
                activation,
                nn.Linear(hidden_dim*expansion_factor, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.prob_head = ProbObjectnessHead(hidden_dim)
    def forward(self, x):
        out = self.flatten(x)
        out = self.norm1(out)
        ff_out = self.ff(out)
        out = self.norm2(ff_out)
        out = self.dropout2(out)
        prob = self.prob_head(out)
        return prob
class ProbObjectnessHeadBlockStack(nn.Module):
    def __init__(self, hidden_dim, activation = None, final_activation = None , expansion_factor=2, dropout=0.1, num_blocks=4):
        super().__init__()
        self.blocks = _get_clones(ProbObjectnessHeadBlock(hidden_dim, activation, expansion_factor, dropout), num_blocks-1)
        if final_activation == None:
            final_activation = nn.Sigmoid()
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.final_activation = final_activation
    def forward(self, x):
        probs = []
        for block in self.blocks:
            x = block(x)
            probs.append(x)
        # Apply final activation to the stacked output
        probs = torch.stack(probs)
        probs = self.pooling(probs).squeeze(-1).squeeze(-1)
        probs = self.final_activation(probs)
        return probs
class FullProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim, device='cpu'):
        super().__init__()
        self.flatten = nn.Flatten(0,1)
        self.momentum = 0.1
        self.device = device
        self.hidden_dim = hidden_dim
        self.obj_mean = nn.Parameter(torch.zeros(hidden_dim, device=device), requires_grad=False)
        self.obj_cov = nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
        self.inv_obj_cov = nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
    def update_parameters(self, x):
        out = self.flatten(x).detach()
        obj_mean = out.mean(dim=0)
        obj_cov = torch.cov(out.T)
        self.obj_mean.data = self.obj_mean*(1-self.momentum) + obj_mean*self.momentum
        self.obj_cov.data = self.obj_cov*(1-self.momentum) + obj_cov*self.momentum
        return
    def update_icov(self):
        self.inv_obj_cov.data = torch.inverse(self.obj_cov.detach().cpu(), rcond=1e-6).to(self.device)
        return
    def mahalanobis(self, x):
        x = self.flatten(x)
        diff = x - self.obj_mean
        m = (diff*torch.matmul(self.inv_obj_cov, diff.T).T).sum(dim=-1)
        return m.unflatten(0, x.shape[:2])
    def set_momentum(self, momentum):
        self.momentum = momentum
        return 
    def forward(self, x):
        if self.training:
            self.update_parameters(x)
            self.update_icov()
        return self.mahalanobis(x)
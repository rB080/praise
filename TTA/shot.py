import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class SHOT(nn.Module):
    """
    SHOT adapts the classifier head via information maximization,
    keeping the feature extractor frozen.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt_shot(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


def entropy_loss(p):
    return -(p * torch.log(p + 1e-6)).sum(dim=1).mean()


def diversity_loss(p):
    p_mean = p.mean(dim=0)
    return (p_mean * torch.log(p_mean + 1e-6)).sum()


@torch.enable_grad()
def forward_and_adapt_shot(x, model, optimizer):
    model.eval()
    with torch.no_grad():
        feat = model.forward_feature(x)  # assumes method to extract features
    outputs = model.classifier(feat)
    probs = F.softmax(outputs, dim=1)

    ent_loss = entropy_loss(probs)
    div_loss = diversity_loss(probs)
    loss = ent_loss - div_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return outputs


def collect_classifier_params(model):
    """Collect parameters of the classifier head only."""
    return list(model.classifier.parameters()), [n for n, _ in model.classifier.named_parameters()]


def configure_model_shot(model):
    """Freeze the feature extractor, enable classifier head params."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def check_model_shot(model):
    has_classifier = hasattr(model, 'classifier') and len(list(model.classifier.parameters())) > 0
    assert has_classifier, "Model must have a classifier attribute."
    assert all(p.requires_grad for p in model.classifier.parameters()), \
        "Classifier parameters should require grad."
    assert all(not p.requires_grad for n, p in model.named_parameters() if 'classifier' not in n), \
        "Feature extractor should be frozen."

import torch
from pathlib import Path
import os


class EarlyStopping:

    def __init__(self, model_name, maximize, patience=10, model_dir='./checkpoint/'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.model_dir = model_dir
        self.model_name = model_name
        self.maximize = maximize
        self.save_dir = os.path.join(model_dir, f"{self.model_name}.bin")
        self.update = False
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def step(self, acc, model, epoch, save=True):
        if self.maximize:
            return self._step_max(acc, model, epoch, save)
        else:
            return self._step_min(acc, model, epoch, save)

    def _step_max(self, acc, model, epoch, save):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = epoch
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(model)
            self.counter = 0
            self.best_epoch = epoch
        return self.early_stop

    def _step_min(self, acc, model, epoch, save):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = epoch
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(model)
            self.counter = 0
            self.best_epoch = epoch
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_dir)

    def remove_checkpoint(self):
        os.remove(self.save_dir)

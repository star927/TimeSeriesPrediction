import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self):
        self.patience = 3
        self.counter = 0
        self.early_stop = False
        self.min_vali_loss = np.Inf

    def __call__(self, vali_loss, model, path):
        if vali_loss < self.min_vali_loss:
            self.save_checkpoint(vali_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, path):
        print(f"Validation loss decreased ({self.min_vali_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.min_vali_loss = val_loss


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

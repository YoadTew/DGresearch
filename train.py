import argparse

import torch
from torch import nn

from models.resnet import resnet18_fixed


def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    return parser.parse_args()

class Trainer:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = resnet18_fixed(pretrained=True, classes=args.n_classes)
        self.model = model.to(device)

        self.train_loader = get_train_loader(args)
        self.val_loader = get_train_loader(args)
        self.test_loader = get_train_loader(args)

        self.optimizer = get_optimizer(args)
        self.scheduler = get_scheduler(args)

        self.criterion = nn.CrossEntropyLoss()

    def _do_epoch(self, epoch_idx):
        correct = 0
        total = 0
        self.model.train()

        for batch_idx, (images, targets) in enumerate(self.train_loader):

            images, targets = images.to(self.device), targets.to(self.device)

            # compute output
            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        self.model.eval()
        with torch.no_grad():
            pass
            # Run val - 3 domains
            # Run test - target domain

    def do_test(self, loader):
        pass

    def do_training(self):

        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self._do_epoch(self.current_epoch)

        # val_res = results["val"]
        # test_res = results["test"]
        # idx_best = val_res.argmax()
        # print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        #     val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        # self.logger.save_best(test_res[idx_best], test_res.max())

        return self.model


def main():
    args = get_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
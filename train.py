import argparse

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchfunc
import numpy as np
import random

from models.resnet import resnet18
from models.resnet_permute import resnet18_permuted
from data.data_manager import get_train_loader, get_test_loader, get_val_loader

pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]

def logdir_name(args):
    model_type = 'permute' if args.permute else 'resent'
    log_dir = f'logs/{model_type}_target_{args.target}_p_{args.permute_precent}_fixed_perm'

    return log_dir

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=pacs_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=pacs_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')

    parser.add_argument('--permute', action='store_true', help='Use permuted resent')
    parser.add_argument("--permute_precent", "-p", type=float, default=0.4, help="Permute precent")
    parser.add_argument('--use_alpha', action='store_true', help='Use alpha from beta distribution')

    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--data_dir", default='D:\\Work\\DomainGeneralization\\data', help="Datasets dir path")
    parser.add_argument("--log_dir", default='logs', help="Logs dir path")

    return parser.parse_args()

class Trainer:

    def __init__(self, args, device):
        args.log_dir = logdir_name(args)
        self.args = args
        self.device = device

        if args.permute:
            model = resnet18_permuted(pretrained=args.pretrained, num_classes=args.n_classes)
        else:
            model = resnet18(pretrained=args.pretrained, num_classes=args.n_classes)

        self.model = model.to(device)

        self.train_loader = get_train_loader(args)
        self.val_loader = get_val_loader(args)
        self.test_loader = get_test_loader(args)

        self.optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=int(args.epochs * .8))

        self.criterion = nn.CrossEntropyLoss()

        self.results = {}

        self.writer = SummaryWriter(log_dir=str(args.log_dir))

    def _do_epoch(self, epoch_idx):
        correct = 0
        total = 0
        self.model.train()

        for batch_idx, (images, targets) in enumerate(self.train_loader):

            images, targets = images.to(self.device), torch.tensor(torch.squeeze(targets), dtype=torch.long).cuda()#targets.to(self.device)

            # compute output
            self.optimizer.zero_grad()
            outputs = self.model(images, self.args)

            classification_loss = self.criterion(outputs, targets)

            loss = classification_loss

            if batch_idx % 30 == 1:
                print(f'epoch:  {epoch_idx}/{self.args.epochs}, batch: {batch_idx}/{len(self.train_loader)}, loss: {loss.item()}')

            self.writer.add_scalar('loss_train', loss.item(), epoch_idx * len(self.train_loader) + batch_idx)

            loss.backward()
            self.optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        self.model.eval()
        with torch.no_grad():
            # Run val - 3 domains
            total = len(self.val_loader.dataset)
            class_correct = self.do_test(self.val_loader)
            class_acc = float(class_correct) / total
            self.results['val'][epoch_idx] = class_acc
            print(f'Validation Accuracy: {class_acc}')

            self.writer.add_scalar('val_accuracy', class_acc, epoch_idx)

            # Run test - target domain
            total = len(self.test_loader.dataset)
            class_correct = self.do_test(self.test_loader)
            class_acc = float(class_correct) / total
            self.results['test'][epoch_idx] = class_acc
            print(f'Test Accuracy: {class_acc}')

            self.writer.add_scalar('test_accuracy', class_acc, epoch_idx)

    def do_test(self, loader):
        class_correct = 0

        for i, (inputs, labels) in enumerate(loader, 1):
            inputs, labels = inputs.to(self.device), torch.tensor(torch.squeeze(labels), dtype=torch.long).cuda()

            # forward
            outputs = self.model(inputs, self.args)

            loss = self.criterion(outputs, labels)
            _, cls_pred = outputs.max(dim=1)

            class_correct += torch.sum(cls_pred == labels)

        return class_correct


    def do_training(self):
        print("Target domain: {}".format(self.args.target))
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self._do_epoch(self.current_epoch)

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        # self.logger.save_best(test_res[idx_best], test_res.max())

        self.writer.close()

        return test_res.max()

def main():
    args = get_args()
    args.source = ['art_painting', 'cartoon', 'photo']
    args.target = 'sketch'
    args.pretrained = True
    args.use_alpha = True
    args.permute = True

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # experiments
    target_options = ['art_painting', 'cartoon', 'photo', 'sketch']
    log_results = {}

    for target in target_options:
        args.target = target
        args.source = [x for x in target_options if x != target]

        trainer = Trainer(args, device)
        best_test_acc = trainer.do_training()

        log_results[target] = best_test_acc

    print(log_results)

if __name__ == "__main__":
    torchfunc.cuda.reset()
    torch.backends.cudnn.benchmark = True
    main()
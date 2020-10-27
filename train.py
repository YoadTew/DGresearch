import argparse

import torch
from torch import nn, optim

from models.resnet import resnet18_fixed
from data.data_manager import get_train_loader, get_test_loader, get_val_loader

pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=pacs_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=pacs_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=222, help="Image size")

    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--data_dir", default='D:\\Work\\DomainGeneralization\\data', help="Datasets dir path")

    return parser.parse_args()

class Trainer:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = resnet18_fixed(pretrained=False, num_classes=args.n_classes)
        self.model = model.to(device)

        self.train_loader = get_train_loader(args)
        self.val_loader = get_val_loader(args)
        self.test_loader = get_test_loader(args)

        self.optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=int(args.epochs * .8))

        self.criterion = nn.CrossEntropyLoss()

        self.results = {}

    def _do_epoch(self, epoch_idx):
        correct = 0
        total = 0
        self.model.train()

        for batch_idx, (images, targets) in enumerate(self.train_loader):

            images, targets = images.to(self.device), torch.tensor(torch.squeeze(targets), dtype=torch.long).cuda()#targets.to(self.device)

            # compute output
            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            if batch_idx % 30 == 1:
                print(f'epoch:  {epoch_idx}/{self.args.epochs}, batch: {batch_idx}/{len(self.train_loader)}, loss: {loss.item()}')
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

            # Run test - target domain
            total = len(self.test_loader.dataset)
            class_correct = self.do_test(self.test_loader)
            class_acc = float(class_correct) / total
            self.results['test'][epoch_idx] = class_acc
            print(f'Test Accuracy: {class_acc}')

    def do_test(self, loader):
        class_correct = 0

        for i, (inputs, labels) in enumerate(loader, 1):
            inputs, labels = inputs.to(self.device), torch.tensor(torch.squeeze(labels), dtype=torch.long).cuda()

            # forward
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            _, cls_pred = outputs.max(dim=1)

            class_correct += torch.sum(cls_pred == labels)

        return class_correct


    def do_training(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

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
    args.source = ['art_painting', 'cartoon', 'photo']
    args.target = 'sketch'
    print("Target domain: {}".format(args.target))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
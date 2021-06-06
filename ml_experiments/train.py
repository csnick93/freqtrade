from comet_ml import Experiment
from typing import Dict
from torch.optim.lr_scheduler import OneCycleLR
import copy
import os
import time
import torch

import dataloader
import models


def _train_model(model, criterion, optimizer, lr_scheduler, dataloaders,
                 num_epochs: int, dataset_sizes: Dict[str, int], experiment):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += int(torch.sum(preds == labels.data))
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            experiment.log_metric(f"{phase} loss", epoch_loss, epoch=epoch)
            experiment.log_metric(f"{phase} accuracy", epoch_acc, epoch=epoch)

            print(f'{phase.capitalize()} Loss: {epoch_loss}')
            print(f'{phase.capitalize()} Acc: {epoch_acc}')

            # deep copy the model if we reached new best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def train(model, dataloaders, lr: float, num_epochs: int, steps_per_epoch: int,
          criterion, dataset_sizes: Dict[str, int], experiment):
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = OneCycleLR(optimizer,
                              max_lr=lr,
                              epochs=num_epochs,
                              steps_per_epoch=steps_per_epoch)
    model_ft = _train_model(model,
                            criterion,
                            optimizer,
                            lr_scheduler,
                            dataloaders,
                            num_epochs=num_epochs,
                            dataset_sizes=dataset_sizes,
                            experiment=experiment)
    torch.save(model_ft, 'best_model.pth')
    experiment.log_model('buy_prediction_model', 'best_model.pth')

    return model_ft


if __name__ == '__main__':
    model = models.CNN()
    batch_size = 128
    dataloaders, dataset_sizes = dataloader.get_dataloaders(batch_size, cnn=True)
    lr = 1e-3
    num_epochs = 10
    steps_per_epoch = int(dataset_sizes['train'] / batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    experiment = Experiment(os.environ['COMET_API_KEY'],
                            project_name='freqtrade_buy_prediction',
                            workspace='csnick93')
    train(model, dataloaders, lr, num_epochs, steps_per_epoch, criterion,
          dataset_sizes, experiment)

import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    """This method will train the model and return the model with trained weights loaded inside the model

    Args:
        model ([dict]): Resnet backbone and DeepLab as classifier
        criterion : Loss function used to calculate an error between ground truth and predictions. For that the efficient Mean squared
                    Error Loss function is used.
        dataloaders : Used to load the data from a single folder be that images of masks and load inside the training process.
                    While loading the data, data augmentation is done simultaneously.
        optimizer : Inorder to apply the gradient descent to correctly initialize the weights with respect to the changing loss function
                    Adam optimizer is used.
        metrics : Inorder to judge the performance of the algorithm, F1 score is calculated which is based on precision and recall.
        bpath : This path stores the weights, logs and all other parameters that are recorded during the training process.
        num_epochs : Total number of passes the data should have to pass through the learning process of its features.

    Returns:
        model: Model dict with learned parameters stored in it.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    # Storing the metrics parameters in a log file and writing it in experimental dir creted in root fir of repo
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Iterating through the number of epochs provided as an arg to the main function.
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        # Batch summary stores the training_loss, test_loss and the calculated F1 score
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                # Transfering the images and masks to the device GPU for faster computation
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # Making the Adam optimizer parameters zero before beginning the back propagation processs
                # This step is necassary, as the the optimizer stores the parameters(weight values of previous layers) during the back prop process.
                optimizer.zero_grad()

                # track history if only in train
                # The below flag is turned on in order to enable the gradient calculation process
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    # y_pred represents the predicted mask calculated as an output to our pretrained model
                    # y_true represents the mask provided as the ground truth while training
                    # y_pred will be used to output the segmentation output
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        # loss.backward() calculates the weights by calculating change in loss d(loss) for every parameter (dx). 
                        # Thats X_new = d(loss) / d(x)
                        loss.backward()
                        # The newly calculated parameters X1_new, X2_new, ... are then updated replacing the previous parameters X1, X2
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            # Storing the Lowest loss
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
    # Calculating the total time required for training after exiting from the main loop of number of epochs
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

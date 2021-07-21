from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data

import datahandler
from model import createDeepLabv3
from trainer import train_model

# Click decorator is used to create argument parsers to the function. (Actually find it better than argparse so not changing it)
@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
def main(data_directory, exp_directory, epochs, batch_size):
    """The main function takes in the arguments which are provided by the user while running the main.py file

    Args:
        data_directory (path): Path to the main data dir which will contain the Images/Masks dirs
        exp_directory (path): Name of the dir where logs and weights will be stored. The dir will be created in the root of repo
        epochs ([int]): Default set to 25.
        batch_size ([int]): Default to 4, However as the number of images are in abundunt, suggested value is 25+
    """
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the Mean Squared Error loss function
    # reduction specifies if the total loss is divided by the total number of training examples and hence calculating the mean of total loss
    # if reduction = "sum", then the divide by the total training examples is avoided and the losses are added
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main()

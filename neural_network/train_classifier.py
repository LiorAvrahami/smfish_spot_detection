import torch
print("has cuda:", torch.cuda.is_available())
import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from tqdm.notebook import tqdm
import torch.optim as optim
import pickle
import datetime

import os
import sys

try:
    import this_is_root
except:
    os.chdir(os.path.pardir)  # change workdir to be root dir
    sys.path.insert(0, os.path.realpath("."))

import create_training_data.training_data_generator
from neural_network.model_classifier import spots_classifier_net
from sklearn.model_selection import train_test_split

torch.cuda.is_available()
np.random.seed(0)

loss_function = nn.BCEWithLogitsLoss()
TAG = 1
IMG = 0


class MyDataset(Dataset):
    def __init__(self, img_array, label_array):
        # convert to tensor
        self.img_array = img_array
        self.label_array = label_array

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, idx):
        img = torch.tensor(self.img_array[idx].copy()).float()
        img = img.permute(3, 2, 1, 0)
        label = torch.tensor(self.label_array[idx])
        return img, label


def train_valid_loop(Nepochs, learning_rate=0.001, batch_size=100, save_model_interval=1500, epoch_report_interval=100, my_seed=0, add_name_str=''):
    
    # Validation Constants
    val_generator = create_training_data.training_data_generator.ClassifierValidationDataGenerator()
    val_data = val_generator.get_next_batch()
    val_img = val_data[IMG]
    val_label = val_data[TAG]
    val_ds = MyDataset(val_img, val_label)
    valid_dl = DataLoader(val_ds, batch_size=len(val_label), shuffle=True)
    
    net = spots_classifier_net()

    start_time = datetime.datetime.now()
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    params_str = 'lr-' + str(learning_rate) + '_seed-' + str(my_seed) + add_name_str

    imgs_generator = create_training_data.training_data_generator.ClassifierTrainingDataGenerator.make_default_training_data_generator(
        batch_size=batch_size)
    imgs = imgs_generator.get_next_batch()
    train_img, train_label = imgs[IMG], imgs[TAG]

    train_ds = MyDataset(train_img, train_label)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    train_loss = []
    valid_loss = []
    epochs = []

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # use Adam

    # Check for GPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print('Found GPU!')
        device = torch.device("cuda:0")

    net.to(device)  # put it on the device

    pickle_output_filename = 'run_statistics_spot_detection_' + params_str + '.pickle'
    print("Pickle file: " + pickle_output_filename)
    # try:
    #     os.remove(pickle_output_filename)
    #     print(f"File '{pickle_output_filename}' deleted successfully.")
    # except FileNotFoundError:
    #     print(f"File '{pickle_output_filename}' not found.")

    for epoch in range(Nepochs):  # loop over Nepochs
        # for epoch in tqdm(range(Nepochs)):# loop over Nepochs
        epochs.append(epoch)
        # Training
        net.train()

        xb, yb = next(iter(train_dl))
        # select ony the first value of yb (dot/not dot), the rest of the values represent presence of spots in the channels
        yb = torch.tensor([lst[0] for lst in yb])

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()  # make sure the gradients are zeroed-out each time!

        pred = net(xb)  # pass the input through the net to get the prediction
        loss = loss_function(pred, yb)  # use the MSE loss between the prediction and the target
        loss.backward()
        optimizer.step()  # optimizer step in the direction of negative gradient

        # take the average of the loss over each batch and append it to the list
        train_loss.append(loss.item())

        # Validation
        net.eval()

        xb, yb = next(iter(valid_dl))
        # edit yb again, but for the validation section
        yb = torch.tensor([lst[0] for lst in yb])  # select only the first element

        xb = xb.to(device)  # move the validation input to the device
        yb = yb.to(device)  # move the validation target to the device

        pred = net(xb)  # same as in training loop
        loss = loss_function(pred, yb)  # same as in training loop
        valid_loss.append(loss.item())

        # Model checkpointing
        if np.mod(epoch, save_model_interval) == 0 and epoch > 0:
            if valid_loss[-1] < min(valid_loss[:-1]):
                torch.save(net.state_dict(), 'final_saved_classifier_model_' + params_str + '.pt')
            with open(pickle_output_filename, "wb+") as f:
                pickle.dump({
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "epoch": epochs,
                    "time": datetime.datetime.now() - start_time,
                    "learning_rate": learning_rate,
                    "seed": my_seed,
                }, f)
        if np.mod(epoch, epoch_report_interval) == 0:
            print("Epoch: ", epoch, "(", "{:.2f}".format(100 * epoch / Nepochs),
                  "%) Train loss: ", train_loss[-1], " Valid loss: ", valid_loss[-1])

    # Bring net back to CPU
    net.cpu()

    return net


if __name__ == "__main__":
    b_single = True
    if b_single:
        # single run
        train_valid_loop(batch_size=3000, Nepochs=201,
                         learning_rate=1e-3, save_model_interval=200,
                         my_seed=7, add_name_str='single')

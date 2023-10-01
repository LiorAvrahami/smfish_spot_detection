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
from spot_classifier_NN.classifier_model import spots_classifier_net

from spot_classifier_NN.confusion_matrix import get_confusion_matrix

torch.cuda.is_available()
np.random.seed(0)

loss_function = nn.BCEWithLogitsLoss()
TAG = 3
SMALL_COORDS = 1
IMG = 0
BATCH_SIZE_MUL_FACTOR = 1


class MyDataset(Dataset):
    def __init__(self, img_array, small_coordinates_array, label_array):
        # convert to tensor
        self.img_array = img_array
        self.label_array = label_array
        self.small_coordinates_array = small_coordinates_array

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, idx):
        img = torch.tensor(self.img_array[idx].copy()).float()
        img = img.permute(3, 2, 1, 0)
        coords = torch.tensor(self.small_coordinates_array[idx])
        label = torch.tensor(self.label_array[idx]).float()
        return (img, coords), label


def train_valid_loop(num_channels, Nepochs, learning_rate, batch_size, my_seed=None, add_name_str='', b_plot_data=True, b_use_empty_backgrounds=False, net_to_continue_training=None):

    # Validation Constants
    val_generator = create_training_data.training_data_generator.ClassifierValidationDataGenerator(num_channels)
    val_data = val_generator.get_next_batch()
    val_img = val_data[IMG]
    val_small_coordinates = val_data[SMALL_COORDS]
    val_label = val_data[TAG]
    val_ds = MyDataset(val_img, val_small_coordinates, val_label)
    valid_dl = DataLoader(val_ds, batch_size=len(val_label), shuffle=True)

    if net_to_continue_training is None:
        net = spots_classifier_net(num_channels)
    else:
        net = net_to_continue_training

    start_time = datetime.datetime.now()
    if my_seed is not None:
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

    if not b_use_empty_backgrounds:
        imgs_generator = create_training_data.training_data_generator.ClassifierTrainingDataGenerator.make_default_training_data_generator(
            batch_size=batch_size, num_channels=num_channels)
    else:
        imgs_generator = create_training_data.training_data_generator.ClassifierTrainingDataGenerator.make_training_data_generator_with_empty_backgrounds(
            batch_size=batch_size, num_channels=num_channels)

    train_loss = []
    valid_loss = []
    epochs = []
    train_tp_list = []
    train_fp_list = []
    train_tn_list = []
    train_fn_list = []
    valid_tp_list = []
    valid_fp_list = []
    valid_tn_list = []
    valid_fn_list = []

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # use Adam

    # Check for GPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print('Found GPU!')
        device = torch.device("cuda:0")

    net.to(device)  # put it on the device
    params_str = f"empty_bg={b_use_empty_backgrounds}_lr-{learning_rate}_batchsz-{batch_size}_num_channels-{num_channels}_seed-{my_seed}__{add_name_str}"
    pickle_output_filename = 'run_statistics_spot_detection_' + params_str + '.pickle'
    print("Pickle file: " + pickle_output_filename)

    for epoch in range(Nepochs):  # loop over Nepochs
        generated_data = imgs_generator.get_next_batch()
        train_img, train_small_coordinates, train_label = generated_data[IMG], generated_data[SMALL_COORDS], generated_data[TAG]

        train_ds = MyDataset(train_img, train_small_coordinates, train_label)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # for epoch in tqdm(range(Nepochs)):# loop over Nepochs
        epochs.append(epoch)
        # Training
        net.train()

        loss_cumulative = []
        b_has_spot_cumulative = []
        pred_prob_cumulative = []
        for i in range(BATCH_SIZE_MUL_FACTOR):
            image_and_small_coords, b_has_spot = next(iter(train_dl))
            image, small_coords = image_and_small_coords

            # plot for debugging
            if b_plot_data:
                import matplotlib.pyplot as plt
                idx_no_spot = np.argmin(b_has_spot)
                idx_yes_spot = np.argmax(b_has_spot)
                z = 2
                _, axes = plt.subplots(num_channels + 1, 2)
                axes[0, 0].set_title(f"NoSpot\n{small_coords[idx_no_spot][-1]}")
                for i in range(num_channels):
                    max_no_spot = torch.max(image[idx_no_spot][i, z])
                    axes[i, 0].imshow(image[idx_no_spot][i, z, :, :], vmax=max_no_spot)
                ch = small_coords[idx_no_spot][-1]
                other_z = torch.mean(image[idx_no_spot][ch, [z - 2, z + 2], :, :], dim=0)
                max_no_spot = torch.max(image[idx_no_spot][ch, z])
                axes[-1, 0].imshow(other_z, vmax=max_no_spot)

                axes[0, 1].set_title(f"YesSpot\n{small_coords[idx_yes_spot][-1]}")
                for i in range(num_channels):
                    max_yes_spot = torch.max(image[idx_yes_spot][i, z])
                    axes[i, 1].imshow(image[idx_yes_spot][i, z, :, :], vmax=max_yes_spot)
                ch = small_coords[idx_yes_spot][-1]
                other_z = torch.mean(image[idx_yes_spot][ch, [z - 2, z + 2], :, :], dim=0)
                max_yes_spot = torch.max(image[idx_yes_spot][ch, z])
                axes[-1, 1].imshow(other_z, vmax=max_yes_spot)
                plt.pause(0.01)

            # image is the small cropped image around some roi,
            # small_coords is the position of the roi relative to the crop,
            # b_has_spot is the tag and can be 0 or 1
            image = image.to(device)
            b_has_spot = b_has_spot.to(device)
            small_coords = small_coords.to(device)

            optimizer.zero_grad()  # make sure the gradients are zeroed-out each time!

            pred = net(image, small_coords)  # pass the input through the net to get the prediction
            loss = loss_function(pred, b_has_spot)  # use the MSE loss between the prediction and the target

            loss.backward()

            optimizer.step()  # optimizer step in the direction of negative gradient

            pred_prob = torch.sigmoid(pred)

            pred = pred.cpu()
            b_has_spot = b_has_spot.cpu()
            b_has_spot = b_has_spot.detach().numpy()
            pred_prob = pred_prob.cpu()
            pred_prob = pred_prob.detach().numpy()
            b_has_spot_cumulative += list(b_has_spot)
            pred_prob_cumulative += list(pred_prob)

            # take the average of the loss over each batch and append it to the list
            loss_cumulative.append(loss.item())

        train_loss.append(np.mean(loss_cumulative))
        # tp tn fp fn for validation
        tp, fp, fn, tn = get_confusion_matrix(pred_prob_cumulative, b_has_spot_cumulative)
        train_tp_list.append(tp)
        train_fp_list.append(fp)
        train_tn_list.append(tn)
        train_fn_list.append(fn)

        # Validation
        net.eval()

        b_has_spot_cumulative = []
        pred_prob_cumulative = []
        loss_cumulative = []
        for image_and_small_coords, b_has_spot in valid_dl:
            image, small_coords = image_and_small_coords

            image = image.to(device)  # move the validation input to the device
            small_coords = small_coords.to(device)  # move the validation input to the device
            b_has_spot = b_has_spot.to(device)  # move the validation target to the device

            pred = net(image, small_coords)  # same as in training loop
            loss = loss_function(pred, b_has_spot)  # same as in training loop
            loss_cumulative.append(loss.item())

            pred_prob = torch.sigmoid(pred)

            pred = pred.cpu()
            b_has_spot = b_has_spot.cpu()
            b_has_spot = b_has_spot.detach().numpy()
            pred_prob = pred_prob.cpu()
            pred_prob = pred_prob.detach().numpy()
            b_has_spot_cumulative += list(b_has_spot)
            pred_prob_cumulative += list(pred_prob)

        valid_loss.append(np.mean(loss_cumulative))
        # tp tn fp fn for validation
        tp, fp, fn, tn = get_confusion_matrix(pred_prob_cumulative, b_has_spot_cumulative)
        valid_tp_list.append(tp)
        valid_fp_list.append(fp)
        valid_tn_list.append(tn)
        valid_fn_list.append(fn)

        print("Epoch: ", epoch, "(", "{:.2f}".format(100 * epoch / Nepochs),
              "%) Train loss: ", train_loss[-1], " Valid loss: ", valid_loss[-1])

        # Model checkpointing
        if epoch > 0:
            if valid_loss[-1] < min(valid_loss[:-1]):
                torch.save(net.state_dict(), 'best_saved_classifier_model_' + params_str + '.pt')
            with open(pickle_output_filename, "wb+") as f:
                pickle.dump({
                    "valid_true_positive": valid_tp_list,
                    "valid_false_positive": valid_fp_list,
                    "valid_false_negative": valid_fn_list,
                    "valid_true_negative": valid_tn_list,
                    "train_true_positive": train_tp_list,
                    "train_false_positive": train_fp_list,
                    "train_false_negative": train_fn_list,
                    "train_true_negative": train_tn_list,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "epoch": epochs,
                    "time": datetime.datetime.now() - start_time,
                    "learning_rate": learning_rate,
                    "seed": my_seed,
                }, f)

    # Bring net back to CPU
    net.cpu()

    return net

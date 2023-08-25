import numpy as np
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader
import tqdm  # insted of tqdm notebook
import torch.optim as optim
from create_training_data.training_data_generator import NoiseReductionTrainingDataGenerator
from custom_dataset import CustomDataset


def create_fresh_batch(num_of_elements=32):
    PicGenerator = NoiseReductionTrainingDataGenerator.make_default_training_data_generator(num_of_elements)
    PicsBatch = PicGenerator.get_next_batch()
    train_dl = CustomDataset(PicsBatch)
    train_dl = DataLoader(train_dl, batch_size=num_of_elements, shuffle=True)
    return train_dl


def train_valid_loop(net, num_of_batches=1000, Nepochs=40, learning_rate=0.001):
    train_loss = []
    valid_loss = []

    ### Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # use Adam

    ### Check for GPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print('Found GPU!')
        device = torch.device("cuda:0")

    net.to(device)  # put it on the device

    for epoch in tqdm.tqdm(range(Nepochs)):  # loop over Nepochs

        ### Training
        net.train()
        train_loss_epoch = []
        for i in range(num_of_batches):
            train_dl = create_fresh_batch()
            for xb, yb in tqdm.tqdm(train_dl):  # xb and yb are tensors
                xb = xb.unsqueeze(1)
                yb = yb.unsqueeze(1)
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()  # ...please make sure the gradients are zeroed-out each time!
                pred = net(xb)  # pass the input through the net to get the prediction
                loss = func.mse_loss(pred, yb)  # use the MSE loss between the prediction and the target
                # --> <-- ADD LINE TO DO BACKPROPAGATION HERE ::: DONE! next line
                loss.backward()
                train_loss_epoch.append(loss.item())
                optimizer.step()  # ...please make the optimizer step in the direction of negative gradient

        ### take the average of the loss over each batch and append it to the list
        train_loss.append(np.mean(train_loss_epoch))

        ### Validation
        net.eval()

        valid_loss_epoch = []
        valid_dl = create_fresh_batch()
        for xb, yb in tqdm.tqdm(valid_dl):  # which dataloader?
            xb = xb.unsqueeze(1)
            yb = yb.unsqueeze(1)
            xb = xb.to(device)  # move the validation input to the device
            yb = yb.to(device)  # move the validation target to the device
            pred = net(xb)  # same as in training loop
            loss = func.mse_loss(pred, yb)  # same as in training loop
            valid_loss_epoch.append(loss.item())

        valid_loss.append(np.mean(valid_loss_epoch))

        ### Model checkpointing
        if epoch > 0:
            if valid_loss[-1] < min(valid_loss[:-1]):
                torch.save(net.state_dict(), 'saved_model.pt')

        print('Epoch: ', epoch, ' Train loss: ', train_loss[-1], ' Valid loss: ', valid_loss[-1])

    # Bring net back to CPU
    net.cpu()

    return train_loss, valid_loss

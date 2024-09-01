import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from tqdm.auto import tqdm

from typing import Dict, List

def plot_loss_curves(results : Dict[str, List[float]]) -> None:
    """Plots both test and train loss curves given a results dictionary
    
    Parameters
    ----------
    results : Dict[str : List[float]]
        dictionary of results

    Returns
    -------
    None
        Outputs a test/train loss curve
    """
    train_loss = results['train_loss']
    train_acc = results['train_acc']
    test_loss = results['test_loss']
    test_acc = results['test_acc']
    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 7))

    #Displays Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    #Dispalys Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend();


def train_step(model : torch.nn.Module,
               dataloader : torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device : str) -> List[float]:
    """Trains model using inputted dataloader and returns loss and accuracy
    
    Parameters
    ----------
    model : torch.nn.Module
        model of neural network
    dataloader : torch.utils.data.DataLoader
        train data in torch dataloader
    loss_fn: torch.nn.Module
        loss function 
    optimizer: torch.optim.Optimizer
        optimizer
    device: str
        cuda or cpu device to put tensors on
    Returns
    -------
    List[float]
        train loss and accuracy
    """
    #Set to train mode
    model.train()
    #Set both loss and accuracy to 0
    train_loss = 0 
    train_acc = 0

    #iterate through dataloader by batches
    for batch, (x, y) in enumerate(dataloader):
        #put onto device
        x = x.to(device)
        y = y.to(device)
        #forward pass
        y_pred = model(x)
        #calculate loss and accumlated loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        #reset optimizer
        optimizer.zero_grad()
        #loss backward
        loss.backward()
        #optimizer step
        optimizer.step()
        y_pred_label = torch.argmax(y_pred, dim = 1)
        train_acc += (y_pred_label == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model : torch.nn.Module,
              dataloader : torch.utils.data.DataLoader, 
              loss_fn : torch.nn.Module,
              device : str) -> List[float]:
    """Tests model using inputted dataloader and returns test loss and accuracy
    
    Parameters
    ----------
    model : torch.nn.Module
        model of neural network
    dataloader : torch.utils.data.DataLoader
        test data in torch dataloader
    loss_fn: torch.nn.Module
        loss function 
    Returns
    -------
    List[float]
        test loss and accuracy
    """
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for batch, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            test_pred = model(x)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            test_pred_labels = torch.argmax(test_pred, dim = 1)
            test_acc = ((test_pred_labels == y).sum().item()/len(test_pred_labels))
             
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          device : str = "cpu",
          epochs: int = 5):
    """Both trains and tests model after each epoch
    
    Parameters
    ----------
    model : torch.nn.Module
        model of neural network
    dataloader : torch.utils.data.DataLoader
        data in torch dataloader
    loss_fn: torch.nn.Module
        loss function 
    optimizer: torch.optim.Optimizer
        optimizer
    device: str
        cuda or cpu device to put tensors on 
    epochs: int
        numbers of epochs for training/testing loop to undergo
    Returns
    -------
    Dict[str : float]
        Dictionary of train/test loss/accuracy
    """
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device = device)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device = device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results
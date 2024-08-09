import matplotlib.pyplot as plt
from typing import Dict, List

def plot_loss_curves(results : Dict[str, List[float]]) -> None:
    """Plots both test and train loss curves given a results dictionary
    
    Parameters
    ----------
    results : Dict[str : List[float]]
        dictionary of results

    Returns
    -------
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
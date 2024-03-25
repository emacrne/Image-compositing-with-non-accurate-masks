import torch
from pathlib import Path
import random
import matplotlib.pyplot as plt
import math
from torch.nn import MSELoss


def compute_psnr(original, reconstructed, max_val=1.0):
    mse = MSELoss(original, reconstructed)
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse.item())
    return psnr


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          n: int,
                          display_shape: bool = True,
                          seed: int = None):
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, bg, mask, gt = dataset[targ_sample][0], dataset[targ_sample][1], dataset[targ_sample][2], dataset[targ_sample][3]

        targ_image_adjust = targ_image.permute(1, 2, 0)
        bg = bg.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)
        gt = gt.permute(1, 2, 0)

        plt.subplot(4, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.subplot(4, n, n+i+1)
        plt.imshow(bg)
        plt.subplot(4, n, 2*n + i + 1)
        plt.imshow(mask, cmap="gray")
        plt.subplot(4, n, 3 * n + i + 1)
        plt.imshow(gt)
    plt.show()


def plot_loss_curves(results: dict[str, list[float]], model_name:str):

    train_loss = results['train_loss']
    #test_loss = results['test_loss']

    test_loss = results['test_loss']
    #test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss', c="b")
    plt.plot(epochs, test_loss, label='test_loss', c="g")
    plt.title('cae3')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    """plt.subplot(1, 2, 2)
    plt.plot(epochs, test_loss, label='test_loss')
    #plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Test')
    plt.xlabel('Epochs')
    plt.legend()"""
    plt.savefig("foo"+ model_name+".png", bbox_inches='tight')
    plt.show()
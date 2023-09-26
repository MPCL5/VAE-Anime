import torch
import matplotlib.pyplot as plt

from train import get_data_loaders

RESULT_MODEL_PATH = './StackedEncoder.model'
DEVICE = torch.device('cuda')


def plot_images(real, generated):
    num_rows = 4
    num_cols = 10
    plt.figure(figsize=(8, 8))
    for i in range(real.shape[0]):
        plt.subplot(num_rows, num_cols, i + 1)
        # Convert to numpy and transpose for RGB format
        plt.imshow(real[i].permute(1, 2, 0).int().cpu().numpy())
        plt.axis('off')
        plt.title(f'Real {i + 1}')

    for i in range(generated.shape[0]):
        plt.subplot(num_rows, num_cols, i + 21)
        # Convert to numpy and transpose for RGB format
        plt.imshow(generated[i].permute(1, 2, 0).int().cpu().numpy())
        plt.axis('off')
        plt.title(f'Generated {i + 1}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    training_loader, val_loader, test_loader = get_data_loaders()
    model = torch.load(RESULT_MODEL_PATH)
    model.eval()

    samples = next(iter(training_loader)).float().to(DEVICE)
    resulst = model(samples)

    plot_images(samples, resulst)

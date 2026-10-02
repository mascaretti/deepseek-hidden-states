from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    imagenet_path="/projects/colt/imagenet21k_resized/imagenet21k_train/"
    # path_file = "./imagenet_closest_class_rank_0.txt"
    path_file = "./imagenet_same_class.txt"
    line_number=353

    save_name = "imagenet_same_class.png"
    # save_name = "imagenet_closest_class.png"

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_dl = datasets.ImageFolder(imagenet_path, transform=transformations)
    with open(path_file, 'r') as f:
        lines = f.readlines()
    # process the images through the model        

    img1_path, img2_path = lines[line_number].strip().split()
    img1 = imagenet_dl.loader(img1_path)
    img2 = imagenet_dl.loader(img2_path)
    img1_tensor = transformations(img1).unsqueeze(0)
    img2_tensor = transformations(img2).unsqueeze(0)

    # plot img1 and img2 side by side
    img1 = img1_tensor.squeeze().permute(1, 2, 0).numpy()
    img2 = img2_tensor.squeeze().permute(1, 2, 0).numpy()
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im = axes[0].imshow(img1, vmin=0, vmax=1.1, cmap='viridis')
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')
    axes[0].set_title(img1_path.split('/')[-2])
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title(img2_path.split('/')[-2])
    plt.tight_layout()
    plt.savefig(save_name)

    

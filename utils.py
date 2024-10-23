import os
import torch
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    im.save(path)



def preprocess_csv(csv_file, samples_per_label=100):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Initialize an empty list to hold the sampled data
    sampled_list = []

    # Loop through each label (0-25)
    for label in range(26):
        # Filter the DataFrame for rows with the current label
        label_df = df[df.iloc[:, 0] == label]

        # Randomly sample 100 data points from the current label
        label_sample = label_df.sample(n=samples_per_label, random_state=42)

        # Append the sampled data to the list
        sampled_list.append(label_sample)

    # Concatenate all the sampled data into a single DataFrame
    sampled_df = pd.concat(sampled_list, ignore_index=True)

    return sampled_df




class CustomCSVImageDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        # Load the CSV file
        self.data_frame = data_frame
        self.transform = transform
    
    def __len__(self):
        # Return the total number of samples
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Get the row of data
        row = self.data_frame.iloc[idx]
        # The label is the first column
        label = int(row[0])
        # The image is the remaining 784 columns, converted to a NumPy array and reshaped to 28x28
        image = np.array(row[1:], dtype=np.float32).reshape(28, 28, 1)  # Adding channel dimension for grayscale
        
        # Convert to PIL Image to apply torchvision transforms
        image = torchvision.transforms.ToPILImage()(image)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data(args):
    sampled_df = preprocess_csv(args.dataset_path, samples_per_label=100)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Single channel normalization for grayscale
    ])
    
    # Create the custom dataset using your CSV file
    dataset = CustomCSVImageDataset(data_frame=sampled_df, transform=transforms)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
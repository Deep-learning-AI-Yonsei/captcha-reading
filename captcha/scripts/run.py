import os

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from captcha.data_processing.data_split import split_dataset
from captcha.data_processing.custom_dataset import load_data
from captcha.model.resnet_based import CaptchaResNet, save_param

def run_training_loop(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_loader, val_loader, test_loader = load_data(args.splited_dir, transform, args.batch_size, args.max_dataset_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example setup
    num_classes = len('0123456789abcdefghijklmnopqrstuvwxyz')  # Adjust as necessary
    sequence_length = 5  # Adjust based on your CAPTCHA length
    model = CaptchaResNet(num_classes, sequence_length)
    model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Assuming 'model' is already defined and modified for the correct output
    model.train()
    for epoch in range(args.num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Assuming labels are now [batch_size, sequence_length] with class indices
            outputs = model(images)  # outputs should be [batch_size, sequence_length, num_classes]
            loss = 0
            for i in range(sequence_length):  # Loop over each character position
                loss += criterion(outputs[:, i, :], labels[:, i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}')

    # Save the model
    if not os.path.exists(args.weights_dir): 
        os.makedirs(args.weights_dir)
    save_param(model.state_dict(), args.weights_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Directory path
    parser.add_argument("--original_dir", type=str, default="./captcha/data/samples")
    parser.add_argument("--splited_dir", type=str, default="./captcha/data/dataset")
    parser.add_argument("--weights_dir", type=str, default="./weights")

    # Hyperparam
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_dataset_size", type=int, default=856)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()

    split_dataset(args.original_dir, args.splited_dir)
    
    run_training_loop(args)

if __name__ == "__main__":
    main()
import os

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.transforms import functional

from captcha.data_processing.data_split import split_dataset
from captcha.data_processing.custom_dataset import load_data
from captcha.model.resnet_based import CaptchaResNet, save_param
from captcha.scripts.eval import visualize_and_save_predictions, evaluate_model, load_model

from PIL import Image
import numpy as np
import random
import cv2

# Function to calculate padding
def get_padding(img):
    width, height = img.size
    if width > height:
        padding = (0, (width - height) // 2, 0, (width - height) - (width - height) // 2)
    else:
        padding = ((height - width) // 2, 0, (height - width) - (height - width) // 2, 0)
    return padding

# Custom transform to add padding
class PadToSquare(object):
    def __call__(self, img):
        padding = get_padding(img)
        return functional.pad(img, padding, fill=0, padding_mode='constant')

# Randomly apply filter
class RandomFilter:    
    def __init__(self, filters, p=None):
        """
        Args:
            filtes (list): list of filter function
            p (list, optional): probability of filter usage, equal if None
        """
        self.filters = filters
        self.p = p

    def __call__(self, img):
        chosen_filter = random.choices(self.filters, weights=self.p, k=1)[0]
        return chosen_filter(img)

# Implement Box filter
class Blur:
    def __init__(self, kernel_size = 5):
        self.kernel_size = kernel_size
    
    def __call__(self, img):
        img = np.array(img)
        img = cv2.blur(img, (self.kernel_size, self.kernel_size))
        return Image.fromarray(img)

# Randomly apply distortion
class RandomDistort:
    def __init__(self, distort, p=None):
        """
        Args:
            distort (list): list of distortion function
            p (list, optional): probability of distortion usage, equal if None
        """
        self.distort = distort
        self.p = p

    def __call__(self, img):
        chosen_distort = random.choices(self.distort, weights=self.p, k=1)[0]
        return chosen_distort(img)

# Implement Perspective distortion
class Perspective:
    def __init__(self, scale=0.2):
        self.scale = scale

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]

        # coordinates before transformation
        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

        # coordinates after transformation
        dst_points = np.float32([
            [0 + random.uniform(-self.scale, self.scale) * w, 0 + random.uniform(-self.scale, self.scale) * h],
            [w - 1 + random.uniform(-self.scale, self.scale) * w, 0 + random.uniform(-self.scale, self.scale) * h],
            [0 + random.uniform(-self.scale, self.scale) * w, h - 1 + random.uniform(-self.scale, self.scale) * h],
            [w - 1 + random.uniform(-self.scale, self.scale) * w, h - 1 + random.uniform(-self.scale, self.scale) * h]
        ])

        # calculate transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # distort
        img = cv2.warpPerspective(img, M, (w, h))
        return Image.fromarray(img)

# Implement No Operation
class NoOp:
    def __call__(self, img):
        return img

def run_training_loop(args):
    transform = transforms.Compose([
    # EnsureRGB(),                       # Ensure the image is in RGB format
    PadToSquare(),                     # Add padding to make it square
    transforms.Resize((224, 224)),     # Final resize to the desired input size for the model
    transforms.RandomRotation(degrees=45), # Randomly select rotation angle
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4), # Randomly select color change 
    RandomFilter([Blur(kernel_size=5),  NoOp()], p = [0.5, 0.5]), # Randomly select filter
    RandomDistort([Perspective(scale = 0.2), NoOp()], p = [0.5, 0.5]), # Randomly select distortion
    transforms.ToTensor(),             # Convert image to tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    train_loader, val_loader, test_loader = load_data(args.splited_dir, transform, args.batch_size, args.max_dataset_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example setup
    characters = '0123456789abcdefghijklmnopqrstuvwxyz'
    
    num_classes = len('0123456789abcdefghijklmnopqrstuvwxyz')  # Adjust as necessary
    sequence_length = 5  # Adjust based on your CAPTCHA length
    model = CaptchaResNet(num_classes, sequence_length)
    model.to(device)
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.eval == 0:
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

            if (epoch + 1) % 10 == 0:
                # Save the model
                if not os.path.exists(args.weights_dir): 
                    os.makedirs(args.weights_dir)
                save_param(epoch ,model.state_dict(), args.weights_dir)


        evaluate_model(model, val_loader, device, characters)
    else:
        model = load_model(model, args.load_model, device)
        visualize_and_save_predictions(model, test_loader, device, characters, args.weights_dir)
        evaluate_model(model, test_loader, device, characters)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Directory path
    parser.add_argument("--original_dir", type=str, default="./captcha/data/samples")
    parser.add_argument("--splited_dir", type=str, default="./captcha/data/dataset")
    parser.add_argument("--weights_dir", type=str, default="./weights")
    parser.add_argument("--load_model", type=str, default="./weights/resnet_18_89_16_10_49.pth")
    parser.add_argument("--eval", type=int, default="0")

    # Hyperparam
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_dataset_size", type=int, default=850)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()

    split_dataset(args.original_dir, args.splited_dir)
    
    run_training_loop(args)

if __name__ == "__main__":
    main()

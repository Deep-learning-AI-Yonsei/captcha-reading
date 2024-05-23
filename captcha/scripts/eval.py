import matplotlib.pyplot as plt
import os
import torch
# Function to load model weights
def load_model(model, weight_path, device):
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to visualize and save predictions
def visualize_and_save_predictions(model, data_loader, device, characters, save_dir='predictions'):
    num_images=5
    model.eval()  # Ensure the model is in evaluation mode
    images, predictions, actuals = [], [], []
    with torch.no_grad():
        for i, (image_batch, label_batch) in enumerate(data_loader):
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            _, predicted = outputs.max(2)
            label_indices = label_batch.argmax(dim=2)

            for j in range(image_batch.size(0)):
                if len(images) >= num_images:
                    break
                images.append(image_batch[j].cpu())
                predictions.append(predicted[j].cpu())
                actuals.append(label_indices[j].cpu())
            if len(images) >= num_images:
                break
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the images with predictions and actual labels
    for idx in range(num_images):
        img = images[idx].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        predicted_label = ''.join([characters[c] for c in predictions[idx]])
        actual_label = ''.join([characters[c] for c in actuals[idx]])
        
        plt.figure()
        plt.imshow(img)
        plt.title(f'Predicted: {predicted_label}\nActual: {actual_label}')
        plt.axis('off')
        save_path = os.path.join(save_dir, f'prediction_{idx + 1}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def evaluate_model(model, data_loader, device, characters):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_images = 0
    with torch.no_grad():  # Disable gradient computation
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(2)  # Get the index of the max log-probability for each character
            # Convert one-hot encoded labels to class indices
            labels_indices = labels.argmax(dim=2)
            # Compare predicted indices with true indices
            total_correct += (predicted == labels_indices).all(1).sum().item()
            total_images += labels.size(0)
            
            # Visualize the first batch
            # visualize_predictions(images, labels_indices, predicted, characters, args.weights_dir)
        accuracy = total_correct / total_images
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy
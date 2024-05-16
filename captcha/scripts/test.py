from captcha.scripts.eval import visualize_predictions

# Function to load model weights
def load_model(model, weight_path, device):
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to load model weights
def load_model(model, weight_path, device):
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to visualize predictions
def visualize_predictions(model, data_loader, device, characters, num_images=5):
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
    
    # Plot the images with predictions and actual labels
    plt.figure(figsize=(15, 10))
    for idx in range(num_images):
        img = images[idx].permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        predicted_label = ''.join([characters[c] for c in predictions[idx]])
        actual_label = ''.join([characters[c] for c in actuals[idx]])
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(img)
        plt.title(f'P: {predicted_label}\nA: {actual_label}')
        plt.axis('off')
    plt.show()


import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.SGDLogistic import SGDLogistic
from data import load_data
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def preprocess_data(train_features, train_labels, test_features, test_labels, validation_split=0.2):
    # Split the training data into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=validation_split, random_state=42
    )

    # Reshape if necessary
    if train_features.dim() > 2:
        batch_size = train_features.size(0)
        train_features = train_features.view(batch_size, -1)
        val_features = val_features.view(batch_size, -1)
    
    if test_features.dim() > 2:
        batch_size = test_features.size(0)
        test_features = test_features.view(batch_size, -1)
    
    train_features = train_features.float()
    val_features = val_features.float()
    test_features = test_features.float()
        
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(train_loader, val_loader, num_epochs=10):
    model = SGDLogistic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = model.get_optimizer()
    criterion = model.get_loss_function()
    scheduler = model.get_scheduler(optimizer)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0        
        for _, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()    

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracy = 100. * val_correct / val_total

        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%")
    
    # Save model
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(script_dir, 'saved_models', 'SGDLogistic.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    return train_loss, train_accuracy, val_loss, val_accuracy

def test_model(test_loader):
    model = SGDLogistic()
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, 'saved_models', 'SGDLogistic.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy, predictions

if __name__ == '__main__':
    torch.manual_seed(42)  # Set random seed for reproducibility
    # Load dataset
    train_features, train_labels, test_features, test_labels = load_data.fetch_mnist_data()
    train_loader, val_loader, test_loader = preprocess_data(train_features, train_labels, test_features, test_labels)
    train_loss, train_accuracy, val_loss, val_accuracy = train_model(train_loader, val_loader, num_epochs=10)
    accuracy, predictions = test_model(test_loader)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the first ten predictions compared to the actual values
    # Set support for Chinese fonts
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei
    plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of displaying minus signs  
    # Create a 2x5 subplot layout to display 10 images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Set subplot spacing
    # Iterate over the first 10 test samples
    for i in range(10):
        row, col = i // 5, i % 5  # Calculate row and column indices
        ax = axes[row, col]
        
        # Display the image in the corresponding subplot
        ax.imshow(test_features[i].reshape(28, 28), cmap=plt.cm.gray)
        ax.set_title(f"实际为：{test_labels[i].item()}，预测为：{predictions[i]}")
        ax.axis('off')  # Hide the axes
    plt.tight_layout()
    plt.show()
    

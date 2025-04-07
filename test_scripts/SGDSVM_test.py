import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.SGDSVM import SGDSVM
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
    model = SGDSVM()
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

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss_value = criterion(val_outputs, val_labels)

                val_running_loss += val_loss_value.item() * val_inputs.size(0)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100 * val_correct / val_total

        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f} %, '
              f'Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.2f} %')

    # 保存模型
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(script_dir, 'saved_models', 'SGDSVM.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    return train_loss, train_accuracy, val_loss, val_accuracy

def test_model(test_loader):
    model = SGDSVM()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(script_dir, 'saved_models', 'SGDSVM.pth')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    criterion = model.get_loss_function()

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = model(test_inputs)
            loss = criterion(test_outputs, test_labels)

            test_loss += loss.item() * test_inputs.size(0)
            _, predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (predicted == test_labels).sum().item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_accuracy = 100 * test_correct / test_total

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.2f}%')
    return avg_test_loss, avg_test_accuracy

if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels = load_data.fetch_mnist_data()
    train_loader, val_loader, test_loader = preprocess_data(train_features, train_labels, test_features, test_labels)
    train_loss, train_accuracy, val_loss, val_accuracy = train_model(train_loader, val_loader, num_epochs=10)

    # Plot training and validation loss
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Plot training and validation accuracy
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()
    
    test_model(test_loader)


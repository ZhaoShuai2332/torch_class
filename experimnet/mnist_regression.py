"""
Implementation of a Linear Classifier for MNIST Digit Recognition
Reference: LeCun et al., 'Gradient-based learning applied to document recognition' (1998)

The model implements a simple linear transformation: y = Wx + b
where:
    x ∈ ℝ^784: flattened input image
    W ∈ ℝ^(10×784): weight matrix
    b ∈ ℝ^10: bias vector
    y ∈ ℝ^10: output logits
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

# Ensure model persistence
os.makedirs('saved_models', exist_ok=True)

# Data Loading and Preprocessing
transform = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

class DigitClassifier(nn.Module):
    """
    Linear classifier for digit recognition.
    Architecture:
        Input (28×28) → Flatten (784) → Linear (784→10) → Softmax
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Probability distribution over 10 digit classes
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten: (batch_size, 784)
        logits = self.linear(x)
        return torch.softmax(logits, dim=1)
    
    def train_model(self, train_loader: torch.utils.data.DataLoader, num_epochs: int = 5):
        """
        Training procedure using SGD with momentum.
        Optimization:
            Loss: Cross Entropy
            Optimizer: SGD with momentum
            Learning rate scheduling: Step decay
        """
        # Initialize training components
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # Forward-backward pass
                logits = self.linear(images.view(-1, 784))
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            # Epoch-end logging
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
            scheduler.step()
            
        # Save trained model
        torch.save(self.state_dict(), 'saved_models/mnist_classifier.pth')
            
    def test(self, test_loader: torch.utils.data.DataLoader):
        """
        Evaluation procedure.
        Displays the prediction probability for true labels of first 10 test samples.
        """
        self.eval()
        correct = total = 0
        samples_shown = 0
        
        print("\nProbabilities for true labels (first 10 samples):")
        print("-" * 50)
        
        with torch.no_grad():
            for images, labels in test_loader:
                probs = self(images)
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Display probabilities for true labels
                if samples_shown < 10:
                    batch_samples = min(10 - samples_shown, labels.size(0))
                    for i in range(batch_samples):
                        true_label = labels[i].item()
                        prob = probs[i][true_label].item() * 100
                        print(f"Sample {samples_shown + 1}: Label = {true_label}, Probability = {prob:.2f}%")
                        samples_shown += 1
                
                if samples_shown >= 10:
                    break

            print("-" * 50)
            print(f"Test Accuracy: {100 * correct / total:.2f}%")

def main():
    """Main execution"""
    model = DigitClassifier()
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=100, shuffle=False)

    # Train and evaluate
    model.train_model(train_loader)
    model.test(test_loader)

if __name__ == "__main__":
    main()






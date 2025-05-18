"""
Implementation of Threshold-Shifted MNIST Classifier
Based on the base MNIST classifier with logit shifting technique.

The model implements the following transformations:
1. x ∈ ℝ^784 → Wx + b (base model logits)
2. u = logits - logit(t), where logit(t) = ln(t/(1-t))
3. Output binary mask where probability > t

Reference: Original MNIST classifier with threshold adjustment
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_regression import DigitClassifier

# Ensure model persistence
os.makedirs('saved_models', exist_ok=True)

class DigitClassifierWithT(nn.Module):
    """
    Extended MNIST classifier with threshold shifting.
    
    This model wraps the base DigitClassifier and adds a learnable threshold t,
    effectively shifting the decision boundary by logit(t).
    
    Args:
        pretrained_path: Path to pretrained base model weights
        t: Threshold value in (0,1) for decision boundary
    """
    def __init__(self, pretrained_path: str, t: float):
        super().__init__()
        # Load pretrained base model
        self.base = DigitClassifier()
        self.base.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        
        # Compute logit(t) = ln(t/(1-t))
        logit_t = torch.log(torch.tensor(t/(1-t), dtype=torch.float))
        self.register_buffer('logit_t', logit_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with threshold shifting.
        
        Args:
            x: Input tensor of shape [B, 784] or [B, 1, 28, 28]
            
        Returns:
            Binary mask tensor of shape [B, 10] where probability > t
        """
        # 1. Flatten and compute base logits
        x = x.view(x.size(0), -1)  # [B, 784]
        logits = self.base.linear(x)  # [B, 10]
        
        # 2. Apply threshold shift
        shifted_logits = logits - self.logit_t  # Broadcasting
        
        # 3. Convert to binary mask where prob > t
        probs = torch.softmax(shifted_logits, dim=1)
        return (probs > torch.sigmoid(self.logit_t)).to(torch.uint8)

    def save_model(self, save_path: str):
        """
        Save the model state dict to the specified path.
        
        Args:
            save_path: Path where to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'base_state_dict': self.base.state_dict(),
            'logit_t': self.logit_t
        }, save_path)
    
    @classmethod
    def load_model(cls, load_path: str):
        """
        Load a saved model from the specified path.
        
        Args:
            load_path: Path to the saved model
            
        Returns:
            Loaded DigitClassifierWithT model
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        t = torch.sigmoid(checkpoint['logit_t']).item()
        model = cls(None, t)  # Initialize with dummy path and correct threshold
        model.base.load_state_dict(checkpoint['base_state_dict'])
        return model

def test_model(model_path: str = 'saved_models/mnist_classifier.pth',
                threshold: float = 0.8,
                num_samples: int = 10):
    """
    Test the threshold-shifted model on MNIST test set.
    
    Args:
        model_path: Path to pretrained model
        threshold: Decision threshold
        num_samples: Number of samples to test
    """
    # Initialize model
    model = DigitClassifierWithT(model_path, threshold).eval()
    
    # Save the model
    save_path = 'saved_models/mnist_classifier_with_threshold.pth'
    model.save_model(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=num_samples, shuffle=False)
    
    # Test and display results
    print(f"\nTesting model with threshold t = {threshold}")
    print("="*50)
    
    with torch.no_grad():
        # Only get first batch (containing num_samples samples)
        images, labels = next(iter(test_loader))
        mask = model(images)
        
        print("\nSample-wise Results:")
        print("-"*50)
        for i in range(num_samples):
            print(f"\nSample {i + 1}:")
            print(f"True Label: {labels[i].item()}")
            print(f"Model Output: {mask[i][labels[i].item()]}")  # Binary predictions for each digit
            active_digits = torch.where(mask[i])[0].tolist()  # Get indices where mask is 1
            print(f"Active Predictions: {active_digits}")
        
        print("\nSummary Statistics:")
        print("-"*50)
        active_predictions = mask.sum(dim=1)
        print(f"Average active predictions per sample: {active_predictions.float().mean():.2f}")
        print(f"Total samples with multiple predictions: {(active_predictions > 1).sum().item()}")
        print(f"Total samples with no predictions: {(active_predictions == 0).sum().item()}")

if __name__ == "__main__":
    test_model()

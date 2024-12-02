import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionRecognitionCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3, clip_length=16):
        """
        3D CNN for video action recognition
        Args:
            num_classes (int): Number of action classes to predict
            input_channels (int): Number of input channels (typically 3 for RGB)
            clip_length (int): Number of frames in each video clip
        """
        super(ActionRecognitionCNN, self).__init__()
        
        # First 3D convolution block
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Second 3D convolution block
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Third 3D convolution block
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Fourth 3D convolution block
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Calculate size of flattened features
        self._to_linear = None
        self._calculate_flat_features(torch.zeros((1, input_channels, clip_length, 224, 224)))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def _calculate_flat_features(self, x):
        """Calculate the size of flattened features after convolution layers"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2] * x[0].shape[3]

    def forward(self, x):
        """
        Forward pass of the network
        Args:
            x: Input tensor of shape (batch_size, channels, frames, height, width)
        """
        # Apply 3D convolution blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Training loop for the action recognition model
    Args:
        model: The CNN model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
    """
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# Example usage
def main():
    # Hyperparameters
    num_classes = 10  # Number of action classes
    clip_length = 16  # Number of frames per clip
    batch_size = 16
    learning_rate = 0.001
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionCNN(num_classes=num_classes, clip_length=clip_length).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Assuming you have a train_loader set up
    # train_loader = create_data_loader(batch_size=batch_size)
    # train_model(model, train_loader, criterion, optimizer, device)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

# Model parameters
input_shape = (16, 64, 64, 3)  # 16 frames, 64x64 resolution, 3 color channels (RGB)

# Define the model
model = Sequential([
    # 1st 3D Convolutional Layer
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
    MaxPooling3D(pool_size=(2, 2, 2)),
    
    # 2nd 3D Convolutional Layer
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    
    # 3rd 3D Convolutional Layer
    Conv3D(128, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    
    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    
    # Output Layer (e.g., for 5 classes)
    Dense(5, activation='softmax')
])

# Model summary
model.summary()

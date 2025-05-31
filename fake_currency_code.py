import numpy as np
from glob import glob
!pip install --upgrade pip setuptools
import numpy as np
from glob import glob
# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = r"C:\Users\yamin\python programs\My_Project\Dataset\Train"
valid_path = r"C:\Users\yamin\python programs\My_Project\Dataset\Test"

# useful for getting number of output classes
folders = glob(r"C:\Users\yamin\python programs\My_Project\Dataset\Train/*")
import matplotlib.image as mpimg
from glob import glob
import os
import matplotlib.pyplot as plt
train_image_files = glob(os.path.join(train_path, '*/*.jpg'))

# Display some sample images from the training directory
num_images_to_display = 5
for img_path in train_image_files[:num_images_to_display]:
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_class_distribution(dataset_path):
    class_counts = Counter()
    for class_folder in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_folder_path):
            class_counts[class_folder] += len(os.listdir(class_folder_path))
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title('Class Distribution')
    
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

# Call this function for training and validation datasets
plot_class_distribution(train_path)
plot_class_distribution(valid_path)



#code for mobilenet
import os
#!pip install tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_set = datagen.flow_from_directory(train_path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), batch_size=32, class_mode='categorical')
valid_set = datagen.flow_from_directory(valid_path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), batch_size=32, class_mode='categorical')

# Load MobileNet model with pre-trained weights, excluding the top (fully connected) layers
base_model = MobileNet(weights='imagenet', include_top=False)

# Add custom top layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(len(folders), activation='softmax')(x)  # Assuming one node for each class

# Create the model
model = Model(inputs=base_model.input, outputs=preds)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
hist=model.fit(train_set, validation_data=valid_set, epochs=10, steps_per_epoch=len(train_set), validation_steps=len(valid_set))
import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# code for resnet50
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
num_classes=7
# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add global average pooling
x = Dense(128, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(num_classes, activation='softmax')(x)  # Output layer for multi-class classification

# Combine the base model with custom classification layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_set, epochs=5, validation_data=valid_set)

# Save the trained model
model.save("resnet_model.h5")

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
# Define paths
train_path = r"C:\Users\yamin\python programs\My_Project\Dataset\Train"
valid_path = r"C:\Users\yamin\python programs\My_Project\Dataset\Test"
IMAGE_SIZE = (227, 227)  # AlexNet input size
# Image preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_set = datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical",
)

valid_set = datagen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode="categorical",
)

# Define AlexNet model
def build_alexnet(input_shape, num_classes):
    model = Sequential([
        # 1st Convolutional Layer
        Conv2D(96, kernel_size=(11, 11), strides=4, activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # 2nd Convolutional Layer
        Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # 3rd, 4th, 5th Convolutional Layers
        Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"),
        Conv2D(384, kernel_size=(3, 3), activation="relu", padding="same"),
        Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # Flatten and Fully Connected Layers
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    return model
# Create the AlexNet model
num_classes = len(train_set.class_indices)  # Number of classes based on training data
alexnet_model = build_alexnet((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes)
# Compile the model
alexnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
# Train the model
history = alexnet_model.fit(
    train_set,
    validation_data=valid_set,
    epochs=10,
    steps_per_epoch=len(train_set),
    validation_steps=len(valid_set),
)
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Define paths
train_path = r"C:\Users\yamin\python programs\My_Project\Dataset\Train"
valid_path = r"C:\Users\yamin\python programs\My_Project\Dataset\Test"
batch_size = 32
num_epochs = 10
learning_rate = 0.0001

# Data Preprocessing and Augmentation
train_transforms = transforms.Compose([
    transforms.Resize((227, 227)),  # AlexNet input size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

valid_transforms = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_path, transform=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Load Pretrained AlexNet Model
alexnet = models.alexnet(pretrained=True)

# Modify the classifier for the custom dataset
num_classes = len(train_dataset.classes)
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=learning_rate)

# Training and Validation Loops
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Loss: {val_loss/len(valid_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

# Train the model
train_model(alexnet, train_loader, valid_loader, criterion, optimizer, num_epochs)

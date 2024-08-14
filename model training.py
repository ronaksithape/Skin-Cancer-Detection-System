import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load metadata
metadata = pd.read_csv('HAM10000_metadata.csv')
print(metadata['dx'].unique())

# Define image directory
image_dir = 'HAM10000 images'

# Merge metadata with image filenames
metadata['path'] = metadata['image_id'].apply(lambda x: os.path.join(image_dir, x + '.jpg'))

# Encode string labels into numeric values
le = LabelEncoder()
labels = le.fit_transform(metadata['dx'])

# Split data into training and testing sets
train_metadata, test_metadata, train_labels, test_labels = train_test_split(metadata, labels, test_size=0.2, random_state=42)

# Convert numeric labels to one-hot encoded vectors
num_classes = len(metadata['dx'].unique())
train_labels_one_hot = to_categorical(train_labels, num_classes=num_classes)
test_labels_one_hot = to_categorical(test_labels, num_classes=num_classes)

# Define image data generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Generate training and validation data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_metadata,
    x_col='path',
    y_col='dx',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_metadata,
    x_col='path',
    y_col='dx',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Construct the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("skin_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, verbose=1, mode='max', min_lr=1e-7)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    epochs=50,
    callbacks=[checkpoint, early_stopping, reduce_lr])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print('Test accuracy:', test_acc)

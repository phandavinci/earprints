import tensorflow as tf

# Define the model
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',  # Load pre-trained weights
    include_top=False,
    input_shape=(224, 224, 3),
    classes=4
)

# Get the summary of the model
base_model.summary()

# Count the total number of layers
total_layers = len(base_model.layers)
print("Total number of layers:", total_layers)

# Initialize variables to store layer types and their counts
layer_types = {}
layer_count = {}

# Loop through each layer and count the occurrences of each layer type
for layer in base_model.layers:
    layer_type = type(layer).__name__
    if layer_type in layer_types:
        layer_types[layer_type] += 1
    else:
        layer_types[layer_type] = 1

    # Count the occurrences of each specific layer
    layer_name = layer.name
    if layer_name in layer_count:
        layer_count[layer_name] += 1
    else:
        layer_count[layer_name] = 1

# Print the types of layers and their counts
print("\nTypes of layers and their counts:")
for layer_type, count in layer_types.items():
    print(f"{layer_type}: {count}")

# Print the count of each specific layer
print("\nCount of each specific layer:")
for layer_name, count in layer_count.items():
    print(f"{layer_name}: {count}")

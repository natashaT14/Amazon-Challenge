import pandas as pd
import re

# Load the training dataset
train_data = pd.read_csv('dataset/train.csv')

# Function to separate numeric values and units
def extract_numeric_and_unit(value):
    # Use regex to separate numeric part and unit part
    match = re.match(r"([0-9.]+)\s*(\w+)", str(value))
    if match:
        numeric_part = float(match.group(1))  # Numeric value
        unit_part = match.group(2)            # Unit
        return numeric_part, unit_part
    return None, None

# Apply the function to the 'entity_value' column
train_data['numeric_value'], train_data['unit'] = zip(*train_data['entity_value'].apply(extract_numeric_and_unit))

# Check if 'unit' column has consistent units and standardize them if needed
# For simplicity, here we're just listing unique units
print("Unique units in the dataset:", train_data['unit'].unique())

# Save the preprocessed data
train_data.to_csv('dataset/train_preprocessed.csv', index=False)

print("Preprocessing complete. Processed data saved to 'dataset/train_preprocessed.csv'.")

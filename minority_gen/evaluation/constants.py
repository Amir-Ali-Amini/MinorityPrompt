"""
Constants and label definitions for demographic evaluation.

These labels are based on the FairFace model's output categories.
Reference: https://github.com/joojs/fairface
"""

# Demographic attribute labels
RACE_LABELS_7 = [
    'White', 
    'Black', 
    'Latino_Hispanic', 
    'East Asian', 
    'Southeast Asian', 
    'Indian', 
    'Middle Eastern'
]

GENDER_LABELS = ['Male', 'Female']

AGE_LABELS = [
    '0-2', 
    '3-9', 
    '10-19', 
    '20-29', 
    '30-39', 
    '40-49', 
    '50-59', 
    '60-69', 
    '70+'
]

# Dictionary mapping attribute names to their possible values
ATTRIBUTES_DICT = {
    'race': RACE_LABELS_7,
    'gender': GENDER_LABELS,
    'age': AGE_LABELS
}

# Attribute combinations to analyze
ATTRIBUTE_COMBINATIONS = [
    ['race'],
    ['gender'],
    ['age'],
    ['race', 'gender'],
    ['race', 'age'],
    ['gender', 'age'],
    ['race', 'gender', 'age']
]

# FairFace model output indices
# The model outputs 18 values: 7 race + 2 gender + 9 age
RACE_OUTPUT_SLICE = slice(0, 7)
GENDER_OUTPUT_SLICE = slice(7, 9)
AGE_OUTPUT_SLICE = slice(9, 18)

# Default image processing parameters
DEFAULT_MAX_SIZE = 800
FACE_SIZE = 300
FACE_PADDING = 0.25

# FairFace model input size
FAIRFACE_INPUT_SIZE = 224

# ImageNet normalization (used by FairFace)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

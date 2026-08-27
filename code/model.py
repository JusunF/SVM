"""SVM backend selection and model construction."""

import numpy as np

from .config import TASKS

try:
    import cupy as cp
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.svm import LinearSVC

    # Check whether a CUDA GPU is actually available
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0

except Exception:
    GPU_AVAILABLE = False


# If GPU is not available, use scikit-learn instead
if not GPU_AVAILABLE:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

# Store which device is being used
if GPU_AVAILABLE:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# TF-IDF Vectorizer
def make_vectorizer(kwargs):
    """
    Create a TF-IDF vectorizer.

    The vectorizer will be:
    - cuML version if GPU is available
    - scikit-learn version otherwise
    """

    vectorizer = TfidfVectorizer(**kwargs)

    return vectorizer

# Convert Data to Numpy
def to_numpy(values):
    """
    Convert data into a NumPy array.

    If GPU is being used, move the data from GPU to CPU first.
    """

    if GPU_AVAILABLE:
        return cp.asnumpy(values)

    return np.asarray(values)

# Maximum class weight
WEIGHT_CAP = 10.0

# Calculate Class Weights
def class_weights(target):
    """
    Calculate an inverse-frequency weight for each class.

    Rare classes receive larger weights.
    Common classes receive smaller weights.

    The maximum weight is limited to WEIGHT_CAP.
    """

    # Count how many samples belong to each class
    target_array = np.asarray(target)
    counts = np.bincount(target_array)

    # Total number of samples
    number_of_samples = len(target_array)

    # Count how many classes are present
    number_of_classes = (counts > 0).sum()

    weights = {}

    # Calculate the weight for each class
    for class_id, count in enumerate(counts):

        # Ignore classes that do not appear in the dataset
        if count == 0:
            continue

        # Inverse-frequency class weight
        weight = number_of_samples / (
            number_of_classes * count
        )

        # Do not allow the weight to exceed 10
        if weight > WEIGHT_CAP:
            weight = WEIGHT_CAP

        weights[class_id] = weight

    return weights

# Train classification SVM models
def train_models(features, single_targets):
    """
    Train one multiclass Linear SVM for each task.

    Example tasks could be:
        sentiment
        emotion
        topic

    Each task gets its own SVM model.
    """

    models = {}

    # Train one model for each task
    for task in TASKS:

        # Get the labels for this task
        target = single_targets[task]

        # Calculate class weights
        weights = class_weights(target)

        # Create one weight for every training sample
        sample_weight = []

        for value in target:
            class_id = int(value)
            weight = weights[class_id]

            sample_weight.append(weight)

        # Convert sample weights to NumPy
        sample_weight = np.array(
            sample_weight,
            dtype=np.float32
        )

        if GPU_AVAILABLE:

            # Move target labels to GPU
            gpu_target = cp.asarray(target)

            # Move sample weights to GPU
            gpu_sample_weight = cp.asarray(sample_weight)

            # Train the GPU SVM
            model = LinearSVC()

            model.fit(
                features,
                gpu_target,
                sample_weight=gpu_sample_weight
            )

        else:

            # Train the CPU SVM
            model = LinearSVC(
                class_weight=weights
            )

            model.fit(
                features,
                target
            )

        # Save the trained model
        models[task] = model

    return models

# Prepare NER Features
def ner_features_to_matrix(ner_vectorizer, token_dicts):
    """
    Convert NER token dictionaries into a sparse matrix.

    DictVectorizer converts the token dictionaries into
    numerical features.

    The sparse matrix indices are converted to int32 because
    sklearn LinearSVC requires 32-bit indices in this case.
    """

    # Convert dictionaries into a sparse matrix
    matrix = ner_vectorizer.transform(token_dicts)

    # Convert matrix to CSR format
    matrix = matrix.tocsr()

    # Convert sparse matrix indices to int32
    matrix.indices = matrix.indices.astype(np.int32)

    # Convert sparse matrix pointers to int32
    matrix.indptr = matrix.indptr.astype(np.int32)

    return matrix

# Train NER SVM
def train_ner_tagger(X_dicts, y_tags, tag_labels):
    """
    Train a token-level BIO NER tagger.

    The NER pipeline is:

        Feature dictionaries
                ↓
        DictVectorizer
                ↓
        Sparse matrix
                ↓
        Class-weighted LinearSVC
    """
    # NER always uses scikit-learn
    # because cuML does not provide DictVectorizer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.svm import LinearSVC as CPULinearSVC

    # Convert NER tag names into numbers
    tag_to_id = {}

    for i, tag in enumerate(tag_labels):
        tag_to_id[tag] = i

    # Convert the string labels into integer labels
    target = []

    for tag in y_tags:
        target.append(tag_to_id[tag])

    target = np.array(target)

    # Create the NER feature vectorizer
    vectorizer = DictVectorizer(
        sparse=True
    )

    # Learn the available features
    vectorizer.fit(X_dicts)

    # Convert token dictionaries into a sparse matrix
    features = ner_features_to_matrix(
        vectorizer,
        X_dicts
    )

    # Calculate class weights
    weights = class_weights(target)

    # Train the NER SVM
    tagger = CPULinearSVC(
        class_weight=weights
    )

    tagger.fit(
        features,
        target
    )

    # Return both the vectorizer and trained model
    return vectorizer, tagger

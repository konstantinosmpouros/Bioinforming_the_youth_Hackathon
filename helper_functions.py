import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense



def transformer_block(x, num_heads, key_dim, block_num):
    """
    Creates a transformer block with multi-head attention, normalization, and dense layers.

    Parameters:
    x (tensor): Input tensor.
    num_heads (int): Number of attention heads.
    key_dim (int): Dimension of the key (and query and value) vectors.
    block_num (int): Block number for naming the layers.

    Returns:
    tensor: Output tensor after applying the transformer block operations.
    """
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim // num_heads, name=f'Attention_{block_num}')(x, x) 
    attention = Dropout(0.1, name=f'Dropout1_b{block_num}')(attention)

    attention = tf.cast(attention, dtype=tf.float16)
    x = tf.cast(x, dtype=tf.float16) 
    
    x = LayerNormalization(epsilon=1e-6, name=f'LayerNorm1_{block_num}')(x + attention)
    
    dense = Dense(key_dim, activation='relu', name=f'Dense1_b{block_num}')(x)
    dense = Dense(key_dim, activation='relu', name=f'Dense2_{block_num}')(dense)
    
    dense = Dropout(0.1, name=f'Dropout2_b{block_num}')(dense)
    
    x = LayerNormalization(epsilon=1e-6, name=f'LayerNorm2_{block_num}')(x + dense)

    return x


def class_report(labels_test, predictions):
    """
    Prints the classification report for the given true labels and predictions.

    Parameters:
    labels_test (array-like): True labels.
    predictions (array-like): Predicted labels.

    Returns:
    None
    """
    print(classification_report(y_true=labels_test, y_pred=predictions, target_names=['Negative', 'Positive'], zero_division=0))


def conf_matrix(labels_test, predictions, label):    
    """
    Plots and saves the confusion matrix for the given true labels and predictions.

    Parameters:
    labels_test (array-like): True labels.
    predictions (array-like): Predicted labels.
    label (str): Label to be used for saving the plot.

    Returns:
    None
    """
    cm = confusion_matrix(labels_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    plt.savefig('Figures/Modeling/' + label + '.png')
    plt.show()


def roc_auc_curve(labels_test, predictions, label):
    """
    Plots and saves the ROC-AUC curve for the given true labels and predictions.

    Parameters:
    labels_test (array-like): True labels.
    predictions (array-like): Predicted probabilities.
    label (str): Label to be used for saving the plot.

    Returns:
    None
    """
    # Calculate the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")

    plt.savefig('Figures/Modeling/' + label + '.png')
    plt.show()


def model_accuracy(labels_test, predictions):
    """
    Calculates and prints the accuracy, F1 score, and loss for the given true labels and predictions.

    Parameters:
    labels_test (array-like): True labels.
    predictions (array-like): Predicted probabilities.

    Returns:
    array: Predicted labels after thresholding at 0.5.
    """
    loss = tf.keras.losses.binary_crossentropy(labels_test, predictions).numpy().mean()
    predictions = tf.squeeze((predictions > 0.5).astype(int)).numpy()
    accuracy = accuracy_score(labels_test, predictions)
    f1 = f1_score(labels_test, predictions)
    macro_f1 = mean_f1(labels_test, predictions)

    print('Accuracy: ', accuracy)
    print('Mean F1 Score: ', macro_f1)
    print('F1 Score: ', f1)
    print('Loss: ', loss)
    
    return predictions


def mean_f1(y_true, y_pred):
    """
    Calculates the mean F1 score (macro F1 score) for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Mean F1 score.
    """
    return f1_score(y_true, y_pred, average='macro')


def f1(y_true, y_pred):
    """
    Calculates the F1 score for the given true and predicted labels.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: F1 score.
    """
    return f1_score(y_true, y_pred)


def plot_history(history, label):
    """
    Plots the training loss and accuracy from the model history.

    Parameters:
    history: keras.callbacks.History
        History object returned by the fit method of a Keras model.
    """
    # Extract loss and accuracy from the history object
    loss = history.history['loss']
    accuracy = history.history['accuracy']

    epochs = range(1, len(loss) + 1)

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, loss, label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, accuracy, label='Training Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Show the plots
    plt.tight_layout()
    plt.savefig('Figures/Modeling/' + label + '.png')
    plt.show()


def split_sequences(sequences, step):
    """
    Splits each sequence in a list of sequences into smaller subsequences of a specified step size.

    Parameters:
    sequences (list of str): The list of sequences to be split.
    step (int): The step size to split each sequence.

    Returns:
    list of list of str: A list of lists, where each inner list contains subsequences of the original sequence.
    """
    splited_sequences = []
    for seq in sequences:
        splited = [seq[i:i+step] for i in range(0, len(seq), step)]
        splited_sequences.append(splited)
    return splited_sequences


def amino_count(sequences):
    """
    Counts the frequency of each amino acid in a list of sequences.

    Parameters:
    sequences (list of str): The list of sequences to be analyzed.

    Returns:
    pandas.DataFrame: A DataFrame where each row corresponds to a sequence and each column represents the count of a specific amino acid.
    """
    amino_counts = []
    for seq in sequences:
        counts = Counter(seq)
        amino_counts.append(counts)
    
    amino_counts = pd.DataFrame(amino_counts).fillna(0).astype(int)
    return amino_counts


def balance_dataset(sequences_train, labels_train):
    """
    Balances a dataset by undersampling the majority class to match the number of samples in the minority class.

    Parameters:
    sequences_train (pandas.DataFrame): The feature DataFrame containing the training sequences.
    labels_train (pandas.DataFrame): The label DataFrame containing the training labels.

    Returns:
    tuple: A tuple containing the balanced feature DataFrame and the balanced label DataFrame.
    """
    features_1 = sequences_train[labels_train['Label'] == 1]
    labels_1 = labels_train[labels_train['Label'] == 1]

    features_0 = sequences_train[labels_train['Label'] == 0]
    labels_0 = labels_train[labels_train['Label'] == 0]

    features_0_sample = features_0.sample(n=len(features_1), random_state=42)
    labels_0_sample = labels_0.sample(n=len(features_1), random_state=42)

    balanced_features = pd.concat([features_1, features_0_sample])
    balanced_labels = pd.concat([labels_1, labels_0_sample])

    balanced_features = shuffle(balanced_features, random_state=42)
    balanced_labels = shuffle(balanced_labels, random_state=42)

    return balanced_features, balanced_labels



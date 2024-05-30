import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense



def transformer_block(x, num_heads, key_dim, block_num):
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
    print(classification_report(y_true=labels_test, y_pred=predictions, target_names=['Negative', 'Positive'], zero_division=0))


def conf_matrix(labels_test, predictions, label):    
    cm = confusion_matrix(labels_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    plt.savefig('Figures/Modeling/' + label + '.png')
    plt.show()


def roc_auc_curve(labels_test, predictions, label):
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
    loss = tf.keras.losses.binary_crossentropy(labels_test, predictions).numpy().mean()
    predictions = tf.squeeze((predictions > 0.5).astype(int)).numpy()
    accuracy = accuracy_score(labels_test, predictions)
    f1 = f1_score(labels_test, predictions)

    print('Accuracy: ', accuracy)
    print('F1 Score: ', f1)
    print('Loss: ', loss)
    
    return predictions


def plot_history(history):
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
    plt.show()


def split_sequences(sequences, step):
    splited_sequences = []
    for seq in sequences:
        splited = [seq[i:i+step] for i in range(0, len(seq), step)]
        splited_sequences.append(splited)
    return splited_sequences


import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt



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

    print('Accuracy: ', accuracy)
    print('Loss: ', loss)
    
    return predictions




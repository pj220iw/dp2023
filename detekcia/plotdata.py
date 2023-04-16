"""
#################################
# plot functions for visualization
#################################
"""
#########################################################
# import libraries

import random
import pickle
import itertools
import numpy as np
import keras 
from matplotlib import pyplot as plt
from skimage.io import imshow
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.utils.vis_utils import plot_model


#########################################################
# Function definition

def plot_training(result, type_model, layers_len):
    fig = plt.figure()
    epochs = len(result.history['accuracy'])
    plt.title("Chybovosť", fontsize=14, fontweight='bold')
    plt.xlabel("Epizóda", fontsize=14, fontweight="bold")
    plt.ylabel("Chybovosť", fontsize=14, fontweight="bold")
    #plt.plot(np.arange(1, epochs), result.history['loss'], label='loss')
    #plt.plot(np.arange(1, epochs), result.history['val_loss'], label='validation_loss')
    plt.plot(np.arange(1, epochs+1), result.history['loss'], label='Chybovosť', linewidth=2.5, linestyle='-', marker='o',
               markersize='10', color='red')
    plt.plot(np.arange(1, epochs+1), result.history['val_loss'], label='Validačná chybovosť', linewidth=2.5, marker='x',
               linestyle='--', markersize='10', color='blue')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=15)
   
    #plt.subplots_adjust(hspace=0.3)
    fig2 = plt.figure()
    plt.title("Presnosť", fontsize=14, fontweight="bold")
    plt.xlabel("Epizóda", fontsize=14, fontweight="bold")
    plt.ylabel("Presnosť", fontsize=14, fontweight="bold")
    #plt.plot(np.arange(1, epochs), result.history['accuracy'], label='accuracy')
    #plt.plot(np.arange(1, epochs), result.history['val_accuracy'], label='validation_accuracy')
    
    plt.plot(np.arange(1, epochs+1), result.history['accuracy'], label='Presnosť', linewidth=2.5, linestyle='-',
              marker='o', markersize='3', color='red')
    plt.plot(np.arange(1, epochs+1), result.history['val_accuracy'], label='Validačná presnosť', linewidth=2.5,
               linestyle='--', marker='x', markersize='3', color='blue')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tick_params(axis='both', which='major', labelsize=15)
    
    file_figobj = 'Vystup/%s_%d_EPOCH_%d_layers_opt.fig.pickle' % (type_model, epochs, layers_len)
    file_pdf = 'Vystup/vystup_chybovost_%s_%d_Epoch_%d_layers.pdf' % (type_model, epochs, layers_len)
    file_pdf2 = 'Vystup/vystup_presnost_%s_%d_Epoch_%d_layers.pdf' % (type_model, epochs, layers_len)
    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')
    fig2.savefig(file_pdf2, bbox_inches='tight')

def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall', 'bin_accuracy']
    epochs = len(history.history['accuracy'])
    (fig, ax) = plt.subplots(1, 5, figsize=(20, 5))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        ax[n].plot(history.epoch, history.history[metric], linewidth=2.5, linestyle='-', marker='o', markersize='10',
                   color='blue', label='Train')
        ax[n].plot(history.epoch, history.history['val_'+metric], linewidth=2.5, linestyle='--', marker='x',
                   markersize='10', color='blue', label='Val')
        ax[n].grid(True)
        # plt.xlabel('Epoch')
        # plt.ylabel(name)
        ax[n].set_xlabel("Epoch", fontsize=14, fontweight="bold")
        ax[n].set_ylabel(name, fontsize=14, fontweight="bold")
        ax[n].legend(prop={'size': 14, 'weight': 'bold'}, loc='best')
        ax[n].tick_params(axis='both', which='major', labelsize=15)

    file_figobj = 'Output/FigureObject/Metric_%d_EPOCH.fig.pickle' % epochs
    file_pdf = 'Output/Figures/Metric_%d_EPOCH.pdf' % epochs

    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    fig_conf = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=12, fontweight='bold')
    plt.xlabel('Predicted label', size=12, fontweight='bold')
    # file_pdf = 'Output/Figures/confusion_matrix.pdf'
    file_figobj = 'Output/FigureObject/confusion_matrix.fig.pickle'
    pickle.dump(fig_conf, open(file_figobj, 'wb'))


def plot_segmentation_test(xval, yval, ypred, num_samples):
    fig = plt.figure(figsize=(16, 13))
    for i in range(0, num_samples):
        plt.subplot(3, num_samples, (0 * num_samples) + i + 1)
        ix_val = random.randint(0, len(ypred) - 1)
        title = str(i+1)
        plt.title(title)
        imshow(xval[ix_val])
        plt.axis('off')

        plt.subplot(3, num_samples, (1 * num_samples) + i + 1)
        plt.imshow(np.squeeze(yval[ix_val]))
        plt.title('gTruth')
        plt.axis('off')

        plt.subplot(3, num_samples, (2 * num_samples) + i + 1)
        plt.imshow(np.squeeze(ypred[ix_val]))
        plt.title('Mask')
        plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    file_figobj = 'Vystup/segmentation/segmentation_test.fig.pickle' % ()
    file_pdf = 'Vystup/segmentation/segmentation_test.pdf' % ()
    pickle.dump(fig, open(file_figobj, 'wb'))
    fig.savefig(file_pdf, bbox_inches='tight')

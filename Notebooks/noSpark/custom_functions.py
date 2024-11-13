#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# create a plot of the invariant mass distribution
def plotSignalvsBg2(x, y):
    
    variable = x.name
    
    def isSignal(x, y):
        if (y>=0.5):
            return x
        else: 
            return -1.
    
    def isBackground(x, y):
        if (y<0.5):
            return x
        else: 
            return -1.
    
    isSignalNP = np.vectorize(isSignal)
    isBackgroundNP = np.vectorize(isBackground)

    x_signal = isSignalNP(x, y)
    x_background = isBackgroundNP(x, y)

    f, ax = plt.subplots()
    plt.hist(x_signal, bins = 100, range=[0, 3.5], alpha=0.5, label='signal') 
    plt.hist(x_background, bins = 100, range=[0, 3.5], alpha=0.5, label='background') 
    
    plt.title("histogram") 
    ax.set_xlabel(variable)
    ax.set_ylabel('counts')
    ax.legend()
    ax.set_title("Distribution of "+variable)
    plt.show()
    f.savefig("SignalvsBackground.pdf", bbox_inches='tight')

    return

# create a plot of the invariant mass distribution with prediction
def plotSignalvsBgWithPrediction2(x_test, y_test, y_pred):
    
    variable = x_test.name
    
    def isSignal(x, y):
        if (y>=0.5):
            return x
        else: 
            return -1.
    
    def isBackground(x, y):
        if (y<0.5):
            return x
        else: 
            return -1.
    
    isSignalNP = np.vectorize(isSignal)
    isBackgroundNP = np.vectorize(isBackground)

    x_signal = isSignalNP(x_test, y_test)
    x_background = isBackgroundNP(x_test, y_test)
    x_signal_pred = isSignalNP(x_test, y_pred)
    x_background_pred = isBackgroundNP(x_test, y_pred)

    f, ax = plt.subplots()
    plt.hist(x_signal, bins = 100, range=[0, 3.5], alpha=0.5, label='signal') 
    plt.hist(x_background, bins = 100, range=[0, 3.5], alpha=0.5, label='background') 
    plt.hist(x_signal_pred, bins = 100, range=[0, 3.5], label='predicted signal', histtype='step',
        linestyle='--', color='green', linewidth=2) 
    plt.hist(x_background_pred, bins = 100, range=[0, 3.5], label='predicted background', histtype='step',
        linestyle='--', color='red', linewidth=2) 
    
    plt.title("histogram") 
    ax.set_xlabel(variable)
    ax.set_ylabel('counts')
    ax.legend()
    ax.set_title("Distribution of "+variable)
    plt.show()
    f.savefig("SignalvsBackgroundPred.pdf", bbox_inches='tight')

    return

def plotCorrelation(x): #correlation matrix

    # Calculate the correlation matrix
    corr_matrix = x.corr()

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12,8))
    ax = sns.heatmap(corr_matrix, cmap="YlGnBu")
    ax.set_title("Correlation Matrix")
    plt.tight_layout()

    # Display the plot
    plt.show()
    fig.savefig("CorrMatrix.pdf", bbox_inches='tight')
    return


# Plot variable (loss, acc) vs. epoch
def plotVsEpoch(history, variable):

    #get_ipython().run_line_magic('matplotlib', 'notebook')
    
    plt.figure()
    plt.plot(history.history[variable], label='train')
    plt.plot(history.history['val_'+variable], label='validation')
    plt.ylabel(variable)
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.show()
    
    return

# Draw roc curve for Keras
def drawROC2(y_scores, y_test):

    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = auc(fpr, tpr)

    f = plt.figure()
    plt.plot([0,1], [0,1], '--', color='orange')
    plt.plot(fpr, tpr, label='auc = {:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.grid()    
    plt.show()
    f.savefig("AUC.pdf", bbox_inches='tight')
    
    return

# Draw roc curve
def drawROC(result):

    result_pd = result.select(['label', 'prediction', 'probability']).toPandas()

    result_pd['probability'] = result_pd['probability'].map(lambda x: list(x))
    result_pd['encoded_label'] = result_pd['label'].map(lambda x: np.eye(2)[int(x)])

    y_pred = np.array(result_pd['probability'].tolist())
    y_true = np.array(result_pd['encoded_label'].tolist())    
    
    drawROC2(y_true[:,0], y_pred[:,0])
    
    return

# Draw feature importance (only GBT models)
def drawFeatures(x, gbt):
    
    # Get feature importances
    feature_importances = gbt.feature_importances_

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': x.columns,
        'Importance': feature_importances
    })

    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("Feature Importance Ranking:")
    print(feature_importance_df)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    feature_importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.show()
    
    return


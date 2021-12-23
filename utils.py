'''
Custom helper functions:
    to visualize results from machine learning binary classification models.
'''

# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

def optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification
    ----------
    target: true labels
    predicted: positive probability predicted by the model.
    i.e. model.prdict_proba(X_test)[:, 1], NOT 0/1 prediction array

    Returns
    -------     
    cut-off value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    
    return round(list(roc_t['threshold'])[0], 2)


def plot_confusion_matrix(y_true, y_pred):
    # confusion matrix: for binary classification results
    conf_matrix = confusion_matrix(y_true, y_pred)
    data = conf_matrix.transpose()  
    
    _, ax = plt.subplots()
    ax.matshow(data, cmap="Blues")
    # printing exact numbers
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{}'.format(z), ha='center', va='center')
    # axis formatting 
    plt.xticks([])
    plt.yticks([])
    plt.title("True label\n 0  {}     1\n".format(" "*18), fontsize=14)
    plt.ylabel("Predicted label\n 1   {}     0".format(" "*18), fontsize=14)

    
def plot_confusion_matrix2(cm, classes, normalize=False):
    """
        Plots confusion matrix for multi-class classification results
        Input: 
            confusion matrix, list of classes, 
            classes: list of unique classes (pass the str class names)
            If normalize = True: plots the normalized confusion matrix, 
                           Otherwise absolute numbers
        Output:
            Plots and show the confusion matrix  
    """
    cm = np.array(cm)
    n_class = len(classes)
    if normalize:
        np.set_printoptions(precision=3)       
        ncm = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                ncm[i, j] = cm[i, j]/sum(cm[i, :])
        cm = ncm
        
    vmin, vmax = min(cm.flatten()), max(cm.flatten())   
    plt.figure(figsize=(8, 8))
    img = plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=vmin, vmax=vmax)
    plt.title("Confusion matrix")
    plt.colorbar(img, shrink=0.7)
    
    ticks = np.arange(n_class)
    plt.xticks(ticks, classes, rotation=0, fontsize=14)
    plt.yticks(ticks, classes, rotation=90, fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 20.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=14, 
                 color="white" if cm[i, j] > thresh else "black"
                 )
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
def draw_roc_curve(y_true, y_proba):
    '''
    y_true: 0/1 true labels for test set
    y_proba: model.predict_proba[:, 1] or probabilities of predictions
    
    Return:
        ROC curve with appropriate labels and legend 
    
    '''
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    _, ax = plt.subplots()
    
    ax.plot(fpr, tpr, color='r');
    ax.plot([0, 1], [0, 1], color='y', linestyle='--')
    ax.fill_between(fpr, tpr, label=f"AUC: {round(roc_auc_score(y_true, y_proba), 3)}")
    ax.set_aspect(0.90)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.02, 1.02);
    ax.set_ylim(-0.02, 1.02);
    plt.legend()
    plt.show()
    
    
def summerize_results(y_true, y_proba):
    '''
     Takes the true labels and the predicted probabilities
     and prints the important performance results.
    '''
    print("\n=========================")
    print("        RESULTS")
    print("=========================")
    thd = optimal_cutoff(y_true, y_proba)
    y_pred = (y_proba > thd).astype(int)
    
    print("Accuracy: ", accuracy_score(y_true, y_pred).round(2))
    print("AUC:  \t", roc_auc_score(y_true, y_proba).round(2))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[1, 0]), 2)
    specificity = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[0, 1]), 2)
    
    ppv = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[0, 1]), 2)
    npv = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[1, 0]), 2)
    
    print("-------------------------")
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    
    print("-------------------------")
    
    print("positive predictive value: ", ppv)
    print("negative predictive value: ", npv)
    
    print("-------------------------")
    print("precision: ", precision_score(y_true, y_pred).round(2))
    print("recall: ", recall_score(y_true, y_pred).round(2))
    print("weighted precision: ", precision_score(y_true, y_pred, average="weighted").round(2))
    print("weighted recall: ", recall_score(y_true, y_pred, average="weighted").round(2))
    
# Radar plot: separate two classes based on few important variables
def radar_plot_class(df):
    '''
    In binary classification problem:
        df: feature dataframe
        edges: select few most important columns/features --> plot radar chart to show separation between the two classes
    '''
    
    target = "target"
    if target not in df.columns:
        raise ValueError("If prediction target variable is named different, name it target!")
    
    corr = df.corr()[target].sort_values(ascending=False)

    edges = list(corr.index)[:7]
    if target in edges:
        edges.remove(target)
    
    # stadardization of features: for scaled radii as standard deviation of features
    df_scaled = df[edges]
    df_scaled = StandardScaler().fit_transform(df_scaled)

    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = edges
    df_scaled[target] = list(df.target.values)

    # radius of the chart
    radii_0 = []
    radii_1 = []
    for edge in edges:
        
        value1 = df_scaled[df_scaled.target == 1][edge].mean()
        value0 = df_scaled[df_scaled.target == 0][edge].mean()
        
        radii_1.append(value1)
        radii_0.append(value0)
            
    edge_labels = [i.upper() for i in edges]

    # plotting
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radii_0,
        theta=edge_labels,
        fill='toself',
        name='Negative'
    ))
    fig.add_trace(go.Scatterpolar(
        r=radii_1,
        theta=edge_labels,
        fill='toself',
        name='Positive'
    ))
    fig.update_layout(
    polar=dict(
            radialaxis=dict(
            visible=True,
            range=[-1, 0.5]
        )),
    showlegend=True
    )
    fig.show()

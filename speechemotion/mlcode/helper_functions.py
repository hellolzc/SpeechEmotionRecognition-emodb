import pandas as pd #数据分析
import numpy as np #科学计算

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
# from sklearn.utils.multiclass import unique_labels

# Setup helper Functions
def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()


def plot_distribution_simple(df, var, target):
    # TODO: Finish it
    values = df.loc[:, target].value_counts()
    for index in values.index:
        df.loc[(df.loc[:, target] == index), var].plot(kind='kde')   
    plt.xlabel(var)
    plt.ylabel(u"Density") 
    plt.title("%s distribution"%var)
    plt.legend(values.index,loc='best')


def plot_box(df, var, target, title=None, value_index=None, ylabel=None):
    '''画箱形图'''
    values = df.loc[:, target].value_counts()
    merged_list = []
    if value_index is None:
        value_index = values.index
    for index in value_index:
        mask = df.loc[:, target] == index
        merged_list.append(df.loc[mask, var])
        print('[INFO] %s group mean %f, std %f' % (index, df.loc[mask, var].mean(), df.loc[mask, var].std()))
    plt.boxplot(merged_list, labels=value_index)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(var)            # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    if title is None:
        title = "%s distribution"%var
    plt.title(title)


def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 16 , 14 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = False#, 
        #annot_kws = { 'fontsize' : 9 }
    )

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending=True )
    print(imp)
    imp[ -10: ].plot( kind='barh' )
    print ('DecisionTreeClassifier.score:', model.score( X , y ))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_confusion_matrix(cm, classes=None,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # if normalize:
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    if normalize:
        cm_1 = cm_norm
        cm_2 = cm
    else:
        cm_1 = cm
        cm_2 = cm_norm
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm_1, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm_1.shape[1]),
           yticks=np.arange(cm_1.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '%.3f\n(%d)' if normalize else '%d\n(%.2f)'
    thresh = cm_1.max() / 2.
    for i in range(cm_1.shape[0]):
        for j in range(cm_1.shape[1]):
            info_str = fmt % (cm_1[i, j], cm_2[i, j])
            # format(cm[i, j], fmt)
            ax.text(j, i, info_str, ha="center", va="center",
                    color="white" if cm_1[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def show_confusion_matrix(confmat, save_pic_path=None):
    '''my simple version of '''
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(confmat, cmap=plt.cm.Reds, alpha=0.4)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.tight_layout()
    if save_pic_path:
        plt.savefig(save_pic_path, dpi=300)
    plt.show()

# Evaluate
def accuracy(Y_true, predictions):
    """ calculate accuracy """
    error = np.squeeze(predictions) - np.squeeze(Y_true)
    #print(error)
    acc = sum(error == 0) / len(error)
    return acc

def precision_recall_f1score(Y_true, predictions, pos_label=1, class_num=2):
    """ calculate precision_recall_f1score
    二分类，建议使用F1分数作为最终评估标准, 这个只会返回关于pos_label类的score,
    三(多)分类，建议使用UAR作为最终评估标准,返回UAPrecision, UARecall, UAF1score
    """
    # error = predictions - Y_true
    # TPN = sum((predictions == pos_label) & (Y_true == pos_label))
    # FNN = sum((predictions != pos_label) & (Y_true == pos_label))
    # FPN = sum((predictions == pos_label) & (Y_true != pos_label))
    # precision = TPN / (TPN+FNN)
    # recall = TPN / (TPN+FPN)
    # f1score = 2*precision*recall/(precision+recall)
    predictions = np.squeeze(predictions)
    Y_true = np.squeeze(Y_true)
    if class_num == 2:
        precision, recall, f1score, _ = precision_recall_fscore_support(Y_true, 
                                            predictions, pos_label=pos_label, average='binary')
        return precision, recall, f1score
    else:
        UAPrecision, UARecall, UAF1score, _ = precision_recall_fscore_support(Y_true,
                                                predictions, average='macro')
        return UAPrecision, UARecall, UAF1score

from api import app
import os
import pandas as pd
import json
from flask import jsonify
from flask import render_template
from sklearn import preprocessing
from sklearn import decomposition
from flask import url_for
from sklearn import metrics
import matplotlib.pyplot as plt; plt.rcdefaults()
import sklearn.cross_validation as cv
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def get_abs_path():
    return os.path.abspath(os.path.dirname(__file__))

def get_data():
    f_name = os.path.join(get_abs_path(), 'data', 'breast-cancer-wisconsin.csv')
    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'adhesion', 'cell_size', 'bare_nuclei',
               'bland_chromatin', 'normal_nuclei', 'mitosis', 'class']
    df = pd.read_csv(f_name, sep =',', header = None, names = columns,na_values = '?' )
    return df.dropna()


@app.route('/')
def index():
    '''
    This function will:
    - import the breast cancer data,
    - then scale it,
    _ then perform PCA on it and
    - on the components of the PCA run K-means clustering and
    - then plot the clusters
    '''
    # get data
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns =='class'].as_matrix()

    # Scaling
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)

    # Perform PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var  = pcomp.explained_variance_ratio_.sum()

    # Perform clustering (K-means)
    model = KMeans(init = 'k-means++', n_clusters=2)
    model.fit(components)

    # Plot
    fig = plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=model.labels_)
    centers = plt.plot(
        [model.cluster_centers_[0,0], model.cluster_centers_[1,0]],
        [model.cluster_centers_[0, 1], model.cluster_centers_[1, 1]],
        'kx', c = 'Green')

    #Increase size of center points
    plt.setp(centers, ms = 11.0)
    plt.setp(centers, mew = 1.8)
    axes = plt.gca()
    axes.set_xlim([-7.5, 3])
    axes.set_ylim([-2, 5])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering of PCS({:.2f}% Var. Explained)'.format(var*100))

    #Save Fig
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cluster.png')
    fig.savefig(fig_path)
    return render_template('index.html', fig = url_for('static', filename = 'tmp/cluster.png'))


@app.route('/d3')
def d3():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()

    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum()
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Generate CSV
    cluster_data = pd.DataFrame(
        {'pc1': components[:,0],
         'pc2': components[:,1],
         'labels': model.labels_}
    )

    csv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'kmeans.csv')
    cluster_data.to_csv(csv_path)
    return render_template('d3.html', data_file = url_for('static', filename = 'tmp/kmeans.csv', ))


@app.route('/head')
def head():
    df = get_data().head()
    data = json.loads(df.to_json())
    return jsonify(data)

@app.route('/v1/prediction_confusion_matrix')
def prediction_confusion_matrix():
    """
        This function will find the number of important features,
        while fitting the data into the model and classifying it.
        It will predict the test data to calculate y_pred.
        It will also plot a bar graph of the feature importance.
        The model will then be validated using the standard metrics.
        """
    # get data
    df = get_data()

    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y.ravel())

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, train_size=0.8, random_state=0)

    rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1, bootstrap=True, warm_start=True, random_state=0)
    rfc.fit(X_train, y_train)
    y_true = y_test
    y_pred = rfc.predict(X_test)

    # use confusion_matrix function from matrix module to calculate the specificity of the data
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # store the output of confusion matrix in appropriate variable
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    TP = confusion_matrix[1, 1]
    FN = confusion_matrix[1, 0]

    cf_data = pd.DataFrame(
        {
            'random forrest': [FP, TP, FN, TN]
        },  index=['fp', 'tp', 'fn', 'tn'])

    data = json.loads(cf_data.to_json())
    return jsonify(data)

@app.route('/prediction')
def prediction():
    """
        This function will find the number of important features,
        while fitting the data into the model and classifying it.
        It will predict the test data to calculate y_pred.
        It will also plot a bar graph of the feature importance.
        The model will then be validated using the standard metrics.
        """
    # get data
    df = get_data()

    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns =='class'].as_matrix()
    # conversion to binary variable
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y.ravel())

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, train_size=0.8, random_state=0)

    rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1, bootstrap=True, warm_start=True, random_state=0)
    rfc.fit(X_train, y_train)
    y_true = y_test
    y_pred = rfc.predict(X_test)
    y_score = rfc.predict_proba(X_test)

    # use confustion_matix function from matrix module to calculate the specificity of the data
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # store the output of confusion matrix in appropriate variable
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    TP = confusion_matrix[1, 1]
    FN = confusion_matrix[1, 0]
    fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, lw=1, color="#0000ff")
    # Plot a straight line from the point (0,0) to (1,1)
    plt.plot([0, 1], [0, 1], 'k--')
    # Set the X axis to the range [-0.05, 1]
    plt.xlim([-0.05, 1.0])
    # Set the Y axis to the range [0, 1.05]
    plt.ylim([0.0, 1.05])
    # Label X axis as False Positive Rate
    plt.xlabel('False Positive Rate')
    # Label Y axis as True Positive Rate
    plt.ylabel('True Positive Rate')
    # Provide a title to the curve
    plt.title('ROC for Random Forrest Classifier-Breast Cancer')
    # provide a legend to the plot and position it to the lower right position
    plt.legend(loc="lower right")

    #Save Fig
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'ROC.png')
    plt.savefig(fig_path)
    return render_template('prediction.html', fig = url_for('static', filename = 'tmp/ROC.png'))
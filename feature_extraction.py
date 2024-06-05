import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read & manipulate data
import pandas as pd
import numpy as np
import tensorflow as tf

# visualisations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')


# misc
import random as rn
'''
Never use accuracy score as a metric with imbalanced datasets - it will be usually very high and misleading (you can use AUC - ROC, Recall, F1 score instead).
Consider to take advantage of undersampling or oversampling techniques.
Use stratified splitting during train-test split.
Be extra careful when dealing with outliers (you can delete meaningull information).
'''


# load the dataset
df = pd.read_csv('creditcard.csv')

# manual parameters
RANDOM_SEED = 42
TRAINING_SAMPLE = 200000
VALIDATE_SIZE = 0.2

# setting random seeds for libraries to ensure reproducibility
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
#tf.set_random_seed(RANDOM_SEED)


"""
t-SNE 是一种用于可视化复杂数据集的降维技术。它将高维数据中的集群映射到二维或三维平面，以便我们了解区分类别的难易程度。它通过尝试保持低维数据点之间的距离与这些数据点在高维中是邻居的概率成比例来实现这一点。

"""

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def tsne_scatter(features, labels, dimensions=2, save_as='graph.png'):
    if dimensions not in (2, 3):
        raise ValueError(
            'tsne_scatter can only plot in 2d or 3d (What are you? An alien that can visualise >3d?). Make sure the "dimensions" argument is in (2, 3)')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=RANDOM_SEED).fit_transform(features)

    # initialising the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Fraud'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Clean'
    )

    # storing it to be displayed later
    plt.legend(loc='best')
    plt.savefig(save_as)
    plt.show

# manual parameter
RATIO_TO_FRAUD = 15

df.columns = map(str.lower, df.columns)
df.rename(columns={'class': 'label'}, inplace=True)


# dropping redundant columns
df = df.drop(['time', 'amount'], axis=1)

# splitting by class
fraud = df[df.label == 1]
clean = df[df.label == 0]

# undersample clean transactions
clean_undersampled = clean.sample(
    int(len(fraud) * RATIO_TO_FRAUD),
    random_state=RANDOM_SEED
)

# concatenate with fraud transactions into a single dataframe
visualisation_initial = pd.concat([fraud, clean_undersampled])
column_names = list(visualisation_initial.drop('label', axis=1).columns)

# isolate features from labels
features, labels = visualisation_initial.drop('label', axis=1).values, \
                   visualisation_initial.label.values



#tsne_scatter(features, labels, dimensions=2, save_as='tsne_initial_2d.png')




def feature_dist(df0,df1,label0,label1,features):
    plt.figure()
    fig,ax=plt.subplots(6,5,figsize=(30,45))
    i=0
    for ft in features:
        i+=1
        plt.subplot(6,5,i)
        # plt.figure()
        sns.distplot(df0[ft], hist=False,label=label0)
        sns.distplot(df1[ft], hist=False,label=label1)
        plt.xlabel(ft, fontsize=11)
        #locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=9)
        plt.tick_params(axis='y', labelsize=9)
    plt.show()

#features = df.columns.values

#feature_dist(clean,fraud ,'Normal', 'Busted', features)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px


def perform_pca_and_plot(df, drop_columns, class_column, n_components_3d=3, n_components_2d=2):
    """
    Perform PCA on the given DataFrame and plot 3D and 2D scatter plots.

    Parameters:
    - n_components_3d: int, number of components for 3D PCA (default is 3)
    - n_components_2d: int, number of components for 2D PCA (default is 2)
    """
    # Drop unnecessary columns
    df_reduced = df.drop(drop_columns, axis=1)

    # Perform PCA for 3D
    pca_3d = PCA(n_components=n_components_3d)
    X3D = pca_3d.fit_transform(df_reduced)

    # Perform PCA for 2D
    pca_2d = PCA(n_components=n_components_2d)
    X2D = pca_2d.fit_transform(df_reduced)

    # Plot 3D scatter plot using Plotly
    fig_3d = px.scatter_3d(
        x=X3D[:, 0], y=X3D[:, 1], z=X3D[:, 2],
        color=df[class_column],
        title='3D PCA'
    )
    fig_3d.show()

    # Plot 2D scatter plot using Seaborn and Matplotlib
    plt.figure(figsize=(20, 10))
    plt.title('2D PCA', size=40, y=1.05, color='#1e90c9')
    sns.scatterplot(x=X2D[:, 0], y=X2D[:, 1], hue=df[class_column], alpha=0.8)
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo


def perform_pca_and_plot2(df, drop_columns, class_column, n_components_3d=3, n_components_2d=2):
    """
    Perform PCA on the given DataFrame and plot 3D and 2D scatter plots.

    Parameters:
    - df: pandas DataFrame, the input data
    - drop_columns: list, columns to drop before performing PCA
    - class_column: str, the name of the column to use for coloring the plots
    - n_components_3d: int, number of components for 3D PCA (default is 3)
    - n_components_2d: int, number of components for 2D PCA (default is 2)
    """
    # Drop unnecessary columns
    df_reduced = df.drop(drop_columns, axis=1)

    # Perform PCA for 3D
    pca_3d = PCA(n_components=n_components_3d)
    X3D = pca_3d.fit_transform(df_reduced)

    # Perform PCA for 2D
    pca_2d = PCA(n_components=n_components_2d)
    X2D = pca_2d.fit_transform(df_reduced)

    # Plot 3D scatter plot using Plotly (offline mode)
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=X3D[:, 0],
        y=X3D[:, 1],
        z=X3D[:, 2],
        mode='markers',
        marker=dict(size=5, color=df[class_column], colorscale='Viridis', opacity=0.8),
        text=df[class_column]
    )])
    fig_3d.update_layout(title='3D PCA', scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))

    # Save and show Plotly figure offline
    pyo.plot(fig_3d, filename='3d_pca.html')

    # Optionally open the HTML file automatically
    import webbrowser
    webbrowser.open('3d_pca.html')

    # Plot 2D scatter plot using Seaborn and Matplotlib
    plt.figure(figsize=(20, 10))
    plt.title('2D PCA', size=40, y=1.05, color='#1e90c9')
    sns.scatterplot(x=X2D[:, 0], y=X2D[:, 1], hue=df[class_column], alpha=0.8)
    plt.show()


# Example usage:
perform_pca_and_plot2(df, drop_columns=[ 'label'], class_column='label')


def corr(data):
    plt.figure(figsize = (11,11))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask = mask, robust = True, center = 0,square = True, cmap="crest",linewidths = .6)
    plt.title('Correlation Table')
    plt.show()


#corr(df)


plt.figure(figsize=(7,4))
d = df.corr()['label'][:-1].abs().sort_values().plot(kind='bar', title='Highly correlated features with Class')
plt.show()


#import matplotlib.pyplot as plt
#df.plot(subplots =True, sharex = True, figsize = (20,50))
#plt.show()



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo


# 创建子图
fig = make_subplots(rows=len(df.columns), cols=1, shared_xaxes=True, vertical_spacing=0.02)

for i, column in enumerate(df.columns):
    fig.add_trace(
        go.Scatter(x=df.index, y=df[column], name=column),
        row=i+1, col=1
    )

fig.update_layout(height=3000, width=1000, title_text="Interactive Signal Plots", showlegend=False)

# 使用 Plotly 离线模式显示图形
pyo.plot(fig, filename='interactive_signal_plots.html')

# 自动打开生成的 HTML 文件，可以使用下面的代码
import webbrowser
webbrowser.open('interactive_signal_plots.html')

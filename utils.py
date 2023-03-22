import matplotlib.pyplot as plt
import numpy as np
import json
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_circles

from pred import predict_Adam, predict_RMSProp, predict_RN, predict_SGD_Moment

import pickle


def load_my_variable():
    with open("my_variable.pkl", "rb") as f:
        my_variable = pickle.load(f)
    return my_variable


fpredict = {
    "RN": predict_RN,
    "adam": predict_Adam,
    "SGD_Moment": predict_SGD_Moment,
    "RMSProp": predict_RMSProp,
}


def make_key(alpha, n_sample, layer):
    key = str(alpha) + "|" + str(n_sample) + "|" + str(layer)
    return key


def unmake_key(key):
    unkey = key.split("|")
    return unkey


def get_courbe_apprentissage(training_history_loss, training_history_acc):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Train Accuracy"))
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(training_history_loss) + 1)),
            y=training_history_loss,
            mode="lines",
            name="Train Loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(training_history_acc) + 1)),
            y=training_history_acc,
            mode="lines",
            name="Train Accuracy",
        )
    )
    fig.update_layout(
        height=400, width=1000, title="Courbe d'apprentissage", showlegend=True
    )
    return fig


def get_data_graph(X, y, n_samples):
    x_class0 = []
    y_class0 = []
    x_class1 = []
    y_class1 = []
    for i in range(n_samples):
        if y[0, i] == 0:
            x_class0.append(X[0, i])
            y_class0.append(X[1, i])
        else:
            x_class1.append(X[0, i])
            y_class1.append(X[1, i])

    # Créer un scatter plot pour chaque classe
    scatter_class0 = go.Scatter(
        x=x_class0,
        y=y_class0,
        mode="markers",
        marker=dict(color="red", symbol="circle"),
        name="Classe 0",
    )
    scatter_class1 = go.Scatter(
        x=x_class1,
        y=y_class1,
        mode="markers",
        marker=dict(color="blue", symbol="circle"),
        name="Classe 1",
    )

    # Créer le layout du graphique
    layout = go.Layout(
        title="Scatter plot des données make_circles",
        xaxis=dict(title="X1"),
        yaxis=dict(title="X2"),
    )

    # Créer un objet Figure avec les scatter plots et le layout
    fig = go.Figure(data=[scatter_class0, scatter_class1], layout=layout)
    return fig


def get_decision_boundary(X, y, parameters, model, fpredict=fpredict):
    fpredict = fpredict[model]
    # Générer une grille pour visualiser la frontière de décision
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01  # résolution de la grille
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Prédire les classes pour chaque point de la grille
    grid_X = np.c_[xx.ravel(), yy.ravel()].T
    grid_y_pred = fpredict(grid_X, parameters)
    grid_y_pred = grid_y_pred.reshape(xx.shape)

    # Prédire les classes pour les points du dataset original
    y_pred = fpredict(X, parameters)

    # Identifier les points correctement et incorrectement classés
    correct_indices = np.where(y_pred == y)[1]
    incorrect_indices = np.where(y_pred != y)[1]

    # Identifier les points de la classe 0 et 1
    class_0_indices = np.where(y == 0)[1]
    class_1_indices = np.where(y == 1)[1]

    # Créer un scatter plot pour les points correctement classés
    scatter_correct_0 = go.Scatter(
        x=X[0, np.intersect1d(correct_indices, class_0_indices)],
        y=X[1, np.intersect1d(correct_indices, class_0_indices)],
        mode="markers",
        marker=dict(color="blue", symbol="circle"),
        name="Classe 0 - Correct",
    )
    scatter_correct_1 = go.Scatter(
        x=X[0, np.intersect1d(correct_indices, class_1_indices)],
        y=X[1, np.intersect1d(correct_indices, class_1_indices)],
        mode="markers",
        marker=dict(color="blue", symbol="x"),
        name="Classe 1 - Correct",
    )

    # Créer un scatter plot pour les points incorrectement classés
    scatter_incorrect_0 = go.Scatter(
        x=X[0, np.intersect1d(incorrect_indices, class_0_indices)],
        y=X[1, np.intersect1d(incorrect_indices, class_0_indices)],
        mode="markers",
        marker=dict(color="red", symbol="circle"),
        name="Classe 0 - Incorrect",
    )
    scatter_incorrect_1 = go.Scatter(
        x=X[0, np.intersect1d(incorrect_indices, class_1_indices)],
        y=X[1, np.intersect1d(incorrect_indices, class_1_indices)],
        mode="markers",
        marker=dict(color="red", symbol="x"),
        name="Classe 1 - Incorrect",
    )

    # Créer un contour plot pour la ligne de séparation
    grid_y_pred = np.array([[int(value) for value in row] for row in grid_y_pred])
    contour = go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=grid_y_pred,
        contours=dict(
            coloring="lines",
            showlabels=False,
            start=0.5,
            end=0.5,
            labelfont={"size": 12, "color": "green"},
        ),
        line=dict(width=2, color="black"),
    )

    # Créer un objet Figure avec le contour plot et les scatter plots
    fig = go.Figure(
        data=[
            scatter_correct_0,
            scatter_correct_1,
            scatter_incorrect_0,
            scatter_incorrect_1,
        ]
    )

    fig = fig.add_trace(contour)
    return fig


def get_Xy(key):
    n_sample = unmake_key(key)
    n_sample = int(n_sample[1])
    X, y = make_circles(n_samples=n_sample, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    return X, y


def get_data(key, model):
    data = load_my_variable()
    return data[key][model]


def get_parameters(data):
    return data["parametres"]

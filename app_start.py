from dash import Dash, dcc, html
from dash import Input, Output
import plotly.graph_objs as go
from utils import (
    get_data,
    get_data_graph,
    get_parameters,
    make_key,
    fpredict,
    get_decision_boundary,
    get_Xy,
)

app = Dash(__name__)

learning_rate_list = [0.1, 0.01, 0.001]
n_samples_list = [100, 1000, 10000]
layer1 = (4, 4, 4)
layer2 = (8, 8, 8)
layer3 = (16, 16, 16)
hidden_layers_list = [layer1, layer2, layer3]
models = [i for i in fpredict]

app.layout = html.Div(
    [
        html.H1("Optimisation Convexe"),
        html.H1("Filter"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Learning Rate:"),
                        dcc.RadioItems(
                            id="learning-rate-radioitems",
                            options=[
                                {"label": str(lr), "value": lr}
                                for lr in learning_rate_list
                            ],
                            value=learning_rate_list[0],
                            className="content",
                        ),
                    ],
                    className="content",
                ),
                html.Div(
                    [
                        html.Label("Number of Samples:"),
                        dcc.RadioItems(
                            id="n-samples-radioitems",
                            options=[
                                {"label": str(ns), "value": ns} for ns in n_samples_list
                            ],
                            value=n_samples_list[0],
                            className="content",
                        ),
                    ],
                    className="content",
                ),
                html.Div(
                    [
                        html.Label("Hidden Layer:"),
                        dcc.RadioItems(
                            id="hidden-layer-radioitems",
                            options=[
                                {"label": str(layer), "value": i}
                                for i, layer in enumerate(hidden_layers_list)
                            ],
                            value=0,
                            className="content",
                        ),
                    ],
                    className="content",
                ),
                html.Div(
                    [
                        html.Label("Model"),
                        dcc.RadioItems(
                            id="model-radioitems",
                            options=[
                                {"label": str(model), "value": model}
                                for model in models
                            ],
                            value=models[0],
                            className="content",
                        ),
                    ],
                    className="content",
                ),
            ],
            className="FiltreG",
        ),
        html.H2("Loss Graph"),
        html.Div(dcc.Graph(id="loss-graph")),
        html.H2("Accuracy Graph"),
        html.Div(dcc.Graph(id="acc-graph")),
        html.H2("Other Stat"),
        html.Div(
            id="stat",
            className="FiltreG",
        ),
        html.H2("Apply Clasification"),
        html.H3("Data"),
        html.Div(dcc.Graph(id="Data-graph")),
        html.H3("Clasified-Data"),
        html.Div(dcc.Graph(id="fd-graph")),
    ]
)


@app.callback(
    Output("Data-graph", "figure"),
    [
        Input("learning-rate-radioitems", "value"),
        Input("n-samples-radioitems", "value"),
        Input("hidden-layer-radioitems", "value"),
        Input("model-radioitems", "value"),
    ],
)
def update_graph(alpha, n_sample, layer_idx, model):
    layer = hidden_layers_list[layer_idx]
    key = make_key(alpha, n_sample, layer)
    X, y = get_Xy(key)
    fig = get_data_graph(X, y, n_sample)
    return fig


@app.callback(
    Output("stat", "children"),
    [
        Input("learning-rate-radioitems", "value"),
        Input("n-samples-radioitems", "value"),
        Input("hidden-layer-radioitems", "value"),
        Input("model-radioitems", "value"),
    ],
)
def update_graph(alpha, n_sample, layer_idx, model):
    layer = hidden_layers_list[layer_idx]
    key = make_key(alpha, n_sample, layer)
    data = get_data(key, model)
    Time = data["Time"]
    n_iter_until_Conv = data["n_iter_until_Conv"]
    return f"The execution time is {Time}. \n  {model} have been conv at {n_iter_until_Conv} iteration. \n (Conv is when loss is lower than 0.5)"


@app.callback(
    Output("loss-graph", "figure"),
    [
        Input("learning-rate-radioitems", "value"),
        Input("n-samples-radioitems", "value"),
        Input("hidden-layer-radioitems", "value"),
        Input("model-radioitems", "value"),
    ],
)
def update_graph(alpha, n_sample, layer_idx, model):
    layer = hidden_layers_list[layer_idx]
    key = make_key(alpha, n_sample, layer)
    data = get_data(key, model)
    training_history_loss = data["training_history_loss"]
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(training_history_loss) + 1)),
            y=training_history_loss,
            mode="lines",
            name="Train Loss",
        )
    )
    fig.update_xaxes(title_text="Epochs")
    fig.update_yaxes(title_text="Loss")
    return fig


@app.callback(
    Output("acc-graph", "figure"),
    [
        Input("learning-rate-radioitems", "value"),
        Input("n-samples-radioitems", "value"),
        Input("hidden-layer-radioitems", "value"),
        Input("model-radioitems", "value"),
    ],
)
def update_graph(alpha, n_sample, layer_idx, model):
    layer = hidden_layers_list[layer_idx]
    key = make_key(alpha, n_sample, layer)
    data = get_data(key, model)
    training_history_acc = data["training_history_acc"]
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(training_history_acc) + 1)),
            y=training_history_acc,
            mode="lines",
            name="Train Loss",
        )
    )
    fig.update_xaxes(title_text="Epochs")
    fig.update_yaxes(title_text="Accuracy")
    return fig


@app.callback(
    Output("fd-graph", "figure"),
    [
        Input("learning-rate-radioitems", "value"),
        Input("n-samples-radioitems", "value"),
        Input("hidden-layer-radioitems", "value"),
        Input("model-radioitems", "value"),
    ],
)
def update_graph(alpha, n_sample, layer_idx, model):
    layer = hidden_layers_list[layer_idx]
    key = make_key(alpha, n_sample, layer)
    data = get_data(key, model)
    parameters = get_parameters(data)
    X, y = get_Xy(key)
    fig = get_decision_boundary(X, y, parameters, model)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)

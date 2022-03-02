import plotly.express as px
import plotly.graph_objects as go

from dataset import Dataset


def blankFig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template="plotly_dark")
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


def generate_fig_3D(data: Dataset, slider_value):
    """inputs: data: dictionary with features and image paths
    outputs: scatterPlot: plotly express 3D object with updates"""

    data.generate_clusters(slider_value)
    # Create 3D scatter plot to visualize
    scatterPlot = px.scatter_3d(
        data.embeddings,
        x=0,
        y=1,
        z=2,
        # color=[str(data.labels[i]) for i in range(len(data.labels))],
        color=data.labels,
        color_discrete_sequence=px.colors.sequential.Viridis,
        labels={"color": "cluster"},
    )

    scatterPlot.update_traces(
        marker_size=1.5,
        hoverinfo="none",
        hovertemplate=None,
    )

    scatterPlot.update_layout(template="plotly_dark")
    scatterPlot.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    return scatterPlot


def generate_fig_2D(embeddings, labels):
    """inputs: data: dictionary with features and image paths
    outputs: scatterPlot: plotly express 2D object with updates"""

    # Create 2D scatter graph with WebGL to handle large datasets
    scatterPlot = go.Figure(
        data=go.Scattergl(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode="markers",
            marker=dict(
                color=labels, colorscale="Viridis", line_width=0.5, showscale=True
            ),
        )
    )

    scatterPlot.update_layout(template="plotly_dark")
    scatterPlot.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    return scatterPlot

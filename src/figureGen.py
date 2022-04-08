import plotly.express as px
import plotly.graph_objects as go

from dataset import generate_clusters

import orjson


def blankFig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template="plotly_dark")
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


def generate_fig_3D(
    redis_client,
    session_id,
    dataset_name,
    n_neighbors_value,
    min_dist_value,
    min_cluster_size,
    min_samples,
):
    """TODO: UPDATE README inputs: data: dictionary with features and image paths
    outputs: scatterPlot: plotly express 3D object with updates"""

    embeddings, labels, percent_clustered = generate_clusters(
        redis_client,
        session_id,
        dataset_name,
        n_neighbors_value,
        min_dist_value,
        min_cluster_size,
        min_samples,
    )
    # Create 3D scatter plot to visualize
    img_paths = orjson.loads(redis_client.get(f"{session_id}:{dataset_name}:img_paths"))
    scatterPlot = px.scatter_3d(
        embeddings,
        x=0,
        y=1,
        z=2,
        # color=[str(data.labels[i]) for i in range(len(data.labels))],
        color=labels,
        hover_name=img_paths,
        # custom_data=img_paths, # must be the same dimension as x,y,z
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

    return scatterPlot, percent_clustered


def generate_fig_2D(embeddings, labels, img_paths):
    """inputs: data: dictionary with features and image paths
    outputs: scatterPlot: plotly express 2D object with updates"""
    # Create 2D scatter graph with WebGL to handle large datasets
    scatterPlot = go.Figure(
        data=go.Scattergl(
            x=embeddings[0][:, 0],
            y=embeddings[0][:, 1],
            mode="markers",
            marker=dict(
                color=labels, colorscale="Viridis", line_width=0.5, showscale=True
            ),
        )
    )
    scatterPlot.update_traces(
        customdata=img_paths,
        hoverinfo="none",
        hovertemplate=None,
    )
    scatterPlot.update_layout(template="plotly_dark")
    scatterPlot.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    return scatterPlot

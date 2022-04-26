import plotly.express as px
import plotly.graph_objects as go

from dataset import generate_clusters, get_img_paths

import orjson


def blankFig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template="plotly_dark")
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


def generate_fig(
    redis_client,
    session_id,
    dataset_name,
    n_neighbors_value,
    min_dist_value,
    min_cluster_size,
    min_samples,
    n_components,
):
    """TODO: UPDATE README inputs: data: dictionary with features and image paths
    outputs: scatterPlot: plotly express 3D object with updates"""

    # Create 3D scattergl plot
    if n_components == 3:
        embeddings, labels, percent_clustered = generate_clusters(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors_value,
            min_dist_value,
            min_cluster_size,
            min_samples,
            n_components=3,
        )
        img_paths = get_img_paths(redis_client, session_id, dataset_name)
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
            # color_discrete_map={"-1": "rgb(255,0,0)"},
            labels={"color": "cluster"},
        )
        scatterPlot.update_traces(
            marker_size=1.5,
            hoverinfo="none",
            hovertemplate=None,
        )

    # Create 2D scattergl plot
    elif n_components == 2:
        embeddings, labels, percent_clustered = generate_clusters(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors_value,
            min_dist_value,
            min_cluster_size,
            min_samples,
            n_components=2,
        )
        img_paths = get_img_paths(redis_client, session_id, dataset_name)
        # scatterPlot = go.Figure(
        #     data=go.Scattergl(
        #         x=embeddings[:, 0],
        #         y=embeddings[:, 1],
        #         mode="markers",
        #         marker=dict(
        #             color=labels, colorscale="Viridis", line_width=0.5, showscale=True
        #         ),

        #     )
        # )
        scatterPlot = px.scatter(
            embeddings,
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            # color=[str(data.labels[i]) for i in range(len(data.labels))],
            color=labels,
            hover_name=img_paths,
            # custom_data=img_paths, # must be the same dimension as x,y,z
            color_discrete_sequence=px.colors.sequential.Viridis,
            # color_discrete_map={"-1": "rgb(255,0,0)"},
            labels={"color": "cluster"},
        )
        scatterPlot.update_traces(
            marker=dict(size=3),
            customdata=img_paths,
            hoverinfo="none",
            hovertemplate=None,
        )
    else:
        print(f"[ERROR]: illegal n_components={n_components}. Should be 2 or 3.")
        return
    scatterPlot.update_layout(template="plotly_dark")
    scatterPlot.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    return scatterPlot, percent_clustered

import os
import uuid
import orjson
import base64
import threading
from datetime import datetime

import flask
import dash
from redis import Redis
import plotly.express as px
from dash import html, dcc, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# open source dash functionality
# https://github.com/np-8/dash-uploader
import dash_uploader as du

# Local Modules
from config import config
from worker import poll_remove_user_data
from dataset import (
    move_unzip_uploaded_file,
    create_clusters_zip,
    gen_img_uri,
    prepare_preview_download,
    del_folder_contents,
    find_similar_imgs,
    parse_tqdm_progress,
)
from componentBuilder import (
    create_LR_label,
    gen_img_preview,
    gen_download_button,
    create_info_loading,
    create_title_with_button,
)
from figureGen import blankFig, generate_fig

################################################################################
""" Initialize Flask Server, Redis, Dash App, Uploader: """
################################################################################

app = flask.Flask(__name__)

# Initialize Cache
redis_client = Redis(
    host=config["app"]["redis_host"],
    port=config["app"]["redis_port"],
    db=0,
    decode_responses=True,
)
# Each time the app server is restarted, redis will be cleared
redis_client.flushall()

# Define dash app
dash_app = dash.Dash(
    __name__,
    server=app,
    external_stylesheets=[dbc.themes.CYBORG],
)

# The temporary folder where uploads are stored before being reorganized to user assets.
du.configure_upload(dash_app, "assets/temp")

################################################################################
""" Dash Components: """
################################################################################

horz_line = html.Hr()

titles_color = "#acdcf2"

# title of the dash
title = dbc.Card(
    [
        html.H5(
            "Dataset Clustering Dash",
            style={
                "color": titles_color,
                "text-align": "center",
                "padding": "15px",
            },
        )
    ]
)


# ------------------------------------------------------------------------------
#   3D Graph and it's Control Components
# ------------------------------------------------------------------------------
graph_with_loading_animation = dcc.Loading(
    children=[
        dcc.Graph(
            id="main_graph",
            clear_on_unhover=True,
            figure=blankFig(),
            loading_state={"is_loading": False},
            style={"height": "70vh"},
        )
    ],
    type="graph",
)
dash_uploader_style = {  # wrapper div style
    "textAlign": "center",
    "width": "100%",
    # "height": "50px",
    "padding": "10px",
    "display": "inline-block",
}
file_info = create_info_loading(
    id="file_info", children=["Upload FolderOfImagesYouWishToCluster.zip"]
)
data_info_text = html.Span(
    id="data_info",
    children=["Then click the generate graphs button below."],
    style={"font-size": "15px"},
)
data_info = html.Div(
    id="data_info_div",
    children=[data_info_text],
    style={
        "textAlign": "center",
        "padding": "10px",
    },
)
n_neighbors_left_text = (
    "This parameter controls how UMAP balances local versus global structure in the data. "
    "As n_neighbors is increased UMAP manages to see more of the overall "
    "structure of the data, gluing more components together, and better coverying "
    "the broader structure of the data. As n_neighbors increases further, more and "
    "more focus in placed on the overall structure of the data. This results in "
    "a plot where the overall structure is well captured, but at the loss of some of "
    "the finer local structure (individual images may no longer necessarily be "
    "immediately next to their closest match)."
)
n_neighbors_slider = [
    create_LR_label(
        id="n_neighbors_label",
        leftText="[UMAP]:",
        rightText="n_neighbors",
        tip_text_left=(
            "Uniform Manifold Approximation and Projection (UMAP) is a "
            "dimension reduction technique used here to allow visualisation."
            "This affects the positions of the data points and shapes of the cluster clouds."
        ),
        tip_text_right=n_neighbors_left_text,
    ),
    dcc.Slider(
        min=10,
        max=240,
        step=10,
        value=80,
        id="n_neighbors_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]

min_dist_left_text = (
    "The min_dist parameter controls how tightly UMAP is allowed to pack points "
    "together. It, quite literally, provides the minimum distance apart that points "
    "are allowed to be in the low dimensional representation. This means that low "
    "values of min_dist will result in clumpier embeddings. This can be useful if "
    "you are interested in clustering, or in finer topological structure. Larger "
    "values of min_dist will prevent UMAP from packing points together and will "
    "focus on the preservation of the broad topological structure instead."
)
min_dist_slider = [
    create_LR_label(
        id="min_dist_label",
        leftText="[UMAP]:",
        rightText="min_dist",
        tip_text_left=(
            "Uniform Manifold Approximation and Projection (UMAP) is a "
            "dimension reduction technique used here to allow visualisation."
            "This affects the positions of the data points and shapes of the cluster clouds."
        ),
        tip_text_right=min_dist_left_text,
    ),
    dcc.Slider(
        min=0.0,
        max=1,
        step=0.01,
        value=0.0,
        id="min_dist_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]

min_cluster_size_left_text = (
    "Set it to the smallest size grouping that you wish to consider a cluster. "
    "It can have slightly non-obvious effects however (unless you fix min_samples)."
)
min_cluster_size_slider = [
    create_LR_label(
        id="min_cluster_size_label",
        leftText="[HDBSCAN]:",
        rightText="min_cluster_size",
        tip_text_left=(
            "Hierarchical Density-Based Spatial Clustering of Applications with Noise."
            "This affects the labels assigned to the data i.e the cluster each point is assigned to."
        ),
        tip_text_right=min_cluster_size_left_text,
    ),
    dcc.Slider(
        min=5,
        max=405,
        step=10,
        value=155,
        id="min_cluster_size_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]

min_samples_left_text = (
    "The simplest intuition for what min_samples does is provide a measure of how "
    "conservative you want your clustering to be. The larger the value of min_samples "
    "you provide, the more conservative the clustering - so more points will be "
    "declared as noise, and clusters will be restricted to progressively more "
    "dense areas. Steadily increasing min_samples will, as we saw in the examples "
    "above, make the clustering progressively more conservative."
)
min_samples_slider = [
    create_LR_label(
        id="min_samples_label",
        leftText="[HDBSCAN]:",
        rightText="min_samples",
        tip_text_left=(
            "Hierarchical Density-Based Spatial Clustering of Applications with Noise."
            "This affects the labels assigned to the data i.e the cluster each point is assigned to."
        ),
        tip_text_right=min_samples_left_text,
    ),
    dcc.Slider(
        min=1,
        max=200,
        step=1,
        value=10,
        id="min_samples_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]
download_clusters_button = gen_download_button(
    id="download_clusters_button",
    children=["Download 3D Clusters"],
    href=dash_app.get_asset_url("clusters.zip"),
)
graph_3D_button = dbc.Button(
    children=["Generate Graphs"],
    id="graph_3D_button",
    n_clicks=0,
    disabled=True,
    color="primary",
)
card_3D_buttons = html.Div(
    children=[html.Div(download_clusters_button), html.Div(graph_3D_button)],
    style={"display": "flex", "justify-content": "space-around"},
)
controls_title = html.Div(
    [html.Span(["Control Parameters:"])],
    style={
        "text-align": "center",
        "padding": "15px",
        "color": titles_color,
        "font-size": "20px",
    },
)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#   2D Graph Components
# ------------------------------------------------------------------------------
graph_2D = dcc.Loading(
    children=[
        dcc.Graph(
            id="graph_2D",
            clear_on_unhover=True,
            figure=blankFig(),
            loading_state={"is_loading": True},
        )
    ],
    type="graph",
)
image_preview = dcc.Loading(
    children=[
        html.Div(
            id="image_preview",
            children=[
                "Use the box or lasso selector on the 2D graph to preview selected images."
            ],
            style={
                "textAlign": "center",
                #'margin': '10px'
            },
        )
    ],
    type="cube",
)

download_preview_button = gen_download_button(
    id="download_preview_button",
    children=["Download"],
    href=dash_app.get_asset_url("preview_2D.zip"),
)
preview_download_name_input = html.Div(
    [
        dbc.Input(
            placeholder="Enter preview download name. Default: 'selection_preview'...",
            type="text",
        ),
    ]
)
preview_title = create_title_with_button([download_preview_button], html.Div())
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#   Image Similarity Search Components
# ------------------------------------------------------------------------------
upload_image_file_button = dcc.Upload(
    id="upload_image_file_button",
    children=["Drag/Select Image File"],
    style={
        "height": "41px",
        "lineHeight": "41px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
    },
    # Don't allow multiple files to be uploaded
    multiple=False,
    disabled=False,
)
download_search_button = html.Div(
    [
        gen_download_button(
            id="download_search_button",
            children=["Download Results"],
            href="",
        )
    ],
    style={"display": "flex", "align-items": "center", "justify-content": "center"},
)
image_file_info = create_info_loading(id="image_file_info", children=[""])
preview_test_image = html.Div(
    id="preview_test_image",
    children=[
        html.H6(
            children=[""],
            style={"textAlign": "center", "padding": "10px"},
        )
    ],
)
search_preview = html.Div(
    id="search_preview",
    children=["Upload image to find similar images."],
    style={
        "textAlign": "center",
        #'margin': '10px'
    },
)
search_preview_results_title = html.H6(
    children="Results:",
    style={
        "textAlign": "left",
        "padding": "10px",
    },
)
search_preview_main_title = html.H5(
    children="Reverse Image Search",
    style={
        "textAlign": "center",
        "padding": "10px",
    },
)
# ------------------------------------------------------------------------------

# Not in use currently
jump_up_button = dbc.Button(
    children=["Jump to Top"],
    id="jump_up_button",
    n_clicks=0,
    disabled=False,
    color="primary",
)
preview_and_search_card = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Selection Preview", tab_id="tab-1"),
                    dbc.Tab(label="Reverse Image Search", tab_id="tab-2"),
                ],
                id="card-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.Div(id="card-content")),
    ]
)
# Progress bar components
progress_text = html.Div(
    id="progress_text",
    children=[html.Span("Beginning...")],
    style={"font-size": "small"},
)
progress_interval = dcc.Interval(id="timer_progress", interval=1000, disabled=True)
# User timestamp updater
activity_interval = dcc.Interval(
    id="interval-component",
    interval=1 * 20000,  # 20000 milliseconds = 20 sec
    n_intervals=0,
)

################################################################################
""" Dash UI Layout: """
################################################################################
def serve_layout():
    session_id = str(uuid.uuid4())
    # add user id to redis list
    redis_client.rpush("users", session_id)
    return dbc.Container(
        [
            activity_interval,
            progress_interval,
            # Empty div used since callbacks with no output are not allowed
            html.Div(id="dummy1"),
            # Using dummy div to keep info % clustered output (dash doesnt allow two callbacks with same output)
            html.Div(id="dummy_percent_clustered_data", style={"display": "none"}),
            # In browser storage objects
            dcc.Store(id="session_id", data=session_id),
            dcc.Store(id="data_uploaded_flag", data=False),
            dcc.Store(id="data_clustered_flag", data=False),
            horz_line,
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.Row(
                                    children=[
                                        dbc.Row(title),
                                        # dbc.Row(horz_line),
                                        dbc.Row(file_info),
                                        dbc.Row(data_info),
                                        dbc.Row(
                                            [
                                                # This component is verbosely placed in layout as it
                                                # requires session_id to be dynamically generated
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            du.Upload(
                                                                id="dash_uploader",
                                                                text="Drag/Select Zip",
                                                                max_files=1,
                                                                filetypes=["zip"],
                                                                max_file_size=20024,
                                                                upload_id=session_id,
                                                            ),
                                                            style=dash_uploader_style,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                controls_title,
                                                dbc.Row(n_neighbors_slider),
                                                dbc.Row(min_dist_slider),
                                                horz_line,
                                                dbc.Row(min_cluster_size_slider),
                                                dbc.Row(min_samples_slider),
                                            ]
                                        ),
                                        horz_line,
                                        dbc.Row(
                                            [card_3D_buttons],
                                            justify="center",
                                            align="center",
                                        ),
                                    ],
                                    justify="center",
                                    align="center",
                                ),
                            ],
                            body=True,
                            style={"height": "124vh"},
                        ),
                        md=3,
                        align="center",
                    ),
                    dbc.Col(
                        [
                            # 3D Graph and its components
                            dbc.Row(
                                children=[
                                    graph_with_loading_animation,
                                    dcc.Tooltip(
                                        id="main_graph_tooltip", direction="right"
                                    ),
                                    dcc.Download(id="main_graph_download"),
                                ],
                                style={"height": "70vh"},
                            ),
                            horz_line,
                            dbc.Row(
                                [
                                    # 2D Graph and its components
                                    dbc.Col(
                                        children=[
                                            graph_2D,
                                            dcc.Tooltip(
                                                id="graph_2D_tooltip",
                                                direction="right",
                                            ),
                                            dcc.Download(id="graph_2D_download"),
                                            horz_line,
                                            # Tabbed card for preview and search here
                                            dbc.Row(
                                                children=(
                                                    dbc.Col(
                                                        children=[
                                                            preview_and_search_card
                                                        ],
                                                        width="auto",
                                                        md=12,
                                                    ),
                                                ),
                                                justify="center",
                                                align="center",
                                            ),
                                            horz_line,
                                            # content will be rendered in this element
                                            html.Div(id="page-content"),
                                        ],
                                    ),
                                ],
                                style={"height": "50vh"},
                                justify="center",
                                align="start",
                            ),
                        ],
                        md=9,
                    ),
                ],
                justify="center",
                align="center",
            ),
        ],
        fluid=True,
    )


# Dynamically serve layout for each user that connects
dash_app.layout = serve_layout


################################################################################
""" Callback Functions: """
################################################################################
########################################################################
""" [CALLBACK]: upload zip file and extract features/paths """
########################################################################
@dash_app.callback(
    [
        Output("data_uploaded_flag", "data"),
        Output("file_info", "children"),
        Output("graph_3D_button", "disabled"),
    ],
    [Input("dash_uploader", "isCompleted")],
    [
        State("dash_uploader", "fileNames"),
        State("session_id", "data"),
    ],
    prevent_initial_call=True,
)
def upload_handler(isCompleted: bool, filename: list, session_id: str):
    """upload_handler Sets the dataset name in redis and unzips/organizes the
    uploaded dataset into the users assets directory.
    Args:
        isCompleted (bool): This callback is triggered by updates to the 'isCompleted'
        attribute of the dash-uploader component.
        filename (list): Filename as provided by the dash-uploader component,
        stored as a string in a list.
        session_id (str): The uuid assigned to the users browser session,
        used to identify their directory.
    """
    if isCompleted:
        global redis_client
        # remove the extension part (.zip) from the filename
        print("filename", filename)
        dataset_name = os.path.splitext(filename[0])[0]
        # store dataset name in redis
        if not redis_client.sismember(f"{session_id}:datasets", dataset_name):
            # arrange and extract files
            move_unzip_uploaded_file(session_id, filename)
            # should only add dataset name to our redis Set, IF zip upload/extract succesful
            redis_client.sadd(f"{session_id}:datasets", dataset_name)
        # Update the current dataset being used by user
        redis_client.set(f"{session_id}:curr_dataset", dataset_name)

        outputText = create_LR_label(
            id="file_info_label",
            leftText="[FILE]:",
            rightText=dataset_name,
        )
        return [True], outputText, False
    return no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Insert progress bar """
########################################################################
@dash_app.callback(
    [
        Output("data_info_div", "children"),
        Output("timer_progress", "disabled"),
        Output("dash_uploader", "disabled"),
    ],
    [
        Input("graph_3D_button", "n_clicks"),
        Input("graph_2D", "figure"),
    ],
    [
        State("dummy_percent_clustered_data", "children"),
        State("data_uploaded_flag", "data"),
    ],
    prevent_initial_call=True,
)
def insert_prog_bar(
    n_clicks: int, figure, percent_clustered_components, data_uploaded_flag
):
    if data_uploaded_flag:
        ctx = dash.callback_context
        if ctx.triggered[0]["prop_id"] == "graph_3D_button.n_clicks":
            return progress_text, False, True
        # otherwise triggered by update to the 2D figure, use dummy div
        # holding percent clustered data to update progress area
        else:
            return percent_clustered_components, True, False
    else:
        return no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Progress bar updater"""
########################################################################
@dash_app.callback(
    [Output("progress_text", "children")],
    [Input("timer_progress", "n_intervals")],
    State("session_id", "data"),
    prevent_initial_call=True,
)
def callback_progress(n_intervals: int, session_id):
    # This fxn returns the last line written to the progress file.
    last_line = parse_tqdm_progress(session_id)
    return [last_line]


########################################################################
""" [CALLBACK]: Create 3D graph and update cluster download """
########################################################################
@dash_app.callback(
    [
        Output("main_graph", "figure"),
        Output("data_clustered_flag", "data"),
        Output("dummy_percent_clustered_data", "children"),
        Output("download_clusters_button", "disabled"),
        Output("download_clusters_button", "href"),
    ],
    [Input("graph_3D_button", "n_clicks"), Input("dash_uploader", "isCompleted")],
    [
        State("data_uploaded_flag", "data"),
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
        State("session_id", "data"),
    ],
    prevent_initial_call=True,
)
def update_output(
    n_clicks,
    isCompleted,
    data_uploaded_flag,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    session_id,
):
    # After feature extraction, enable 3D graph gen button
    ctx = dash.callback_context
    # check if this callback was fired after a dataset upload,
    # if yes set data_info back to button click instruction.
    if (
        ctx.triggered[0]["prop_id"] == "dash_uploader.isCompleted"
        and ctx.triggered[0]["value"]
    ):
        data_info_text = create_info_loading(
            id="data_info", children=["Click the generate graphs button below."]
        )
        return no_update, no_update, data_info_text, no_update, no_update

    if n_clicks and data_uploaded_flag:
        global redis_client
        # Generate the 3D graph using most recently uploaded dataset
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        figure, percent_clustered = generate_fig(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
            n_components=3,
        )
        # Output Clustering statistics
        output_text = create_LR_label(
            id="percentClusteredText",
            leftText="[INFO]:",
            rightText=f"{percent_clustered}% clustered",
        )
        # arrange zip files to create download
        create_clusters_zip(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
            n_components=3,
        )
        dataset_dir = f"assets/{session_id}/{dataset_name}"
        cluster_zip_path = os.path.join(dataset_dir, "clusters.zip")

        return figure, [True], output_text, False, cluster_zip_path
    else:
        return no_update, no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 2D Graph """
########################################################################
@dash_app.callback(
    [Output("graph_2D", "figure")],
    [Input("data_clustered_flag", "data")],
    [
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
        State("session_id", "data"),
    ],
)
def create_graph_2D(
    data_clustered_flag,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    session_id,
):
    if data_clustered_flag:
        global redis_client
        # Calculate UMAP embeddings with two components.
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        fig, percent_clustered = generate_fig(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
            n_components=2,
        )

        return [fig]
    else:
        return no_update


########################################################################
""" [CALLBACK]: 3D hover to show image """
########################################################################
@dash_app.callback(
    [
        Output("main_graph_tooltip", "show"),
        Output("main_graph_tooltip", "bbox"),
        Output("main_graph_tooltip", "children"),
        # Output("main_graph_tooltip", "style"),
    ],
    [Input("main_graph", "hoverData")],
    [
        State("data_clustered_flag", "data"),
        State("session_id", "data"),
    ],
)
def display_hover(hoverData, data_clustered_flag, session_id):
    if (hoverData is None) or (not data_clustered_flag):
        return False, no_update, no_update
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    hover_data = hoverData["points"][0]
    # Load image with pillow
    hover_img_index = hover_data["pointNumber"]
    # demo only shows the first point, but other points may also be available
    img_path = hover_data["hovertext"]
    cluster = hover_data["marker.color"]
    im_uri = gen_img_uri(
        redis_client, session_id, dataset_name, hover_img_index, img_path
    )
    bbox = hover_data["bbox"]
    img_name = img_path.split("/")[-1]
    children = [
        html.Div(
            [
                html.Img(
                    src=im_uri,
                    style={"width": "150px"},
                ),
                # dbc.FormText(f"Name: {img_name}", style={"font-size": "x-small"}),
                html.Span(
                    f"cluster: {cluster}   ,   index: {hover_img_index}",
                    style={
                        "display": "block",
                        "font-size": "small",
                        "color": "black",
                        "textAlign": "center",
                    },
                ),
            ],
            style={"white-space": "normal", "display": "block", "margin": "0 auto"},
        )
    ]
    return True, bbox, children


########################################################################
""" [CALLBACK]: 2D Hover preview"""
########################################################################
@dash_app.callback(
    [
        Output("graph_2D_tooltip", "show"),
        Output("graph_2D_tooltip", "bbox"),
        Output("graph_2D_tooltip", "children"),
    ],
    [Input("graph_2D", "hoverData")],
    State("session_id", "data"),
)
def display_hover2D(hoverData, session_id):
    if hoverData is None:
        return False, no_update, no_update
    # Load image with pillow
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    hover_data = hoverData["points"][0]
    hover_img_index = hover_data["pointNumber"]
    img_path = hover_data["customdata"]
    cluster = hover_data["marker.color"]
    try:
        im_uri = gen_img_uri(
            redis_client, session_id, dataset_name, hover_img_index, img_path
        )
    except:
        return False, no_update, no_update
    # demo only shows the first point, but other points may also be available

    bbox = hover_data["bbox"]
    children = [
        html.Img(
            src=im_uri,
            style={"width": "150px"},
        ),
        html.P(
            f"cluster: {cluster}   ,   index: {hover_img_index}",
            style={"font-size": "small", "color": "black", "textAlign": "center"},
        ),
    ]
    return True, bbox, children


########################################################################
""" [CALLBACK]: 3D Click Download"""
########################################################################
@dash_app.callback(
    Output("main_graph_download", "data"),
    Input("main_graph", "clickData"),
    State("session_id", "data"),
    prevent_initial_call=True,
)
def func(clickData, session_id):
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    img_paths = orjson.loads(redis_client.get(f"{session_id}:{dataset_name}:img_paths"))
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Click Download"""
########################################################################
@dash_app.callback(
    Output("graph_2D_download", "data"),
    Input("graph_2D", "clickData"),
    State("session_id", "data"),
    prevent_initial_call=True,
)
def func(clickData, session_id):
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    img_paths = orjson.loads(redis_client.get(f"{session_id}:{dataset_name}:img_paths"))
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Lasso to preview"""
########################################################################
@dash_app.callback(
    [
        Output("image_preview", "children"),
        Output("download_preview_button", "disabled"),
        Output("download_preview_button", "href"),
    ],
    [Input("graph_2D", "selectedData")],
    [State("session_id", "data"), State("card-tabs", "active_tab")],
    prevent_initial_call=True,
)
def display_selected_data(selectedData, session_id, active_tab):
    """get all the selected point idx from the selectedData dictionary
    then get the img path for each idx, then create previwe with paths"""
    if selectedData["points"] and active_tab == "tab-1":
        global redis_client

        points = selectedData["points"]

        selected_img_idxs = []
        for i in range(len(points)):
            idx = points[i]["pointNumber"]
            selected_img_idxs.append(idx)

        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        prepare_preview_download(
            redis_client, session_id, dataset_name, selected_img_idxs, "preview_2D"
        )
        dataset_dir = f"assets/{session_id}/{dataset_name}"
        preview_zip_path = os.path.join(dataset_dir, "preview_2D.zip")
        return (
            gen_img_preview(
                redis_client, session_id, dataset_name, selected_img_idxs, scale=0.915
            ),
            False,
            preview_zip_path,
        )
    else:
        # no points selected
        return no_update, no_update, no_update


########################################################################
""" [CALLBACK]: upload image file and find similar images """
########################################################################
@dash_app.callback(
    [
        Output("image_file_info", "children"),
        Output("preview_test_image", "children"),
        Output("search_preview", "children"),
        Output("download_search_button", "href"),
        Output("download_search_button", "disabled"),
    ],
    [Input("upload_image_file_button", "contents")],
    [
        State("upload_image_file_button", "filename"),
        State("session_id", "data"),
        State("data_clustered_flag", "data"),
    ],
)
def uploadData(content, filename, session_id, data_clustered_flag):
    if content is not None and data_clustered_flag:
        content_type, content_str = content.split(",")
        output_filename = (
            html.P(
                children=["[Filename]: ", html.Br(), filename],
                style={
                    "textAlign": "left",
                    "padding": "10px",
                    "word-wrap": "break-word",
                },
            ),
        )
        # TODO: Why am I using content and not content_str? maybe because html img default something?
        test_image = html.Div(
            [
                html.Img(
                    src=content,
                    style={
                        "width": "100%",
                        "height": "100%",
                        "min-height": "10vh",
                        "max-height": "15vh",
                        "object-fit": "contain",
                    },
                )
            ],
        )
        # content is an encoded picture
        # decode picture and save as file
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        test_image_folder = f"assets/{session_id}/{dataset_name}/image_search/input/"
        if not os.path.exists(test_image_folder):
            os.makedirs(test_image_folder)
        del_folder_contents(test_image_folder)
        test_image_path = os.path.join(test_image_folder, filename)
        with open(test_image_path, "wb") as f:
            img_bytes = base64.b64decode(content.split("base64,")[-1])
            f.write(img_bytes)
        # print("test_image_path is:", test_image_path)
        # search for 6 similar images
        result_idxs = find_similar_imgs(session_id, dataset_name, test_image_path)
        # display them
        result_preview = gen_img_preview(
            redis_client, session_id, dataset_name, result_idxs[0], scale=0.99
        )
        prepare_preview_download(
            redis_client,
            session_id,
            dataset_name,
            result_idxs[0],
            "image_search/search_results",
        )
        dataset_dir = f"assets/{session_id}/{dataset_name}/image_search"
        preview_zip_path = os.path.join(dataset_dir, "search_results.zip")
        return output_filename, test_image, result_preview, preview_zip_path, False
    return no_update, no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Selection Preview / Image serch tab """
########################################################################
@dash_app.callback(
    Output("card-content", "children"),
    [Input("card-tabs", "active_tab")],
    State("session_id", "data"),
)
def tab_content(active_tab, session_id):
    if active_tab == "tab-1":
        content = [
            dbc.Row(
                [preview_title],
            ),
            # horz_line,
            dbc.Row(
                dbc.Col(image_preview),
                justify="center",
                align="center",
            ),
        ]

    else:
        content = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row([upload_image_file_button]),
                            horz_line,
                            dbc.Row([download_search_button]),
                            horz_line,
                            dbc.Row([image_file_info]),
                        ],
                        style={"max-height": "30vh"},
                        md=2,
                    ),
                    dbc.Col(
                        [
                            preview_test_image,
                        ],
                        style={"max-height": "30vh"},
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Row([search_preview]),
                        ],
                        style={"max-height": "30vh"},
                        md=8,
                    ),
                ],
                justify="center",
                align="center",
            ),
        ]

    return content


########################################################################
""" [CALLBACK]: Interval to check user activity"""
########################################################################
@dash_app.callback(
    Output("dummy1", "children"),
    Input("interval-component", "n_intervals"),
    State("session_id", "data"),
)
def update_activity_timestamp(n, session_id):
    # store and update redis timestamps for each user
    global redis_client
    curr_time = str(datetime.now())
    redis_client.set(f"{session_id}:latest_timestamp", curr_time)
    return no_update


########################################################################
""" [CALLBACK]: Run server """
########################################################################
if __name__ == "__main__":
    # start thread for user activity worker here
    x = threading.Thread(
        target=poll_remove_user_data, args=(redis_client,), daemon=True
    )
    x.start()
    app.run(
        debug=False, host=config["app"]["flask_host"], port=config["app"]["flask_port"]
    )

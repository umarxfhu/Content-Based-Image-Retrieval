import os
import uuid
import orjson
import base64

import flask
import dash
from redis import Redis
from dash import html, dcc, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from dataset import (
    decode_and_extract_zip,
    create_clusters_zip,
    setup_umap,
    get_labels,
    gen_img_uri,
    prepare_preview_download,
    del_folder_contents,
    find_similar_imgs,
)
from componentBuilder import (
    create_LR_label,
    gen_img_preview,
    gen_download_button,
    create_info_loading,
    create_title_with_button,
)
from figureGen import blankFig, generate_fig_3D, generate_fig_2D

################################################################################
""" Global variables: """
################################################################################

config = {
    "debug": True,
    "assets": "assets",
    "unzipped_dir": "assets/unzipped",
    "resources_dir": "assets/resources",
    "clusters_dir": "assets/clusters",
}


################################################################################
""" Initialize Dash App: """
################################################################################

server = flask.Flask(__name__)
# Initialize Cache
redis_client = Redis(host="redis", port=6379, db=0, decode_responses=True)
# [TODO]: before deploying consider memory management i.e when should you clear redis
redis_client.flushall()
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG])


################################################################################
""" Dash Components: """
################################################################################

horz_line = html.Hr()

# title of the dash
title = html.H4(
    "Dataset Clustering Dashboard",
    style={"color": "white", "text-align": "center", "padding": "10px"},
)

# ------------------------------------------------------------------------------
#   3D Graph and it's Control Components
# ------------------------------------------------------------------------------
uploadButton = dcc.Upload(
    id="upload-image-folder",
    children=["Drag/Select ZIP"],
    style={
        # 'width': '100%',
        "height": "60px",
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "margin": "10px",
    },
    accept=".zip",
    # Don't allow multiple files to be uploaded
    multiple=False,
)
graphWithLoadingAnimation = dcc.Loading(
    children=[
        dcc.Graph(
            id="mainGraph",
            clear_on_unhover=True,
            figure=blankFig(),
            loading_state={"is_loading": True},
            style={"height": "70vh"},
        )
    ],
    type="graph",
)
fileInfo = create_info_loading(
    id="fileInfo", children=["Please upload FolderOfImagesYouWishToCluster.zip"]
)
dataInfo = create_info_loading(
    id="dataInfo", children=["Then click the generate graphs button below."]
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
        ),
        tip_text_right=n_neighbors_left_text,
    ),
    dcc.Slider(
        min=20,
        max=240,
        step=20,
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
        ),
        tip_text_right=min_dist_left_text,
    ),
    dcc.Slider(
        min=0.0,
        max=0.5,
        step=0.1,
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
        tip_text_left="Hierarchical Density-Based Spatial Clustering of Applications with Noise.",
        tip_text_right=min_cluster_size_left_text,
    ),
    dcc.Slider(
        min=20,
        max=400,
        step=20,
        value=240,
        id="min_cluster_size_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]

min_samples_left_text = (
    "The simplest intuition for what min_samples does is provide a measure of how "
    "conservative you want you clustering to be. The larger the value of min_samples "
    "you provide, the more conservative the clustering - more points will be "
    "declared as noise, and clusters will be restricted to progressively more "
    "dense areas. Steadily increasing min_samples will, as we saw in the examples "
    "above, make the clustering progressively more conservative."
)
min_samples_slider = [
    create_LR_label(
        id="min_samples_label",
        leftText="[HDBSCAN]:",
        rightText="min_samples",
        tip_text_left="Hierarchical Density-Based Spatial Clustering of Applications with Noise.",
        tip_text_right=min_samples_left_text,
    ),
    dcc.Slider(
        min=1,
        max=100,
        step=1,
        value=10,
        id="min_samples_slider",
        marks=None,
        tooltip={"placement": "bottom"},
    ),
]
download_clusters_button = gen_download_button(
    id="download_clusters_button",
    children=["Download Clusters"],
    href=app.get_asset_url("clusters.zip"),
)
graph3DButton = dbc.Button(
    children=["Generate Graphs"],
    id="graph3DButton",
    n_clicks=0,
    disabled=True,
    color="primary",
)
card3DButtons = html.Div(
    children=[html.Div(download_clusters_button), html.Div(graph3DButton)],
    style={"display": "flex", "justify-content": "space-around"},
)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#   2D Graph Components
# ------------------------------------------------------------------------------
graph2D = dcc.Loading(
    children=[
        dcc.Graph(
            id="graph2D",
            clear_on_unhover=True,
            figure=blankFig(),
            loading_state={"is_loading": True},
        )
    ],
    type="graph",
)
imagePreview = html.Div(
    id="imagePreview",
    children=[
        "Use the box or lasso selector on the 2D graph to preview selected images."
    ],
    style={
        "textAlign": "center",
        #'margin': '10px'
    },
)
download_preview_button = gen_download_button(
    id="download_preview_button",
    children=["Download"],
    href=app.get_asset_url("preview_2D.zip"),
)
preview_title = create_title_with_button(["Selection Preview"], download_preview_button)
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
    disabled=True,
)
download_search_button = gen_download_button(
    id="download_search_button",
    children=["Download"],
    href="",
)
image_search_title = html.Div(
    children=[
        html.Div(
            [upload_image_file_button],
            style={
                "textAlign": "left",
                "width": "47%",
                "display": "inline-block",
                "margin-left": "2%",
            },
        ),
        html.Div(
            [download_search_button],
            style={
                "textAlign": "right",
                "width": "50%",
                "display": "inline-block",
            },
        ),
    ],
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
    children=["Upload image to view similar images."],
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


################################################################################
""" Dash UI Layout: """
################################################################################


def serve_layout():
    session_id = str(uuid.uuid4())
    redis_client.rpush("users", session_id)
    return dbc.Container(
        [
            # In browser storage objects
            dcc.Store(id="session-id", data=session_id),
            dcc.Store(id="dataProcessedFlag", data=False),
            dcc.Store(id="dataClusteredFlag", data=False),
            dbc.Row(title),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.Row(
                                    children=[
                                        dbc.Row(fileInfo),
                                        dbc.Row(dataInfo),
                                        dbc.Row(uploadButton),
                                        horz_line,
                                        dbc.Row(n_neighbors_slider),
                                        dbc.Row(min_dist_slider),
                                        horz_line,
                                        dbc.Row(min_cluster_size_slider),
                                        dbc.Row(min_samples_slider),
                                        horz_line,
                                        dbc.Row(
                                            [card3DButtons],
                                            justify="center",
                                            align="center",
                                        ),
                                    ],
                                    justify="center",
                                    align="center",
                                ),
                            ],
                            body=True,
                            style={"height": "70vh"},
                        ),
                        md=3,
                        align="start",
                    ),
                    dbc.Col(
                        children=[
                            dbc.Row(
                                children=[
                                    graphWithLoadingAnimation,
                                    dcc.Tooltip(
                                        id="mainGraphTooltip", direction="right"
                                    ),
                                    dcc.Download(id="mainGraphDownload"),
                                ],
                                style={"height": "70vh"},
                            ),
                        ],
                        md=9,
                    ),
                ],
                align="center",
            ),
            horz_line,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.Row(search_preview_main_title),
                                    dbc.Row(
                                        [image_search_title],
                                    ),
                                    horz_line,
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [preview_test_image],
                                                style={"max-height": "15vh"},
                                                md=6,
                                            ),
                                            dbc.Col([image_file_info], md=6),
                                        ]
                                    ),
                                    horz_line,
                                    dbc.Row(
                                        [
                                            dbc.Col(search_preview_results_title),
                                            dbc.Row([search_preview]),
                                        ],
                                        justify="center",
                                        align="center",
                                    ),
                                ],
                                body=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        children=[
                            graph2D,
                            dcc.Tooltip(id="graph2DTooltip", direction="right"),
                            dcc.Download(id="graph2DDownload"),
                            horz_line,
                            dbc.Row(
                                children=(
                                    dbc.Col(
                                        children=[
                                            dbc.Card(
                                                [
                                                    dbc.Row(
                                                        [preview_title],
                                                    ),
                                                    horz_line,
                                                    dbc.Row(
                                                        dbc.Col(imagePreview),
                                                        justify="center",
                                                        align="center",
                                                    ),
                                                ],
                                                body=True,
                                            ),
                                        ],
                                        width="auto",
                                        md=12,
                                    ),
                                ),
                                justify="center",
                                align="center",
                            ),
                            horz_line,
                        ],
                        md=8,
                    ),
                ],
                style={"height": "50vh"},
                justify="center",
                align="start",
            ),
        ],
        fluid=True,
    )


app.layout = serve_layout


################################################################################
""" Callback fxns: """
################################################################################
########################################################################
""" [CALLBACK]: upload zip file and extract features/paths """
########################################################################
@app.callback(
    [
        Output("dataProcessedFlag", "data"),
        Output("fileInfo", "children"),
        Output("graph3DButton", "disabled"),
        Output("upload_image_file_button", "disabled"),
    ],
    [Input("upload-image-folder", "contents")],
    [
        State("upload-image-folder", "filename"),
        State("session-id", "data"),
    ],
)
def uploadData(content, filename, session_id):
    # the content needs to be split. It contains the type and the real content
    if content is not None:
        global redis_client
        content_type, content_str = content.split(",")
        # Create Dataset object, remove the extension part (.zip) from the filename
        dataset_name = filename.split(".")[0]
        # store dataset name in redis
        if not redis_client.sismember(f"{session_id}:datasets", dataset_name):
            redis_client.sadd(f"{session_id}:datasets", dataset_name)
            decode_and_extract_zip(session_id, dataset_name, content_str)
        # Update the current dataset being used by user
        redis_client.set(f"{session_id}:curr_dataset", dataset_name)
        # read uploaded data and create ZipFile obj
        outputText = create_LR_label(
            id="file_info_label",
            leftText="[FILE]:",
            rightText=dataset_name,
        )
        return [True], outputText, False, False
    return no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: upload image file and find similar images """
########################################################################
@app.callback(
    [
        Output("image_file_info", "children"),
        Output("preview_test_image", "children"),
        Output("search_preview", "children"),
        Output("download_search_button", "href"),
    ],
    [Input("upload_image_file_button", "contents")],
    [
        State("upload_image_file_button", "filename"),
        State("session-id", "data"),
    ],
)
def uploadData(content, filename, session_id):
    if content is not None:
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
        # TODO: Why am I using content and not content_str? maybe cuz html img default something?
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
            redis_client, session_id, dataset_name, result_idxs[0], scale=2.5
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
        return output_filename, test_image, result_preview, preview_zip_path
    return no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 3D graph and update cluster download """
########################################################################
@app.callback(
    [
        Output("mainGraph", "figure"),
        Output("dataClusteredFlag", "data"),
        Output("dataInfo", "children"),
        Output("download_clusters_button", "disabled"),
        Output("download_clusters_button", "href"),
    ],
    [Input("graph3DButton", "n_clicks")],
    [
        State("dataProcessedFlag", "data"),
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
        State("session-id", "data"),
    ],
)
def update_output(
    n_clicks,
    dataProcessedFlag,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    session_id,
):
    # After feature extraction, enable 3D graph gen button
    if dataProcessedFlag:
        global redis_client
        # Generate the 3D graph and update global variable
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        figure, percent_clustered = generate_fig_3D(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
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
        )
        dataset_dir = f"assets/{session_id}/{dataset_name}"
        cluster_zip_path = os.path.join(dataset_dir, "clusters.zip")

        return figure, [True], output_text, False, cluster_zip_path
    else:
        return no_update, no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 2D Graph """
########################################################################
@app.callback(
    [Output("graph2D", "figure")],
    [Input("dataClusteredFlag", "data")],
    [
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
        State("session-id", "data"),
    ],
)
def create_graph_2D(
    dataClusteredFlag,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    session_id,
):
    if dataClusteredFlag:
        global redis_client
        # Calculate UMAP embeddings with two components.
        dataset_name = redis_client.get(f"{session_id}:curr_dataset")
        embeddings_2D = setup_umap(
            redis_client,
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            n_components=2,
        )
        labels = get_labels(
            session_id,
            dataset_name,
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
        )
        fig = generate_fig_2D(embeddings_2D, labels)

        return [fig]
    else:
        return no_update


########################################################################
""" [CALLBACK]: 3D hover to show image """
########################################################################
@app.callback(
    [
        Output("mainGraphTooltip", "show"),
        Output("mainGraphTooltip", "bbox"),
        Output("mainGraphTooltip", "children"),
    ],
    [Input("mainGraph", "hoverData")],
    [
        State("dataClusteredFlag", "data"),
        State("session-id", "data"),
    ],
)
def display_hover(hoverData, dataClusteredFlag, session_id):
    if (hoverData is None) or (not dataClusteredFlag):
        return False, no_update, no_update
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    # Load image with pillow
    hover_img_index = hoverData["points"][0]["pointNumber"]
    im_uri = gen_img_uri(redis_client, session_id, dataset_name, hover_img_index)

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    children = [
        html.Img(
            src=im_uri,
            style={"width": "150px"},
        ),
        # html.P("Image from base64 string"),
    ]
    return True, bbox, children


########################################################################
""" [CALLBACK]: 2D Hover preview"""
########################################################################
@app.callback(
    [
        Output("graph2DTooltip", "show"),
        Output("graph2DTooltip", "bbox"),
        Output("graph2DTooltip", "children"),
    ],
    [Input("graph2D", "hoverData")],
    State("session-id", "data"),
)
def display_hover2D(hoverData, session_id):
    if hoverData is None:
        return False, no_update, no_update
    # Load image with pillow
    global redis_client
    dataset_name = redis_client.get(f"{session_id}:curr_dataset")
    hover_img_index = hoverData["points"][0]["pointNumber"]
    im_uri = gen_img_uri(redis_client, session_id, dataset_name, hover_img_index)
    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    children = [
        html.Img(
            src=im_uri,
            style={"width": "150px"},
        ),
        # html.P("Image from base64 string"),
    ]
    return True, bbox, children


########################################################################
""" [CALLBACK]: 3D Click Download"""
########################################################################
@app.callback(
    Output("mainGraphDownload", "data"),
    Input("mainGraph", "clickData"),
    State("session-id", "data"),
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
@app.callback(
    Output("graph2DDownload", "data"),
    Input("graph2D", "clickData"),
    State("session-id", "data"),
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
@app.callback(
    [
        Output("imagePreview", "children"),
        Output("download_preview_button", "disabled"),
        Output("download_preview_button", "href"),
    ],
    [Input("graph2D", "selectedData")],
    State("session-id", "data"),
    prevent_initial_call=True,
)
def display_selected_data(selectedData, session_id):
    """get all the selected point idx from the selectedData dictionary
    then get the img path for each idx, then create previwe with paths"""
    if selectedData["points"]:
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
            gen_img_preview(redis_client, session_id, dataset_name, selected_img_idxs),
            False,
            preview_zip_path,
        )
    else:
        # no points selected
        return no_update, no_update, no_update


if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0", port=5000)

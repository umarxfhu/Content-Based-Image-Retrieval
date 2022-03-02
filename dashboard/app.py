import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State

from dataset import Dataset
from componentBuilder import create_LR_label, gen_img_preview
from figureGen import blankFig, generate_fig_3D, generate_fig_2D

################################################################################
""" Global variables: """
################################################################################

from config import config

dataset_obj = Dataset()


################################################################################
""" Initialize Dash App: """
################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])


################################################################################
""" Dash Components: """
################################################################################

horz_line = html.Hr()

# title of the dash
title = html.H4(
    "Dataset Clustering Dashboard",
    style={"color": "white", "text-align": "center", "padding": "10px"},
)


fileInfo = dcc.Loading(
    id="fileInfo",
    children=[
        html.H6(
            children=["Please upload FolderOfImagesYouWishToCluster.zip"],
            style={"textAlign": "center", "padding": "10px"},
        )
    ],
)

dataInfo = dcc.Loading(
    id="dataInfo",
    children=[
        html.Div(
            children=["Then click the generate graphs button below."],
            style={"textAlign": "Center", "padding": "10px"},
        )
    ],
)

graphWithLoadingAnimation = dcc.Graph(
    id="mainGraph",
    clear_on_unhover=True,
    figure=blankFig(),
    loading_state={"is_loading": True},
)

graph2D = dcc.Graph(
    id="graph2D",
    clear_on_unhover=True,
    figure=blankFig(),
    loading_state={"is_loading": True},
)

downloadClustersButton = dbc.Button(
    children=["Download Clusters"],
    id="downloadClustersButton",
    download="example.jpg",
    n_clicks=0,
    disabled=True,
    external_link=True,
    color="primary",
)

graph3DButton = dbc.Button(
    children=["Generate Graphs"],
    id="graph3DButton",
    n_clicks=0,
    disabled=True,
    color="primary",
)

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

n_neighbors_slider = [
    create_LR_label(
        id="n_neighbors_label", leftText="[UMAP]:", rightText="n_neighbors"
    ),
    dcc.Slider(min=40, max=240, step=40, value=80, id="n_neighbors_slider"),
]

min_dist_slider = [
    create_LR_label(id="min_dist_label", leftText="[UMAP]:", rightText="min_dist"),
    dcc.Slider(min=0.0, max=0.5, step=0.1, value=0.0, id="min_dist_slider"),
]

min_cluster_size_slider = [
    create_LR_label(
        id="min_cluster_size_label", leftText="[HDBSCAN]:", rightText="min_cluster_size"
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

min_samples_slider = [
    create_LR_label(
        id="min_samples_label", leftText="[HDBSCAN]:", rightText="min_samples"
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

# Tooltips for umap sliders
n_neighbors_tooltip = dbc.Tooltip(
    "This parameter controls how UMAP balances local versus global structure in the data. "
    "As n_neighbors is increased UMAP manages to see more of the overall "
    "structure of the data, gluing more components together, and better coverying "
    "the broader structure of the data. As n_neighbors increases further more and "
    "more focus in placed on the overall structure of the data. This results in, "
    "a plot where the overall structure is well captured, but at the loss of some of "
    "the finer local structure (individual images may no longer necessarily be "
    "immediately next to their closest match).",
    target="n_neighbors_label",
)

min_dist_tooltip = dbc.Tooltip(
    "The min_dist parameter controls how tightly UMAP is allowed to pack points "
    "together. It, quite literally, provides the minimum distance apart that points "
    "are allowed to be in the low dimensional representation. This means that low "
    "values of min_dist will result in clumpier embeddings. This can be useful if "
    "you are interested in clustering, or in finer topological structure. Larger "
    "values of min_dist will prevent UMAP from packing points together and will "
    "focus on the preservation of the broad topological structure instead.",
    target="min_dist_label",
)

min_cluster_size_tooltip = dbc.Tooltip(
    "Set it to the smallest size grouping that you wish to consider a cluster. "
    "It can have slightly non-obvious effects however (unless you fix min_samples).",
    target="min_cluster_size_label",
)

min_samples_tooltip = dbc.Tooltip(
    "The simplest intuition for what min_samples does is provide a measure of how "
    "conservative you want you clustering to be. The larger the value of min_samples "
    "you provide, the more conservative the clustering - more points will be "
    "declared as noise, and clusters will be restricted to progressively more "
    "dense areas. Steadily increasing min_samples will, as we saw in the examples "
    "above, make the clustering progressively more conservative.",
    target="min_samples_label",
)

# In browser storage objects
dataProcessedFlagStore = dcc.Store(id="dataProcessedFlag", data=False)
dataClusteredFlagStore = dcc.Store(id="dataClusteredFlag", data=False)


################################################################################
""" Dash UI Layout: """
################################################################################

app.layout = dbc.Container(
    [
        dataProcessedFlagStore,
        dataClusteredFlagStore,
        n_neighbors_tooltip,
        min_dist_tooltip,
        min_cluster_size_tooltip,
        min_samples_tooltip,
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
                                        [
                                            dbc.Col(
                                                downloadClustersButton,
                                                width="auto",
                                            ),
                                            dbc.Col(graph3DButton, width="auto"),
                                        ],
                                        justify="center",
                                        align="center",
                                    ),
                                ],
                                justify="center",
                                align="center",
                            ),
                        ],
                        body=True,
                        style={"height": "80vh"},
                    ),
                    md=3,
                    align="start",
                ),
                dbc.Col(
                    children=[
                        dbc.Row(
                            children=[
                                graphWithLoadingAnimation,
                                dcc.Tooltip(id="mainGraphTooltip", direction="right"),
                                dcc.Download(id="mainGraphDownload"),
                            ],
                            style={"height": "80vh"},
                        ),
                    ],
                    md=9,
                ),
            ],
            align="center",
        ),
        horz_line,
        dbc.Row(
            dbc.Col(
                children=[
                    graph2D,
                    dcc.Tooltip(id="graph2DTooltip", direction="right"),
                    dcc.Download(id="graph2DDownload"),
                    horz_line,
                    dbc.Row(dbc.Card([imagePreview], body=True), align="center"),
                ],
            ),
            style={"height": "50vh"},
            justify="center",
            align="center",
        ),
        horz_line,
    ],
    fluid=True,
)

################################################################################
""" Callback fxns: """
################################################################################
########################################################################
""" [CALLBACK]: upload zip file and extract features/paths """
########################################################################
@app.callback(
    [Output("dataProcessedFlag", "data"), Output("fileInfo", "children")],
    [Input("upload-image-folder", "contents")],
    State("upload-image-folder", "filename"),
)
def uploadData(content, filename):
    # the content needs to be split. It contains the type and the real content
    global dataset_obj
    if content is not None:
        content_type, content_str = content.split(",")
        dataset_obj = dataset_obj.reset_dataset()
        # Create Dataset object, remove the extension part (.zip) from the filename
        dataset_obj.name = filename.split(".")[0]
        # read uploaded data and create ZipFile obj
        dataset_obj.decode_and_extract_zip(content_str)
        dataset_obj.load_features_paths()
        outputText = create_LR_label(
            id="file_info_label",
            leftText="[FILE]:",
            rightText=dataset_obj.name,
        )
        return [True], outputText

    return no_update, no_update


########################################################################
""" [CALLBACK]: Create 3D graph, n_neighbors umap slider """
########################################################################
@app.callback(
    [
        Output("mainGraph", "figure"),
        Output("dataClusteredFlag", "data"),
        Output("dataInfo", "children"),
        Output("graph3DButton", "disabled"),
    ],
    [Input("dataProcessedFlag", "data"), Input("graph3DButton", "n_clicks")],
    [
        State("n_neighbors_slider", "value"),
        State("min_dist_slider", "value"),
        State("min_cluster_size_slider", "value"),
        State("min_samples_slider", "value"),
    ],
)
def update_output(
    dataProcessedFlag, n_clicks, n_neighbors, min_dist, min_cluster_size, min_samples
):
    if dataProcessedFlag:
        # After feature extraction, enable 3D graph gen button
        if n_clicks == 0:
            return no_update, no_update, no_update, False
        else:
            # Generate the 3D graph and update global variable
            figure = generate_fig_3D(
                dataset_obj, n_neighbors, min_dist, min_cluster_size, min_samples
            )
            # Output Clustering statistics
            output_text = create_LR_label(
                id="percentClusteredText",
                leftText="[INFO]:",
                rightText=f"{dataset_obj.calculate_percent_clustered()}% clustered",
            )
            # arrange zip files to create download
            dataset_obj.create_clusters_zip()
            return figure, [True], output_text, no_update
    else:
        return no_update, no_update, no_update, no_update


########################################################################
""" [CALLBACK]: Create 2D Graph """
########################################################################
@app.callback(
    [Output("graph2D", "figure")],
    [Input("dataClusteredFlag", "data")],
    [State("n_neighbors_slider", "value"), State("min_dist_slider", "value")],
)
def create_graph_2D(dataClusteredFlag, n_neighbors_value, min_dist_value):
    print("dataClusteredFlag", dataClusteredFlag)
    if dataClusteredFlag:
        print("got in line 340!")
        # Calculate UMAP embeddings with two components.
        embeddings_2D = dataset_obj.setup_umap(
            n_neighbors_value, min_dist_value, n_components=2
        )
        # Use localinputdata (global) to get precalculated 3D labels
        # Make 2D px scattergl graph and update callback
        fig = generate_fig_2D(embeddings_2D, dataset_obj.labels)

        return [fig]
    else:
        return no_update


########################################################################
""" [CALLBACK]: Download Clusters """
########################################################################
@app.callback(
    [
        Output("downloadClustersButton", "href"),
        Output("downloadClustersButton", "disabled"),
    ],
    [Input("dataClusteredFlag", "data"), Input("downloadClustersButton", "n_clicks")],
)
def create_clusters(dataClusteredFlag, n_clicks):
    if dataClusteredFlag:
        filePath = "clusters.zip"
        # return path to running assets folder, update the download button text,
        # and enable the button to be clicked
        return app.get_asset_url(filePath), False
    else:
        return no_update, no_update


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
    [State("dataClusteredFlag", "data")],
)
def display_hover(hoverData, dataClusteredFlag):
    if (hoverData is None) or (not dataClusteredFlag):
        return False, no_update, no_update
    # Load image with pillow
    hover_img_index = hoverData["points"][0]["pointNumber"]
    im_uri = dataset_obj.gen_img_uri(hover_img_index)
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
)
def display_hover2D(hoverData):
    if hoverData is None:
        return False, no_update, no_update
    # Load image with pillow
    hover_img_index = hoverData["points"][0]["pointNumber"]
    im_uri = dataset_obj.gen_img_uri(hover_img_index)
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
    prevent_initial_call=True,
)
def func(clickData):
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = dataset_obj.img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Click Download"""
########################################################################
@app.callback(
    Output("graph2DDownload", "data"),
    Input("graph2D", "clickData"),
    prevent_initial_call=True,
)
def func(clickData):
    print("entered 2d click fxn")
    clicked_img_idx = clickData["points"][0]["pointNumber"]
    image_path = dataset_obj.img_paths[clicked_img_idx]
    return dcc.send_file(image_path)


########################################################################
""" [CALLBACK]: 2D Lasso to preview"""
########################################################################
@app.callback(
    Output("imagePreview", "children"),
    Input("graph2D", "selectedData"),
    prevent_initial_call=True,
)
def display_selected_data(selectedData):
    """get all the selected point idx from the selectedData dictionary
    then get the img path for each idx, then create previwe with paths"""
    ctx = dash.callback_context

    points = selectedData["points"]
    selected_img_idxs = []

    for i in range(len(points)):
        idx = points[i]["pointNumber"]
        selected_img_idxs.append(idx)

    return gen_img_preview(dataset_obj, selected_img_idxs)


################################################################################
""" __main__: """
################################################################################

if __name__ == "__main__":
    app.run_server(debug=False)

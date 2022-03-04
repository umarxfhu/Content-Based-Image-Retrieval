from dash import html, dcc
from numpy import place
from dataset import Dataset
import dash_bootstrap_components as dbc

# Only certain component building functions are here


def gen_img_preview(dataset_obj: Dataset, selected_img_idxs):
    """Generate the selection preview when the lasso/box tool on 2D graph is used.
    inputs:
        - img_paths: (type: string) Component identifier for callback use
    output:
        - imagesList list of html div components where each contains one html.Img"""
    # Helper function to create html img
    def generate_thumbnail(image_uri):
        return html.Div(
            [
                html.Img(
                    src=image_uri,
                    style={
                        "height": "10%",
                        "width": "10%",
                        "float": "left",
                        "position": "relative",
                        "padding-top": 1,
                        "padding-right": 1,
                    },
                ),
            ]
        )

    # Functionality
    imagesList = []
    for idx in selected_img_idxs:
        image_uri = dataset_obj.gen_img_uri(idx)
        imagesList.append(generate_thumbnail(image_uri))

    return imagesList


def gen_download_button(id: str, children: "list[str]", href: str):
    return dbc.Button(
        children=children,
        id=id,
        download="example.jpg",
        n_clicks=0,
        disabled=True,
        external_link=True,
        color="primary",
        href=href,
    )


def update_preview_button(dataset_obj: Dataset, selected_img_idxs):
    pass


def create_LR_label(id, leftText, rightText, tip_text_left=None, tip_text_right=None):
    """Insert an html.Div with text on one line with left and right
    justified portions.
    inputs:
        - id: (type: string) Component identifier for callback use
        - leftText: (type: string) Left justified text
        - rightText: (type: string) Right justified text
    output:
        - html.Div component object"""

    if tip_text_left:
        left_children = [
            leftText,
            dbc.Tooltip(
                tip_text_left,
                target=f"{id}_left",
                style={
                    "border-radius": "10px",
                    "background": "#282828",
                    "border": "2px solid #FFFFFF",
                },
            ),
        ]
    else:
        left_children = [leftText]

    if tip_text_right:
        right_children = [
            html.H6(
                children=rightText,
                style={
                    "word-wrap": "break-word",
                },
            ),
            dbc.Tooltip(
                tip_text_right,
                target=f"{id}_right",
                style={
                    "border-radius": "10px",
                    "background": "#282828",
                    "border": "2px solid #FFFFFF",
                },
            ),
        ]
    else:
        right_children = [rightText]

    label = html.Div(
        id=id,
        children=[
            html.P(
                id=f"{id}_left",
                children=left_children,
                style={"textAlign": "left", "width": "49%", "display": "inline-block"},
            ),
            html.P(
                id=f"{id}_right",
                children=right_children,
                style={
                    "textAlign": "right",
                    "width": "50%",
                    "display": "inline-block",
                },
            ),
        ],
    )

    return label


def create_title_with_button(left_child, button):
    return html.Div(
        children=[
            html.Div(
                html.H5(
                    children=left_child,
                    className="card-title",
                    style={
                        "textAlign": "left",
                    },
                ),
                style={
                    "textAlign": "left",
                    "width": "47%",
                    "display": "inline-block",
                    "margin-left": "2%",
                },
            ),
            html.Div(
                children=button,
                style={
                    "textAlign": "right",
                    "width": "50%",
                    "display": "inline-block",
                },
            ),
        ],
    )


def create_info_loading(id: str, children: list):
    return dcc.Loading(
        id=id,
        children=[
            html.Div(
                html.H6(
                    children=children,
                    style={
                        "textAlign": "center",
                        "padding": "10px",
                    },
                )
            )
        ],
    )

from datetime import datetime
import os
import uuid
import shutil
from zipfile import ZipFile

import time
from redis import Redis

import threading
from worker import poll_remove_user_data

import flask
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output, State

import dash_uploader as du

# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# app = Dash(__name__, external_stylesheets=external_stylesheets)

server = flask.Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG])


redis_client = Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
redis_client.flushall()

temp_upload_directory = "assets/temp"
# optional argument during configuration: , use_upload_id=False
du.configure_upload(app, temp_upload_directory)

"""warningSecurity note: The Upload component allows POST requests and uploads
of arbitrary files to the server harddisk and one should take this into account
(with user token checking etc.) if used as part of a public website! For this 
you can utilize the http_request_handler argument of the du.configure_upload."""

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

card = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Image Search", tab_id="tab-1"),
                    dbc.Tab(label="Selection Preview", tab_id="tab-2"),
                ],
                id="card-tabs",
                active_tab="tab-1",
            )
        ),
        dbc.CardBody(html.P(id="card-content", className="card-text")),
    ]
)


def serve_layout():
    session_id = str(uuid.uuid4())
    return html.Div(
        [
            # User timestamp updater
            dcc.Interval(
                id="interval-component",
                interval=1 * 20000,  # 20000 milliseconds = 20 sec
                n_intervals=0,
            ),
            # Empty div used since callbacks with no output are not allowed
            html.Div(id="dummy1"),
            dcc.Store(id="session-id", data=session_id),
            html.Div([card]),
            html.Div([f"Hi! I am user#: {session_id}"]),
            html.Div(id="output-image-upload"),
            html.Div(children=[upload_image_file_button]),
        ]
    )


app.layout = serve_layout


@app.callback(
    Output("card-content", "children"),
    [Input("card-tabs", "active_tab")],
    State("session-id", "data"),
)
def tab_content(active_tab, session_id):
    if active_tab == "tab-1":
        upload = (
            html.Div(
                du.Upload(
                    id="dash_uploader",
                    text="Drag and Drop Here to upload!",
                    max_files=1,
                    filetypes=["zip"],
                    default_style={
                        "overflow": "hide",
                        "minHeight": "2vh",
                        "lineHeight": "2vh",
                    },
                    upload_id=session_id,
                ),
                style={  # wrapper div style
                    "textAlign": "center",
                    "width": "100%",
                    # "height": "50px",
                    "padding": "10px",
                    "display": "inline-block",
                },
            ),
        )
        return upload
    else:
        return f"I am the selection preview"


@app.callback(
    [
        Output("output-image-upload", "children"),
    ],
    [Input("dash_uploader", "isCompleted")],
    [
        State("dash_uploader", "fileNames"),
        State("session-id", "data"),
    ],
)
def update_output(isCompleted, fileNames, session_id):
    def move_unzip_uploaded_file(session_id):
        print("i here")
        user_temp_path = os.path.join(temp_upload_directory, session_id, fileNames[0])
        user_unzip_path = os.path.join(f"assets/{session_id}/unzipped")
        if not os.path.exists(user_unzip_path):
            os.makedirs(user_unzip_path)
        shutil.move(user_temp_path, user_unzip_path)
        file_path = os.path.join(user_unzip_path, f"{fileNames[0]}")
        print("file_path:", file_path)
        with ZipFile(file_path, "r") as zipObj:
            zipObj.extractall(user_unzip_path)
        os.remove(file_path)
        shutil.rmtree(os.path.join(temp_upload_directory, session_id))

    if isCompleted:
        move_unzip_uploaded_file(session_id)
        return [f"file donezo: {fileNames[0]}"]
    else:
        return no_update


# @du.callback(
#     output=Output('callback-output', 'children'),
#     id='dash_uploader',
# )
# def get_a_list(filenames):
#     return html.Ul([html.Li(filenames)])


@app.callback(
    Output("dummy1", "children"),
    Input("interval-component", "n_intervals"),
    State("session-id", "data"),
)
def update_activity_timestamp(n, session_id):
    # store and update redis timestamps for each user
    global redis_client
    redis_client.set("latest_timestamp", str(datetime.now()))

    return no_update


def test_redis_poll(redis_client):
    while True:

        time.sleep(20)
        time_then = redis_client.get("latest_timestamp")
        print("time when saved was:", time_then)
        print("time now:", datetime.now())


if __name__ == "__main__":
    print("Starting thread")
    x = threading.Thread(target=test_redis_poll, args=(redis_client,), daemon=True)
    x.start()
    print("Done starting thread")
    # app.run_server(debug=False)
    server.run(debug=True, host="0.0.0.0", port=5000)

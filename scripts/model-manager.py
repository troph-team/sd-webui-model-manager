import requests
import json
import glob
import modules.scripts as scripts
import gradio as gr
from modules import script_callbacks, shared, sd_models
from modules.ui import create_refresh_button
import time
import threading
import urllib.request
import os
from tqdm import tqdm
import re
import pandas as pd
from requests.exceptions import ConnectionError
import psutil

NO_SELECT = "<no select>"

def download_file(url, file_name, progress):
    # Maximum number of retries
    max_retries = 5

    # Delay between retries (in seconds)
    retry_delay = 10

    while True:
        # Check if the file has already been partially downloaded
        if os.path.exists(file_name):
            # Get the size of the downloaded file
            downloaded_size = os.path.getsize(file_name)

            # Set the range of the request to start from the current size of the downloaded file
            headers = {"Range": f"bytes={downloaded_size}-"}
        else:
            downloaded_size = 0
            headers = {}

        # Split filename from included path
        tokens = re.split(re.escape('\\'), file_name)
        file_name_display = tokens[-1]

        # Initialize the progress bar
        progress = progress.tqdm(total=1000000000, unit="B", unit_scale=True, desc=f"Downloading {file_name_display}", initial=downloaded_size, leave=False)

        # Open a local file to save the download
        with open(file_name, "ab") as f:
            while True:
                try:
                    # Send a GET request to the URL and save the response to the local file
                    response = requests.get(url, headers=headers, stream=True)

                    # Get the total size of the file
                    total_size = int(response.headers.get("Content-Length", 0))

                    # Update the total size of the progress bar if the `Content-Length` header is present
                    if total_size == 0:
                        total_size = downloaded_size
                    progress.total = total_size 

                    # Write the response to the local file and update the progress bar
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            progress.update(len(chunk))

                    downloaded_size = os.path.getsize(file_name)
                    # Break out of the loop if the download is successful
                    break
                except ConnectionError as e:
                    # Decrement the number of retries
                    max_retries -= 1

                    # If there are no more retries, raise the exception
                    if max_retries == 0:
                        raise e

                    # Wait for the specified delay before retrying
                    time.sleep(retry_delay)

        # Close the progress bar
        progress.close()
        downloaded_size = os.path.getsize(file_name)
        # Check if the download was successful
        if downloaded_size >= total_size:
            print(f"{file_name_display} successfully downloaded.")
            break
        else:
            print(f"Error: File download failed. Retrying... {file_name_display}")

def download_file_thread(url, file_name, progress=gr.Progress()):
    folder = "models/Stable-diffusion"
    path_to_new_file = os.path.join(folder, file_name)     
    thread = threading.Thread(target=download_file, args=(url, path_to_new_file, progress))
    # Start the thread
    thread.start()

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.3f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def get_files_size(path: list|str):
    inputs = path if type(path) is list else [path]
    return sum([os.path.getsize(path) for path in inputs])

# -- states --

local_models = []
is_delete_button_show = False

def reload_models():
    global local_models
    model_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
    exts = ["*.ckpt", "*.pt", "*.safetensors"]
    models = sum([glob.glob(os.path.join(model_dir, "**", ext), recursive=True) for ext in exts], [])
    local_models = [
        {"name": os.path.relpath(m, model_dir), "path": m}
        for m in models
    ]

def add_model(path):
    global local_models
    if [m for m in local_models if m["path"] == path]:
        return
    model_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
    local_models.append({"name": os.path.realpath(path, model_dir), "path": path})

def remove_model(path):
    global local_models
    local_models = [m for m in local_models if m["path"] != path]

def get_model_by_name(name):
    return next(iter([m for m in local_models if m["name"] == name]), None)

# -- updates --

def get_all_models():
    global local_models
    reload_models()
    # disk usage
    hdd = psutil.disk_usage('/')
    model_total_size = get_files_size([m["path"] for m in local_models])
    return (
        gr.Dropdown.update(choices=[NO_SELECT] + [m["name"] for m in local_models], value=NO_SELECT),
        gr.HTML.update(value=f"""
            <p>Total Model Size: {sizeof_fmt(model_total_size)}</p>
            <p>Disk Free: {sizeof_fmt(hdd.free)} / {sizeof_fmt(hdd.total)}</p>
        """),
        gr.Button.update(visible=False),
        gr.Button.update(visible=False)
    )

def select_model(model_name):
    model = get_model_by_name(model_name)
    global is_delete_button_show
    is_delete_button_show = False
    if model:
        model_path = model["path"]
        model_size = get_files_size(model_path)
        return gr.HTML.update(value=f"""
            <p><b>{model_name}</b></p>
            <p>{model_path}</p>
            <p>{sizeof_fmt(model_size)}</p>
        """), gr.Button.update(value=f"Delete {model_name}", visible=True), gr.Button.update(visible=False)
    else:
        return gr.HTML.update(value=""), gr.Button.update(value="", visible=False), gr.Button.update(visible=False)

def switch_delete_btn(model_name):
    global is_delete_button_show
    if model_name == NO_SELECT: return
    is_delete_button_show = not is_delete_button_show
    if is_delete_button_show:
        return gr.Button.update(value=f"Cancel Delete {model_name}"), gr.Button.update(visible=True)
    else:
        return gr.Button.update(value=f"Delete {model_name}"), gr.Button.update(visible=False)

def delete_model_from_disk(model_name):
    if model_name == NO_SELECT: return
    model_path = next(iter([m["path"] for m in local_models if m["name"] == model_name]), None)
    if model_path:
        os.remove(model_path)
    remove_model(model_path)
    # refresh webui model list
    sd_models.list_models()
    return *get_all_models(), *switch_delete_btn(model_name), gr.HTML.update(value=""), gr.Button.update(visible=False)

# -- ui --

def on_ui_tabs():
    with gr.Blocks() as manager_interface:
        with gr.Row():
            with gr.Column():
                gr.HTML(value="<h1>Manage Models</h1>")
                with gr.Row():
                    reload = gr.Button(value='\U0001f504', variant='tool')
                    disk_usage = gr.HTML(label="Disk Usage", value="")
                model_dropdown = gr.Dropdown(label="Model List", choices=[], interactive=True, value=None)
                model_info = gr.HTML(label="Model Description", value="")
                delete_model_guard = gr.Button(value="Delete Model")
                delete_model = gr.Button(value="Comfirm Delete Model", variant="primary", visible=False)
            with gr.Column():
                gr.HTML(value="<h1>Download Model</h1>")
                # model_type = gr.Radio(label='Model Type', choices=["Checkpoint","Hypernetwork","TextualInversion","AestheticGradient","VAE"], value="Checkpoint", type="value")
                model_url = gr.Textbox(label='Model URL', value="", interactive=True, lines=1)
                download_model = gr.Button(value="Download")

        reload.click(
            fn=get_all_models,
            inputs=[],
            outputs=[model_dropdown, disk_usage, delete_model_guard, delete_model]
        )

        model_dropdown.change(
            fn=select_model,
            inputs=[model_dropdown],
            outputs=[model_info, delete_model_guard, delete_model]
        )

        delete_model_guard.click(
            fn=switch_delete_btn,
            inputs=[model_dropdown],
            outputs=[delete_model_guard, delete_model]
        )

        delete_model.click(
            fn=delete_model_from_disk,
            inputs=[model_dropdown],
            outputs=[model_dropdown, disk_usage, delete_model_guard, delete_model, model_info, delete_model_guard]
        )
        # download_model.click(
        #     fn=download_file_thread,
        #     inputs=[
        #     model_url,
        #     model_filename,
        #     model_type,
        #     save_model_in_new,
        #     list_models,
        #     ],
        #     outputs=[]
        # )

        manager_interface.load(
            fn=get_all_models,
            inputs=[],
            outputs=[model_dropdown, disk_usage]
        )
    
    return (manager_interface, "Model Manager", "model_manager"),

script_callbacks.on_ui_tabs(on_ui_tabs)
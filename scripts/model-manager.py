import requests
import glob
import modules.scripts as scripts
import modules.shared as shared
import gradio as gr
from modules import script_callbacks, sd_models
from modules.call_queue import wrap_gradio_gpu_call
import time
import os
from tqdm import tqdm
import re
from requests.exceptions import ConnectionError
import psutil

NO_SELECT = "<no select>"

def get_file_name_from_url(url):
    r = requests.get(url, headers={"Range": f"bytes=0-10"})
    print(r.status_code)
    if r.status_code >= 400:
        raise Exception("Resource not found.")
    d = r.headers['content-disposition']
    fname = next(iter(re.findall("filename=(.+)", d)), None)
    if not fname:
        raise Exception("File name not found.")
    if '"' in fname:
        fname = next(iter(re.findall("\"([^\"]+)\"", fname)), None)
    return fname

def download_file(model_type, url):
    shared.state.begin()
    shared.state.job = 'model-download'

    def fail(message):
        shared.state.textinfo = message
        shared.state.end()
        return message

    # Maximum number of retries
    max_retries = 5

    # Delay between retries (in seconds)
    retry_delay = 10

    shared.state.job_count = 1
    shared.state.textinfo = "Loading Model File Name..."
    # Get file name
    try:
        file_name = get_file_name_from_url(url)
    except Exception as e:
        return fail("Get model file name failed.")
    model_dir = get_model_dir(model_type)
    if not model_dir:
        return fail("Invalid model type")
    shared.state.textinfo = f"Get model name: {file_name}"
    file_name = os.path.join(model_dir, file_name)

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
        progress = tqdm(total=1000000000, unit="B", unit_scale=True, desc=f"Downloading {file_name_display}", initial=downloaded_size, leave=False)
        shared.state.textinfo = f"Downloading {file_name_display}..."
        shared.state.sampling_steps = 100
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
                    shared.state.textinfo = f"Downloading {file_name_display}... ({sizeof_fmt(progress.n)} / {sizeof_fmt(progress.total)})"
                    shared.state.sampling_step = int((progress.n / progress.total) * 100)

                    # Write the response to the local file and update the progress bar
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            progress.update(len(chunk))
                            shared.state.textinfo = f"Downloading {file_name_display}... ({sizeof_fmt(progress.n)} / {sizeof_fmt(progress.total)})"
                            shared.state.sampling_step = int((progress.n / progress.total) * 100)

                    downloaded_size = os.path.getsize(file_name)

                    # Break out of the loop if the download is successful
                    break
                except ConnectionError as e:
                    # Decrement the number of retries
                    max_retries -= 1

                    # If there are no more retries, raise the exception
                    if max_retries == 0:
                        return fail("Download failed, reach max try count.")

                    # Wait for the specified delay before retrying
                    shared.state.textinfo = f"Download failed, retrying..."
                    time.sleep(retry_delay)
                    

        # Close the progress bar
        progress.close()
        downloaded_size = os.path.getsize(file_name)
        # Check if the download was successful
        if downloaded_size >= total_size:
            print(f"{file_name_display} successfully downloaded.")
            shared.state.nextjob()
            shared.state.textinfo = "Model downloaded."
            shared.state.end()
            return f"{file_name_display} successfully downloaded."
        else:
            print(f"Error: File download failed. Retrying... {file_name_display}")
            shared.state.textinfo = f"Error: File download failed. Retrying..."

def download_file_thread(id_task, model_type, url):
    try:
        return [gr.update(), download_file(model_type, url)]
    except Exception as e:
        return [gr.update(), str(e)]
    # thread = threading.Thread(target=download_file, args=(url,download_event))
    # thread.start()

# -- utils --

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

def get_model_dir(model_type = None):
    model_dir = None
    if model_type is None:
        model_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
    elif model_type == 'Checkpoint':
        model_dir = "models/Stable-diffusion"
    elif model_type == 'Hypernetwork':
        model_dir = "models/hypernetworks"
    elif model_type == 'TextualInversion':
        model_dir = "embeddings"
    elif model_type == 'AestheticGradient':
        model_dir = "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings"
    elif model_type == 'VAE':
        model_dir = 'models/VAE'
    elif model_type == 'LoRA':
        model_dir = "extensions/sd-webui-additional-networks/models/lora"
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def reload_models():
    global local_models
    model_dir = get_model_dir()
    exts = ["*.ckpt", "*.pt", "*.safetensors"]
    models = sum([glob.glob(os.path.join(model_dir, "**", ext), recursive=True) for ext in exts], [])
    local_models = [
        {"name": os.path.relpath(m, model_dir), "path": m}
        for m in models
    ]


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
            <p><b>{model_name}</b></p>s
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
        return [gr.Button.update(value=f"Cancel Delete {model_name}"), gr.Button.update(visible=True)]
    else:
        return [gr.Button.update(value=f"Delete {model_name}"), gr.Button.update(visible=False)]

def delete_model_from_disk(model_name):
    global is_delete_button_show
    if model_name == NO_SELECT: return
    model_path = next(iter([m["path"] for m in local_models if m["name"] == model_name]), None)
    if model_path:
        os.remove(model_path)
    is_delete_button_show = False
    # refresh webui model list
    sd_models.list_models()
    return [*get_all_models(), gr.update(visible=False), gr.update(visible=False), gr.HTML.update(value="")]

# -- ui --

def on_ui_tabs():
    with gr.Blocks() as manager_interface:
        dummy_component = gr.Label(visible=False)
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
                model_type = gr.Radio(
                    label='Model Type',
                    choices=["Checkpoint", "Hypernetwork", "TextualInversion", "AestheticGradient", "VAE", "LoRA"],
                    value="Checkpoint",
                    type="value",
                    interactive=True
                )
                model_url = gr.Textbox(label='Model URL', value="", interactive=True, lines=1)
                download_model = gr.Button(value="Download", variant="primary")
                with gr.Group(elem_id="model_manager_download_panel"):
                    download_result = gr.HTML(elem_id="model_manager_download_result", show_label=False)

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
            outputs=[
                model_dropdown,
                disk_usage,
                delete_model_guard,
                delete_model,
                model_info,
            ]
        )
        download_model.click(fn=lambda: '', inputs=[], outputs=[download_result])
        download_model.click(
            # the gpu version of method will start a task that make progress bar alive
            fn=wrap_gradio_gpu_call(download_file_thread, extra_outputs=lambda: [gr.update()]),
            _js='model_manager_download',
            inputs=[dummy_component, model_type, model_url],
            # why need a dummy output? bugfix: https://github.com/gradio-app/gradio/issues/2815
            outputs=[dummy_component, download_result]
        )

        manager_interface.load(
            fn=get_all_models,
            inputs=[],
            outputs=[model_dropdown, disk_usage]
        )

    return (manager_interface, "Model Manager", "model_manager"),

script_callbacks.on_ui_tabs(on_ui_tabs)
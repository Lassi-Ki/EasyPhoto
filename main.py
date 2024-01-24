import argparse
import time
import sagemaker_serve
from easyphoto.easyphoto_ui import on_ui_tabs
from easyphoto.easyphoto_utils import reload_javascript
from easyphoto.api import easyphoto_infer_forward_api, easyphoto_train_forward_api

parser = argparse.ArgumentParser()

parser.add_argument("--port", type=int, help="launch gradio with given server port, defaults to 7860 if available", default=None)
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--inference', action='store_true', default=False)
# TODO: 确定是否要区分训练和推理
parser.add_argument('--sd-models-s3uri', default='', type=str, help='SD Models S3Uri')
parser.add_argument('--lora-models-s3uri', default='', type=str, help='Lora Models S3Uri')
parser.add_argument('--embeddings-s3uri', default='', type=str, help='Embedding S3Uri')
parser.add_argument('--hypernetwork-s3uri', default='', type=str, help='Hypernetwork S3Uri')
parser.add_argument('--region-name', default='us-west-2', type=str, help='Region name')
parser.add_argument('--bucket', default='sagemaker-us-west-2-011299426194', type=str)

cmd_opts = parser.parse_args()

# load javascript
reload_javascript()

# create ui
easyphoto = on_ui_tabs()

# launch gradio
app, _, _ = easyphoto.queue(status_update_rate=1).launch(
    server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
    server_port=cmd_opts.port,
    share=cmd_opts.share,
    prevent_thread_lock=True
)

if __name__ == "__main__":
    if cmd_opts.train:
        payload = {}
        sagemaker_serve.train(payload)
    elif cmd_opts.inference:
        easyphoto_infer_forward_api(None, app)
    else:
        pass

    # not close the python
    while True:
        time.sleep(5)
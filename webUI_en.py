import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from itertools import chain
from pathlib import Path

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"


def upload_mix_append_file(files, sfiles):
    try:
        if sfiles is None:
            file_paths = [file.name for file in files]
        else:
            file_paths = [file.name for file in chain(files, sfiles)]
        p = {file: 100 for file in file_paths}
        return file_paths, mix_model_output1.update(value=json.dumps(p, indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def mix_submit_click(js, mode):
    try:
        assert js.lstrip() != ""
        modes = {"Convex Combination": 0, "Linear Combination": 1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path, mix_rate = zip(*data)
        path = mix_model(model_path, mix_rate, mode)
        return f"Success, mixed model saved to {path}"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def updata_mix_info(files):
    try:
        if files is None:
            return mix_model_output1.update(value="")
        p = {file.name: 100 for file in files}
        return mix_model_output1.update(value=json.dumps(p, indent=2))
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def modelAnalysis(
    model_path,
    config_path,
    cluster_model_path,
    device,
    enhance,
    diff_model_path,
    diff_config_path,
    only_diffusion,
    use_spk_mix,
    local_model_enabled,
    local_model_selection,
):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device
        cluster_filepath = (
            os.path.split(cluster_model_path.name)
            if cluster_model_path is not None
            else "no_cluster"
        )
        # get model and config path
        if local_model_enabled:
            # local path
            model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
            config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
        else:
            # upload from webpage
            model_path = model_path.name
            config_path = config_path.name
        fr = ".pkl" in cluster_filepath[1]
        model = Svc(
            model_path,
            config_path,
            device=device if device != "Auto" else None,
            cluster_model_path=cluster_model_path.name
            if cluster_model_path is not None
            else "",
            nsf_hifigan_enhance=enhance,
            diffusion_model_path=diff_model_path.name
            if diff_model_path is not None
            else "",
            diffusion_config_path=diff_config_path.name
            if diff_config_path is not None
            else "",
            shallow_diffusion=True if diff_model_path is not None else False,
            only_diffusion=only_diffusion,
            spk_mix_enable=use_spk_mix,
            feature_retrieval=fr,
        )
        #spks = list(model.spk2id.keys())
        spks = list(model.spk2id.keys()) if model.spk2id else ["default"]
        device_name = (
            torch.cuda.get_device_properties(model.dev).name
            if "cuda" in str(model.dev)
            else str(model.dev)
        )
        msg = f"Successfully loaded model on device {device_name}\n"
        if cluster_model_path is None:
            msg += "No cluster model or feature retrieval model loaded\n"
        elif fr:
            msg += f"Feature retrieval model {cluster_filepath[1]} loaded successfully\n"
        else:
            msg += f"Cluster model {cluster_filepath[1]} loaded successfully\n"
        if diff_model_path is None:
            msg += "No diffusion model loaded\n"
        else:
            msg += f"Diffusion model {diff_model_path.name} loaded successfully\n"
        msg += "Available speakers in the current model:\n"
        for i in spks:
            msg += i + " "
        
        #return sid.update(choices=spks, value=spks[0]), msg
        return gr.Dropdown(choices=spks, value=spks[0]), msg

    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def modelUnload():
    global model
    if model is None:
        return sid.update(choices=[], value=""), "No model to unload!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices=[], value=""), "Model unloaded!"


def vc_infer(
    output_format,
    sid,
    audio_path,
    truncated_basename,
    vc_transform,
    auto_f0,
    cluster_ratio,
    slice_db,
    noise_scale,
    pad_seconds,
    cl_num,
    lg_num,
    lgr_num,
    f0_predictor,
    enhancer_adaptive_key,
    cr_threshold,
    k_step,
    use_spk_mix,
    second_encoding,
    loudness_envelope_adjustment,
):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment,
    )
    model.clear_empty()
    # Build the save path and store in the results folder
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"

    output_file_name = (
        'result_'
        + truncated_basename
        + f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    )
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file


def vc_fn(
    sid,
    input_audio,
    output_format,
    vc_transform,
    auto_f0,
    cluster_ratio,
    slice_db,
    noise_scale,
    pad_seconds,
    cl_num,
    lg_num,
    lgr_num,
    f0_predictor,
    enhancer_adaptive_key,
    cr_threshold,
    k_step,
    use_spk_mix,
    second_encoding,
    loudness_envelope_adjustment,
):
    global model
    try:
        if input_audio is None:
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        audio, sampling_rate = soundfile.read(input_audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # For unknown reasons, the filepath uploaded by Gradio has a strange fixed suffix; remove it here
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(
            output_format,
            sid,
            processed_audio,
            truncated_basename,
            vc_transform,
            auto_f0,
            cluster_ratio,
            slice_db,
            noise_scale,
            pad_seconds,
            cl_num,
            lg_num,
            lgr_num,
            f0_predictor,
            enhancer_adaptive_key,
            cr_threshold,
            k_step,
            use_spk_mix,
            second_encoding,
            loudness_envelope_adjustment,
        )

        return "Success", output_file
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)


def vc_fn2(
    _text,
    _lang,
    _gender,
    _rate,
    _volume,
    sid,
    output_format,
    vc_transform,
    auto_f0,
    cluster_ratio,
    slice_db,
    noise_scale,
    pad_seconds,
    cl_num,
    lg_num,
    lgr_num,
    f0_predictor,
    enhancer_adaptive_key,
    cr_threshold,
    k_step,
    use_spk_mix,
    second_encoding,
    loudness_envelope_adjustment,
):
    global model
    try:
        if model is None:
            return "You need to upload an model", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
        _rate = f"+{int(_rate * 100)}%" if _rate >= 0 else f"{int(_rate * 100)}%"
        _volume = (
            f"+{int(_volume * 100)}%"
            if _volume >= 0
            else f"{int(_volume * 100)}%"
        )
        if _lang == "Auto":
            # _gender is already "Male" or "Female" from the UI
            subprocess.run(
                [sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender]
            )
        else:
            subprocess.run(
                [sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume]
            )
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype="PCM_16")
        input_audio = "tts.wav"
        output_file_path = vc_infer(
            output_format,
            sid,
            input_audio,
            "tts",
            vc_transform,
            auto_f0,
            cluster_ratio,
            slice_db,
            noise_scale,
            pad_seconds,
            cl_num,
            lg_num,
            lgr_num,
            f0_predictor,
            enhancer_adaptive_key,
            cr_threshold,
            k_step,
            use_spk_mix,
            second_encoding,
            loudness_envelope_adjustment,
        )
        os.remove("tts.wav")
        return "Success", output_file_path
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def model_compression(_model):
    if _model == "":
        return "Please select a model to compress first"
    else:
        model_path = os.path.split(_model.name)
        filename, extension = os.path.splitext(model_path[1])
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(os.getcwd(), output_model_name)
        removeOptimizer(_model.name, output_path)
        return f"Compressed model has been saved to {output_path}"


def scan_local_models():
    res = []
    candidates = glob.glob(
        os.path.join(local_model_root, '**', '*.json'), recursive=True
    )
    candidates = set([os.path.dirname(c) for c in candidates])
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if len(jsons) == 1 and len(pths) == 1:
            # must contain exactly one json and one pth file
            res.append(candidate)
    return res


def local_model_refresh_fn():
    choices = scan_local_models()
    return gr.Dropdown.update(choices=choices)


def debug_change():
    global debug
    debug = debug_button.value


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=['JetBrains mono', "Consolas", 'Courier New'],
    ),
) as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.Markdown(
                value="""
                So-VITS-SVC 4.0 Inference WebUI
                """
            )
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(
                        value="""
                        <font size=2> Model Settings</font>
                        """
                    )
                    with gr.Tabs():
                        # invisible checkbox that tracks tab status
                        local_model_enabled = gr.Checkbox(value=False, visible=False)
                        with gr.TabItem('Upload') as local_model_tab_upload:
                            with gr.Row():
                                model_path = gr.File(label="Select Model File")
                                config_path = gr.File(label="Select Config File")
                        with gr.TabItem('Local') as local_model_tab_local:
                            gr.Markdown(
                                f'Model folders should be placed under {local_model_root}'
                            )
                            local_model_refresh_btn = gr.Button(
                                'Refresh Local Model List'
                            )
                            local_model_selection = gr.Dropdown(
                                label='Select Model Folder',
                                choices=[],
                                interactive=True,
                            )
                    with gr.Row():
                        diff_model_path = gr.File(
                            label="Select Diffusion Model File"
                        )
                        diff_config_path = gr.File(
                            label="Select Diffusion Config File"
                        )
                    cluster_model_path = gr.File(
                        label="Select Cluster/Feature Retrieval File (optional)"
                    )
                    device = gr.Dropdown(
                        label="Inference Device (default: auto-select CPU/GPU)",
                        choices=["Auto", *cuda.keys(), "cpu"],
                        value="Auto",
                    )
                    enhance = gr.Checkbox(
                        label=(
                            "Use NSF_HIFIGAN enhancement. This may improve models trained "
                            "on small datasets, but can degrade well-trained models. "
                            "Disabled by default."
                        ),
                        value=False,
                    )
                    only_diffusion = gr.Checkbox(
                        label=(
                            "Use diffusion-only inference. When enabled, the So-VITS "
                            "model is not used; only the diffusion model is used for "
                            "full diffusion inference. Disabled by default."
                        ),
                        value=False,
                    )
                with gr.Column():
                    gr.Markdown(
                        value="""
                        <font size=3>
                        After selecting all files on the left (all file modules show "download"),
                        click "Load Model" to parse and load:
                        </font>
                        """
                    )
                    model_load_button = gr.Button(value="Load Model", variant="primary")
                    model_unload_button = gr.Button(
                        value="Unload Model", variant="primary"
                    )
                    sid = gr.Dropdown(label="Speaker")
                    sid_output = gr.Textbox(label="Output Message")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(
                        value="""
                        <font size=2> Inference Settings</font>
                        """
                    )
                    auto_f0 = gr.Checkbox(
                        label=(
                            "Automatic F0 prediction. Works better with a cluster/feature "
                            "retrieval model. Disables pitch shifting. For singing "
                            "conversion this often causes extreme off-pitch results."
                        ),
                        value=False,
                    )
                    f0_predictor = gr.Dropdown(
                        label=(
                            "Select F0 predictor (crepe, pm, dio, harvest, rmvpe). "
                            "Default: pm (note: crepe uses a mean filter on raw F0)."
                        ),
                        choices=["pm", "dio", "harvest", "crepe", "rmvpe"],
                        value="pm",
                    )
                    vc_transform = gr.Number(
                        label=(
                            "Pitch shift (integer semitones, can be positive or negative; "
                            "+12 is one octave up)"
                        ),
                        value=0,
                    )
                    cluster_ratio = gr.Number(
                        label=(
                            "Cluster/feature retrieval mix ratio (0–1). 0 disables cluster/"
                            "feature retrieval. Using it can improve timbre similarity but "
                            "may worsen articulation (0.5 is often a good choice)."
                        ),
                        value=0,
                    )
                    slice_db = gr.Number(label="Slice threshold (dB)", value=-40)
                    output_format = gr.Radio(
                        label="Output audio format",
                        choices=["wav", "flac", "mp3"],
                        value="wav",
                    )
                    noise_scale = gr.Number(
                        label=(
                            "noise_scale (recommended not to change; affects audio quality; "
                            "somewhat 'mystical' parameter)"
                        ),
                        value=0.4,
                    )
                    k_step = gr.Slider(
                        label=(
                            "Shallow diffusion steps (only effective when a diffusion model "
                            "is used). More steps approach the pure diffusion result."
                        ),
                        value=100,
                        minimum=1,
                        maximum=1000,
                    )
                with gr.Column():
                    pad_seconds = gr.Number(
                        label=(
                            "Pad seconds added to start/end of audio. Due to unknown "
                            "reasons, artifacts can appear at boundaries; padding with "
                            "short silence removes them."
                        ),
                        value=0.5,
                    )
                    cl_num = gr.Number(
                        label=(
                            "Automatic audio slicing length (seconds). 0 disables slicing."
                        ),
                        value=0,
                    )
                    lg_num = gr.Number(
                        label=(
                            "Crossfade length at slice boundaries (seconds). Increase if "
                            "sliced audio is discontinuous. 0 is recommended when already "
                            "smooth. Larger values slow inference."
                        ),
                        value=0,
                    )
                    lgr_num = gr.Number(
                        label=(
                            "Proportion of each slice to keep around the crossfade "
                            "region after discarding heads/tails. Range 0–1 (left-open, "
                            "right-closed)."
                        ),
                        value=0.75,
                    )
                    enhancer_adaptive_key = gr.Number(
                        label=(
                            "Make the enhancer adapt to higher pitch ranges "
                            "(semitones; default 0)"
                        ),
                        value=0,
                    )
                    cr_threshold = gr.Number(
                        label=(
                            "F0 filter threshold (only effective when using crepe). "
                            "Range 0–1. Lower values reduce off-pitch artifacts but "
                            "increase muted segments."
                        ),
                        value=0.05,
                    )
                    loudness_envelope_adjustment = gr.Number(
                        label=(
                            "Mix ratio between source loudness envelope and output "
                            "loudness envelope. Values closer to 1 favor the output "
                            "envelope."
                        ),
                        value=0,
                    )
                    second_encoding = gr.Checkbox(
                        label=(
                            "Second-pass encoding. Performs a second encoding of the "
                            "original audio before shallow diffusion. Behavior is "
                            "somewhat unpredictable; disabled by default."
                        ),
                        value=False,
                    )
                    use_spk_mix = gr.Checkbox(
                        label="Dynamic speaker mixing",
                        value=False,
                        interactive=False,
                    )
            with gr.Tabs():
                with gr.TabItem("Audio to Audio"):
                    vc_input3 = gr.Audio(label="Select audio", type="filepath")
                    vc_submit = gr.Button("Convert Audio", variant="primary")
                with gr.TabItem("Text to Audio"):
                    text2tts = gr.Textbox(
                        label=(
                            "Enter text to synthesize. For best results, enable F0 "
                            "prediction; otherwise the result may sound strange."
                        )
                    )
                    with gr.Row():
                        tts_gender = gr.Radio(
                            label="Speaker Gender",
                            choices=["Male", "Female"],
                            value="Male",
                        )
                        tts_lang = gr.Dropdown(
                            label=(
                                "Select language. 'Auto' will detect based on the input text."
                            ),
                            choices=SUPPORTED_LANGUAGES,
                            value="Auto",
                        )
                        tts_rate = gr.Slider(
                            label="TTS speed (relative multiplier)",
                            minimum=-1,
                            maximum=3,
                            value=0,
                            step=0.1,
                        )
                        tts_volume = gr.Slider(
                            label="TTS volume (relative value)",
                            minimum=-1,
                            maximum=1.5,
                            value=0,
                            step=0.1,
                        )
                    vc_submit2 = gr.Button("Convert Text", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="Output Message")
                with gr.Column():
                    vc_output2 = gr.Audio(
                        label="Output Audio",
                        interactive=False,
                    )

        with gr.TabItem("Tools / Experimental"):
            gr.Markdown(
                value="""
                        <font size=2> So-VITS-SVC 4.0 Tools / Experimental Features</font>
                        """
            )
            with gr.Tabs():
                with gr.TabItem("Static Voice Blending"):
                    gr.Markdown(
                        value="""
                        <font size=2>
                        Description: This feature can merge multiple voice models into a
                        single model (a convex or linear combination of their parameters),
                        creating new voices that do not exist in reality.

                        Notes:
                        1. This feature only supports single-speaker models.
                        2. If you force multi-speaker models, make sure they all have the
                           same number of speakers so that the same speaker IDs align.
                        3. Ensure that the `model` field in `config.json` is identical
                           across all models to be mixed.
                        4. The mixed model can use any one of the original models'
                           `config.json`, but cluster models cannot be used with it.
                        5. For batch uploading, it is best to put all models to be mixed
                           into a single folder and upload that folder.
                        6. Mixing ratios are recommended to be in the 0–100 range, but
                           other values are allowed. In linear combination mode this may
                           lead to unpredictable results.
                        7. After mixing, the result will be saved in the project root
                           directory as `output.pth`.
                        8. In convex combination mode, the mixing ratios are passed
                           through a softmax so they sum to 1, while in linear
                           combination mode they are not normalized.
                        </font>
                        """
                    )
                    mix_model_path = gr.Files(
                        label="Select model files to mix"
                    )
                    mix_model_upload_button = gr.UploadButton(
                        "Select/append model files to mix",
                        file_count="multiple",
                    )
                    mix_model_output1 = gr.Textbox(
                        label="Mixing ratios (percent, %)",
                        interactive=True,
                    )
                    mix_mode = gr.Radio(
                        choices=["Convex Combination", "Linear Combination"],
                        label="Mixing Mode",
                        value="Convex Combination",
                        interactive=True,
                    )
                    mix_submit = gr.Button("Start Voice Blending", variant="primary")
                    mix_model_output2 = gr.Textbox(
                        label="Output Message",
                    )
                    mix_model_path.change(
                        updata_mix_info,
                        [mix_model_path],
                        [mix_model_output1],
                    )
                    mix_model_upload_button.upload(
                        upload_mix_append_file,
                        [mix_model_upload_button, mix_model_path],
                        [mix_model_path, mix_model_output1],
                    )
                    mix_submit.click(
                        mix_submit_click,
                        [mix_model_output1, mix_mode],
                        [mix_model_output2],
                    )

                with gr.TabItem("Model Compression Tool"):
                    gr.Markdown(
                        value="""
                        This tool compresses the model size. It can reduce a ~600 MB
                        So-VITS model to about ~200 MB **without affecting inference
                        quality**, greatly reducing disk usage.

                        **Warning: compressed models cannot be further trained. Only
                        compress models you consider final.**
                    """
                    )
                    model_to_compress = gr.File(label="Upload Model")
                    compress_model_btn = gr.Button(
                        "Compress Model",
                        variant="primary",
                    )
                    compress_model_output = gr.Textbox(
                        label="Output Message",
                        value="",
                    )

                    compress_model_btn.click(
                        model_compression,
                        [model_to_compress],
                        [compress_model_output],
                    )

    with gr.Tabs():
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(
                    value="""
                    <font size=2> WebUI Settings</font>
                    """
                )
                debug_button = gr.Checkbox(
                    label=(
                        "Debug mode. Enable this when reporting issues; detailed "
                        "error messages will be printed in the console."
                    ),
                    value=debug,
                )
        # refresh local model list
        local_model_refresh_btn.click(
            local_model_refresh_fn,
            outputs=local_model_selection,
        )
        # set local enabled/disabled on tab switch
        local_model_tab_upload.select(
            lambda: False,
            outputs=local_model_enabled,
        )
        local_model_tab_local.select(
            lambda: True,
            outputs=local_model_enabled,
        )

        vc_submit.click(
            vc_fn,
            [
                sid,
                vc_input3,
                output_format,
                vc_transform,
                auto_f0,
                cluster_ratio,
                slice_db,
                noise_scale,
                pad_seconds,
                cl_num,
                lg_num,
                lgr_num,
                f0_predictor,
                enhancer_adaptive_key,
                cr_threshold,
                k_step,
                use_spk_mix,
                second_encoding,
                loudness_envelope_adjustment,
            ],
            [vc_output1, vc_output2],
        )
        vc_submit2.click(
            vc_fn2,
            [
                text2tts,
                tts_lang,
                tts_gender,
                tts_rate,
                tts_volume,
                sid,
                output_format,
                vc_transform,
                auto_f0,
                cluster_ratio,
                slice_db,
                noise_scale,
                pad_seconds,
                cl_num,
                lg_num,
                lgr_num,
                f0_predictor,
                enhancer_adaptive_key,
                cr_threshold,
                k_step,
                use_spk_mix,
                second_encoding,
                loudness_envelope_adjustment,
            ],
            [vc_output1, vc_output2],
        )

        debug_button.change(debug_change, [], [])
        model_load_button.click(
            modelAnalysis,
            [
                model_path,
                config_path,
                cluster_model_path,
                device,
                enhance,
                diff_model_path,
                diff_config_path,
                only_diffusion,
                use_spk_mix,
                local_model_enabled,
                local_model_selection,
            ],
            [sid, sid_output],
        )
        model_unload_button.click(
            modelUnload,
            [],
            [sid, sid_output],
        )
    os.system("start http://127.0.0.1:7860")
    app.launch()

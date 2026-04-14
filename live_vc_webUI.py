import glob
import logging
import os
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile
import torch

from inference.infer_tool import Svc
from live_vc import LiveEngine, SvcFilter, get_live_logs, clear_live_logs

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)

model = None
live_filter = None
live_engine = None
cuda = {}

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"


def _query_devices():
    """Return raw sounddevice device list or empty on failure."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        return sd.query_devices()
    except Exception:
        return []


def list_input_devices():
    """
    Devices usable as input (max_input_channels > 0),
    labeled with channel counts to avoid incompatible choices.
    """
    devices = _query_devices()
    res = []
    for idx, d in enumerate(devices):
        max_in = d.get("max_input_channels", 0)
        max_out = d.get("max_output_channels", 0)
        if max_in > 0:
            name = d.get("name", f"Device {idx}")
            res.append(f"{idx}: {name} (in={max_in}, out={max_out})")
    return res


def list_output_devices():
    """
    Devices usable as output (max_output_channels > 0),
    labeled with channel counts to avoid incompatible choices.
    """
    devices = _query_devices()
    res = []
    for idx, d in enumerate(devices):
        max_in = d.get("max_input_channels", 0)
        max_out = d.get("max_output_channels", 0)
        if max_out > 0:
            name = d.get("name", f"Device {idx}")
            res.append(f"{idx}: {name} (in={max_in}, out={max_out})")
    return res


def load_model_fn(model_path, config_path, device):
    global model
    if model_path is None or config_path is None:
        return gr.Dropdown.update(choices=[], value=None), "Please select both model (.pth) and config (.json) files. / モデル(.pth)と設定(.json)の両方を選択してください。"

    try:
        device = cuda[device] if "CUDA" in device else device
        net_g = model_path.name
        cfg = config_path.name
        model = Svc(
            net_g,
            cfg,
            device=device if device != "Auto" else None,
            cluster_model_path="",
            nsf_hifigan_enhance=False,
            shallow_diffusion=False,
            only_diffusion=False,
            spk_mix_enable=False,
            feature_retrieval=False,
        )
        spks = list(model.spk2id.keys())
        dev_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = (
            f"Model loaded on: {dev_name}\n"
            f"Available speakers: {' '.join(spks)}\n"
            f"モデルがデバイス {dev_name} に読み込まれました。利用可能な話者: {' '.join(spks)}"
        )
        return gr.Dropdown.update(choices=spks, value=spks[0] if spks else None), msg
    except Exception as e:
        logging.exception(e)
        return gr.Dropdown.update(choices=[], value=None), f"Load failed: {e} / 読み込みに失敗しました: {e}"


def _normalize_speaker_from_model(sid):
    """
    Ensure the speaker value we pass into Svc.infer is valid.
    Prefer an exact key match in model.spk2id; otherwise fall back
    to the first available speaker.
    """
    global model
    if model is None:
        return None
    spk2id = getattr(model, "spk2id", None)
    if spk2id is None:
        return None

    try:
        keys = list(spk2id.keys())
    except Exception:
        return None
    if not keys:
        return None

    # exact match
    if sid in keys:
        return sid

    # sometimes UI may pass index-like values, try to map them
    try:
        idx = int(sid)
        if 0 <= idx < len(keys):
            return keys[idx]
    except Exception:
        pass

    # fallback: first speaker
    return keys[0]


def test_infer_fn(sid, input_audio, tran, cluster_ratio, auto_f0, noise_scale):
    global model
    if input_audio is None:
        return "Please upload a test audio file first. / 先にテスト用音声ファイルをアップロードしてください。", None
    if model is None:
        return "Please load a model first. / 先にモデルを読み込んでください。", None

    try:
        audio, sampling_rate = soundfile.read(input_audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        truncated_basename = Path(input_audio).stem
        processed_audio_path = os.path.join("raw", f"{truncated_basename}_livevc_test.wav")
        os.makedirs("raw", exist_ok=True)
        soundfile.write(processed_audio_path, audio, sampling_rate, format="wav")

        import io as _io

        buf = _io.BytesIO()
        soundfile.write(buf, audio, sampling_rate, format="wav")
        buf.seek(0)

        speaker = _normalize_speaker_from_model(sid)
        if speaker is None:
            return "No speakers found in model config. / モデル設定に話者が見つかりません。", None

        audio_out, _, _ = model.infer(
            speaker=speaker,
            tran=int(tran),
            raw_path=buf,
            cluster_infer_ratio=float(cluster_ratio),
            auto_predict_f0=bool(auto_f0),
            noice_scale=float(noise_scale),
            f0_filter=False,
        )
        out = audio_out.cpu().numpy()
        output_file = os.path.join("results", f"livevc_test_{truncated_basename}.wav")
        os.makedirs("results", exist_ok=True)
        soundfile.write(output_file, out, model.target_sample, format="wav")
        return "Test conversion completed. / テスト変換が完了しました。", output_file
    except Exception as e:
        logging.exception(e)
        return f"Test failed: {e} / テストに失敗しました: {e}", None


def live_start_fn(in_dev_str, out_dev_str, sid_value, tran, cluster_ratio, auto_f0, noise_scale, bypass):
    global model, live_filter, live_engine
    if model is None:
        return "Please load a model first. / 先にモデルを読み込んでください。"
    if in_dev_str is None or out_dev_str is None:
        return "Please select input and output audio devices. / 入力デバイスと出力デバイスを選択してください。"

    try:
        import sounddevice as sd
    except Exception:
        return "You need to install sounddevice: pip install sounddevice / sounddevice をインストールしてください: pip install sounddevice"

    try:
        in_idx = int(str(in_dev_str).split(":", 1)[0])
        out_idx = int(str(out_dev_str).split(":", 1)[0])
    except Exception:
        return "Failed to parse device indices, please re-select devices. / デバイス番号の解析に失敗しました。デバイスを再選択してください。"

    speaker = _normalize_speaker_from_model(sid_value)
    if speaker is None:
        return "No speakers found in model config. / モデル設定に話者が見つかりません。"

    live_filter = SvcFilter(model, speaker=speaker)
    live_filter.tran = int(tran)
    live_filter.cluster_ratio = float(cluster_ratio)
    live_filter.auto_f0 = bool(auto_f0)
    live_filter.noise_scale = float(noise_scale)
    live_filter.bypass = bool(bypass)

    live_engine = LiveEngine(live_filter)
    try:
        live_engine.start(in_idx, out_idx)
        return "Live stream started. In OBS, select the monitor of this output device as input. / リアルタイム変換を開始しました。OBS でこの出力デバイスのモニターを入力として選択してください。"
    except Exception as e:
        # error text also goes into logs via LiveEngine.start
        return f"Live stream start error: {e} / リアルタイム開始エラー: {e}"


def live_stop_fn():
    global live_engine
    if live_engine is not None:
        live_engine.stop()
        live_engine = None
        return "Live stream stopped. / リアルタイム変換を停止しました。"
    return "Live stream is not running. / リアルタイム変換は実行されていません。"


def live_fetch_logs_fn():
    """
    Return current live VC logs for display in the UI console.
    """
    logs = get_live_logs()
    if not logs:
        return "(no logs yet) / まだログはありません"
    return logs


def live_clear_logs_fn():
    clear_live_logs()
    return ""


def play_test_tone_fn(out_dev_str, freq_hz, duration_s):
    """
    Play a simple sine test tone to the selected output device.
    This helps verify routing to SoVITS-VC / OBS independently of the model.
    """
    if out_dev_str is None:
        return "Select an output device first. / 先に出力デバイスを選択してください。"

    try:
        import sounddevice as sd
    except Exception:
        return "Need sounddevice installed: pip install sounddevice / sounddevice をインストールしてください: pip install sounddevice"

    try:
        out_idx = int(str(out_dev_str).split(":", 1)[0])
    except Exception:
        return "Could not parse output device index. / 出力デバイス番号を解析できませんでした。"

    try:
        sr = 44100
        duration_s = float(duration_s)
        freq_hz = float(freq_hz)
        if duration_s <= 0:
            duration_s = 1.0
        if freq_hz <= 0:
            freq_hz = 440.0
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        sd.play(tone, sr, device=out_idx)
        sd.wait()
        return "Test tone played on selected output device. / 選択した出力デバイスでテストトーンを再生しました。"
    except Exception as e:
        return f"Test tone error: {e} / テストトーンエラー: {e}"


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        font=["Source Sans Pro", "Arial", "sans-serif"],
        font_mono=["JetBrains mono", "Consolas", "Courier New"],
    ),
) as app:
    gr.Markdown(
        """
        # So-VITS-SVC Live VC WebUI / So-VITS-SVC リアルタイムVC WebUI

        Standalone real-time voice conversion UI based on `live_vc.py`:
        - Load `.pth` model and config  
        - Test offline conversion  
        - Configure input/output devices and start live streaming  

        `live_vc.py` をベースにした独立のリアルタイム変換用 WebUI です:
        - `.pth` モデルと設定ファイルの読み込み  
        - オフライン変換テスト  
        - 入力/出力デバイスを設定してリアルタイムストリーミングを開始  
        """
    )

    with gr.Tab("Model & Test / モデルとテスト"):
        with gr.Row():
            with gr.Column():
                model_path = gr.File(label="Model file (.pth) / モデルファイル (.pth)")
                config_path = gr.File(label="Config file (.json) / 設定ファイル (.json)")
                device = gr.Dropdown(
                    label="Device / デバイス",
                    choices=["Auto", *cuda.keys(), "cpu"],
                    value="Auto",
                )
                sid = gr.Dropdown(label="Speaker / 話者", choices=[], interactive=True)
                load_btn = gr.Button("Load model / モデル読込", variant="primary")
                load_status = gr.Textbox(label="Model status / モデル状態", interactive=False)

                load_btn.click(
                    load_model_fn,
                    inputs=[model_path, config_path, device],
                    outputs=[sid, load_status],
                )

            with gr.Column():
                test_audio = gr.Audio(label="Test input audio / テスト入力音声", type="filepath")
                tran = gr.Slider(
                    label="Pitch shift (semitones) / ピッチシフト(半音)",
                    minimum=-12,
                    maximum=12,
                    value=0,
                    step=1,
                )
                cluster_ratio = gr.Slider(
                    label="Cluster ratio (0-1) / クラスタ比率 (0-1)",
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.05,
                )
                auto_f0 = gr.Checkbox(label="Auto F0 prediction / 自動F0予測", value=False)
                noise_scale = gr.Slider(
                    label="Noise scale (noice_scale) / ノイズスケール",
                    minimum=0,
                    maximum=1,
                    value=0.4,
                    step=0.05,
                )
                test_btn = gr.Button("Run test conversion / テスト変換を実行", variant="secondary")
                test_msg = gr.Textbox(label="Test status / テスト状態", interactive=False)
                test_out = gr.Audio(label="Test output audio / テスト出力音声", interactive=False)

                test_btn.click(
                    test_infer_fn,
                    inputs=[sid, test_audio, tran, cluster_ratio, auto_f0, noise_scale],
                    outputs=[test_msg, test_out],
                )

    with gr.Tab("Live Stream / リアルタイム"):
        gr.Markdown(
            """
            Real-time voice conversion on Ubuntu 22.04:

            1. Load a model and select a speaker on the "Model & Test" tab.  
            2. Use `pactl` to create a null-sink as virtual output (e.g. `SoVITS-VC`).  
            3. Select input (mic/monitor) and output (null-sink) devices below.  
            4. Click "Start live stream".  
            5. In OBS, select the monitor of that output device as audio input.  

            Ubuntu 22.04 でのリアルタイム変換手順:

            1. 「モデルとテスト」タブでモデルを読み込み、話者を選択します。  
            2. `pactl` で null-sink を作成し、仮想出力(例: `SoVITS-VC`)にします。  
            3. 下で入力デバイス(マイク/monitor)と出力デバイス(null-sink)を選択します。  
            4. 「Start live stream」を押します。  
            5. OBS 側でその出力デバイスの Monitor を音声入力として選択します。  
            """
        )

        with gr.Row():
            live_in = gr.Dropdown(
                label="Input device (mic/monitor) / 入力デバイス(マイク/monitor)",
                choices=list_input_devices(),
                interactive=True,
            )
            live_out = gr.Dropdown(
                label="Output device (null-sink recommended) / 出力デバイス(null-sink 推奨)",
                choices=list_output_devices(),
                interactive=True,
            )
        with gr.Row():
            tran_live = gr.Slider(
                label="Live pitch shift (semitones) / リアルタイム ピッチシフト(半音)",
                minimum=-12,
                maximum=12,
                value=0,
                step=1,
            )
            cluster_ratio_live = gr.Slider(
                label="Live cluster ratio / リアルタイム クラスタ比率",
                minimum=0,
                maximum=1,
                value=0,
                step=0.05,
            )
        with gr.Row():
            auto_f0_live = gr.Checkbox(label="Live auto F0 prediction / リアルタイム自動F0予測", value=False)
            noise_scale_live = gr.Slider(
                label="Live noise scale / リアルタイム ノイズスケール",
                minimum=0,
                maximum=1,
                value=0.4,
                step=0.05,
            )
            bypass_live = gr.Checkbox(
                label="Bypass model (dry monitor) / モデルをバイパス(素の音)", value=False
            )
        with gr.Row():
            live_status = gr.Textbox(label="Live stream status / リアルタイム状態", interactive=False)
        with gr.Row():
            live_console = gr.Textbox(
                label="Live logs / ライブログ",
                interactive=False,
                lines=10,
            )
        with gr.Row():
            test_freq = gr.Number(
                label="Test tone frequency (Hz) / テストトーン周波数(Hz)",
                value=440,
            )
            test_duration = gr.Number(
                label="Test tone duration (s) / テストトーン長さ(秒)",
                value=1.0,
            )
            test_tone_btn = gr.Button("Play test tone on output / 出力にテストトーンを再生", variant="secondary")
            test_tone_status = gr.Textbox(label="Test tone status / テストトーン状態", interactive=False)
        with gr.Row():
            live_start = gr.Button("Start live stream / リアルタイム開始", variant="primary")
            live_stop = gr.Button("Stop live stream / リアルタイム停止", variant="primary")
            live_refresh_logs = gr.Button("Refresh logs / ログ更新")
            live_clear_logs = gr.Button("Clear logs / ログ削除")

        live_start.click(
            live_start_fn,
            inputs=[
                live_in,
                live_out,
                sid,
                tran_live,
                cluster_ratio_live,
                auto_f0_live,
                noise_scale_live,
                bypass_live,
            ],
            outputs=[live_status],
        )
        live_stop.click(
            live_stop_fn,
            inputs=[],
            outputs=[live_status],
        )
        live_refresh_logs.click(
            live_fetch_logs_fn,
            inputs=[],
            outputs=[live_console],
        )
        live_clear_logs.click(
            live_clear_logs_fn,
            inputs=[],
            outputs=[live_console],
        )
        test_tone_btn.click(
            play_test_tone_fn,
            inputs=[live_out, test_freq, test_duration],
            outputs=[test_tone_status],
        )


if __name__ == "__main__":
    # use a different port from webUI.py so they can co-exist
    app.launch(server_name="0.0.0.0", server_port=7861)

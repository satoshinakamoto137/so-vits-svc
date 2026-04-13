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
        modes = {"凸結合": 0, "線形結合": 1}
        mode = modes[mode]
        data = json.loads(js)
        data = list(data.items())
        model_path, mix_rate = zip(*data)
        path = mix_model(model_path, mix_rate, mode)
        return f"成功しました。混合モデルは {path} に保存されました"
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
        # モデルと設定ファイルのパスを取得
        if local_model_enabled:
            # ローカルパス
            model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
            config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
        else:
            # Web UI からアップロード
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
        spks = list(model.spk2id.keys())
        device_name = (
            torch.cuda.get_device_properties(model.dev).name
            if "cuda" in str(model.dev)
            else str(model.dev)
        )
        msg = f"モデルをデバイス {device_name} にロードしました\n"
        if cluster_model_path is None:
            msg += "クラスタモデルまたは特徴量検索モデルはロードされていません\n"
        elif fr:
            msg += f"特徴量検索モデル {cluster_filepath[1]} のロードに成功しました\n"
        else:
            msg += f"クラスタモデル {cluster_filepath[1]} のロードに成功しました\n"
        if diff_model_path is None:
            msg += "拡散モデルはロードされていません\n"
        else:
            msg += f"拡散モデル {diff_model_path.name} のロードに成功しました\n"
        msg += "現在のモデルで利用可能な話者：\n"
        for i in spks:
            msg += i + " "
        return sid.update(choices=spks, value=spks[0]), msg
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def modelUnload():
    global model
    if model is None:
        return sid.update(choices=[], value=""), "アンロードするモデルがありません！"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return sid.update(choices=[], value=""), "モデルをアンロードしました！"


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
    # 結果を保存するパスを作成し、results フォルダに保存
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
            return "音声ファイルをアップロードしてください", None
        if model is None:
            return "モデルをロードしてください", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "クラスタモデルまたは特徴量検索モデルをロードしてからクラスタ比率を設定してください", None
        audio, sampling_rate = soundfile.read(input_audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        # Gradio が渡す filepath には原因不明のサフィックスが付くため、ここで削除
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

        return "成功しました", output_file
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
            return "モデルをロードしてください", None
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            if cluster_ratio != 0:
                return "クラスタモデルまたは特徴量検索モデルをロードしてからクラスタ比率を設定してください", None
        _rate = f"+{int(_rate * 100)}%" if _rate >= 0 else f"{int(_rate * 100)}%"
        _volume = (
            f"+{int(_volume * 100)}%"
            if _volume >= 0
            else f"{int(_volume * 100)}%"
        )

        # UI からの性別表示（「男性」「女性」）を EdgeTTS 用の "Male"/"Female" に変換
        gender_for_tts = _gender
        if _gender in ("男性", "男", "Male"):
            gender_for_tts = "Male"
        elif _gender in ("女性", "女", "Female"):
            gender_for_tts = "Female"

        if _lang == "Auto":
            subprocess.run(
                [sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, gender_for_tts]
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
        return "成功しました", output_file_path
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def model_compression(_model):
    if _model == "":
        return "先に圧縮するモデルを選択してください"
    else:
        model_path = os.path.split(_model.name)
        filename, extension = os.path.splitext(model_path[1])
        output_model_name = f"{filename}_compressed{extension}"
        output_path = os.path.join(os.getcwd(), output_model_name)
        removeOptimizer(_model.name, output_path)
        return f"圧縮済みモデルは {output_path} に保存されました"


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
            # json と pth がそれぞれ 1 つずつあるディレクトリのみ使用
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
        with gr.TabItem("推論"):
            gr.Markdown(
                value="""
                So-VITS-SVC 4.0 推論 WebUI
                """
            )
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(
                        value="""
                        <font size=2> モデル設定</font>
                        """
                    )
                    with gr.Tabs():
                        # タブ状態を追跡するための不可視チェックボックス
                        local_model_enabled = gr.Checkbox(value=False, visible=False)
                        with gr.TabItem('アップロード') as local_model_tab_upload:
                            with gr.Row():
                                model_path = gr.File(label="モデルファイルを選択")
                                config_path = gr.File(label="設定ファイルを選択")
                        with gr.TabItem('ローカル') as local_model_tab_local:
                            gr.Markdown(
                                f'モデルフォルダは {local_model_root} 以下に配置してください'
                            )
                            local_model_refresh_btn = gr.Button(
                                'ローカルモデル一覧を更新'
                            )
                            local_model_selection = gr.Dropdown(
                                label='モデルフォルダを選択',
                                choices=[],
                                interactive=True,
                            )
                    with gr.Row():
                        diff_model_path = gr.File(
                            label="拡散モデルファイルを選択"
                        )
                        diff_config_path = gr.File(
                            label="拡散モデルの設定ファイルを選択"
                        )
                    cluster_model_path = gr.File(
                        label="クラスタ / 特徴量検索モデルファイルを選択（任意）"
                    )
                    device = gr.Dropdown(
                        label="推論デバイス（デフォルト：CPU/GPU 自動選択）",
                        choices=["Auto", *cuda.keys(), "cpu"],
                        value="Auto",
                    )
                    enhance = gr.Checkbox(
                        label=(
                            "NSF_HIFIGAN 強調を使用する。学習データが少ないモデルでは音質改善の可能性あり。"
                            "一方で十分に学習されたモデルでは悪化することもあります（デフォルト：オフ）。"
                        ),
                        value=False,
                    )
                    only_diffusion = gr.Checkbox(
                        label=(
                            "拡散モデルのみで推論する。オンにすると So-VITS モデルは使用せず、"
                            "拡散モデルのみで完全な拡散推論を行います（デフォルト：オフ）。"
                        ),
                        value=False,
                    )
                with gr.Column():
                    gr.Markdown(
                        value="""
                        <font size=3>
                        左側で必要なファイルをすべて選択し（各ファイルモジュールが「download」と表示された状態）、  
                        「モデルを読み込み」をクリックして解析・ロードしてください：
                        </font>
                        """
                    )
                    model_load_button = gr.Button(value="モデルを読み込み", variant="primary")
                    model_unload_button = gr.Button(
                        value="モデルをアンロード", variant="primary"
                    )
                    sid = gr.Dropdown(label="話者（スピーカー）")
                    sid_output = gr.Textbox(label="出力メッセージ")

            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(
                        value="""
                        <font size=2> 推論設定</font>
                        """
                    )
                    auto_f0 = gr.Checkbox(
                        label=(
                            "自動 F0 予測。クラスタ / 特徴量検索モデルと併用すると効果的ですが、"
                            "ピッチ変更機能は無効になります。歌声変換では極端な音程ずれを"
                            "引き起こすことがあります。"
                        ),
                        value=False,
                    )
                    f0_predictor = gr.Dropdown(
                        label=(
                            "F0 予測器を選択（crepe, pm, dio, harvest, rmvpe）。"
                            "デフォルト：pm（注意：crepe は生の F0 に平均フィルタを適用します）。"
                        ),
                        choices=["pm", "dio", "harvest", "crepe", "rmvpe"],
                        value="pm",
                    )
                    vc_transform = gr.Number(
                        label=(
                            "ピッチシフト量（半音単位、正負可。+12 で 1 オクターブ上）。"
                        ),
                        value=0,
                    )
                    cluster_ratio = gr.Number(
                        label=(
                            "クラスタ / 特徴量検索の混合比率（0–1）。0 で無効。"
                            "使用すると音色の類似度は向上しますが、発音が不明瞭になる可能性があります "
                            "（0.5 前後が一般的な目安）。"
                        ),
                        value=0,
                    )
                    slice_db = gr.Number(label="自動スライスのしきい値（dB）", value=-40)
                    output_format = gr.Radio(
                        label="出力音声フォーマット",
                        choices=["wav", "flac", "mp3"],
                        value="wav",
                    )
                    noise_scale = gr.Number(
                        label=(
                            "noise_scale（変更非推奨）。音質に影響する、やや『おまじない』的なパラメータ。"
                        ),
                        value=0.4,
                    )
                    k_step = gr.Slider(
                        label=(
                            "浅い拡散のステップ数（拡散モデル使用時のみ有効）。"
                            "ステップ数を増やすほど純粋な拡散モデルの結果に近づきます。"
                        ),
                        value=100,
                        minimum=1,
                        maximum=1000,
                    )
                with gr.Column():
                    pad_seconds = gr.Number(
                        label=(
                            "推論音声の前後に追加するパディング秒数。原因不明のアーティファクトが"
                            "冒頭と末尾に出る場合があるため、短い無音を追加することで回避します。"
                        ),
                        value=0.5,
                    )
                    cl_num = gr.Number(
                        label=(
                            "自動音声スライスの長さ（秒）。0 でスライス無効。"
                        ),
                        value=0,
                    )
                    lg_num = gr.Number(
                        label=(
                            "スライス間のクロスフェード長（秒）。"
                            "スライス後に音声が途切れて聞こえる場合に値を大きくしてください。"
                            "十分滑らかな場合は 0 を推奨。値を大きくすると推論速度が低下します。"
                        ),
                        value=0,
                    )
                    lgr_num = gr.Number(
                        label=(
                            "自動スライス後に各スライス両端を破棄する際、"
                            "クロスフェード部分として残す割合。範囲は 0–1（左開右閉）。"
                        ),
                        value=0.75,
                    )
                    enhancer_adaptive_key = gr.Number(
                        label=(
                            "エンハンサーをより高い音域に適応させる半音数（デフォルト 0）。"
                        ),
                        value=0,
                    )
                    cr_threshold = gr.Number(
                        label=(
                            "F0 フィルタのしきい値（crepe 使用時のみ有効）。範囲 0–1。"
                            "値を下げると音程の破綻は減りますが、無声区間が増える可能性があります。"
                        ),
                        value=0.05,
                    )
                    loudness_envelope_adjustment = gr.Number(
                        label=(
                            "入力元のラウドネス包絡と出力側のラウドネス包絡の混合比。"
                            "1 に近づくほど出力側を優先します。"
                        ),
                        value=0,
                    )
                    second_encoding = gr.Checkbox(
                        label=(
                            "二重エンコード。浅い拡散の前に入力音声をもう一度エンコードします。"
                            "効果はやや予測しづらく、デフォルトではオフです。"
                        ),
                        value=False,
                    )
                    use_spk_mix = gr.Checkbox(
                        label="動的声質ミックス",
                        value=False,
                        interactive=False,
                    )
            with gr.Tabs():
                with gr.TabItem("音声 → 音声"):
                    vc_input3 = gr.Audio(label="音声ファイルを選択", type="filepath")
                    vc_submit = gr.Button("音声を変換", variant="primary")
                with gr.TabItem("テキスト → 音声"):
                    text2tts = gr.Textbox(
                        label=(
                            "変換したいテキストを入力してください。"
                            "この機能を使用する場合、F0 予測をオンにすることを推奨します。"
                            "オフのままだと違和感のある音声になる場合があります。"
                        )
                    )
                    with gr.Row():
                        tts_gender = gr.Radio(
                            label="話者の性別",
                            choices=["男性", "女性"],
                            value="男性",
                        )
                        tts_lang = gr.Dropdown(
                            label="言語を選択（Auto はテキストから自動判定）",
                            choices=SUPPORTED_LANGUAGES,
                            value="Auto",
                        )
                        tts_rate = gr.Slider(
                            label="TTS の再生速度（相対倍率）",
                            minimum=-1,
                            maximum=3,
                            value=0,
                            step=0.1,
                        )
                        tts_volume = gr.Slider(
                            label="TTS の音量（相対値）",
                            minimum=-1,
                            maximum=1.5,
                            value=0,
                            step=0.1,
                        )
                    vc_submit2 = gr.Button("テキストを変換", variant="primary")
            with gr.Row():
                with gr.Column():
                    vc_output1 = gr.Textbox(label="出力メッセージ")
                with gr.Column():
                    vc_output2 = gr.Audio(
                        label="出力音声",
                        interactive=False,
                    )

        with gr.TabItem("ツール / 実験的機能"):
            gr.Markdown(
                value="""
                        <font size=2> So-VITS-SVC 4.0 ツール / 実験的機能</font>
                        """
            )
            with gr.Tabs():
                with gr.TabItem("静的声質ブレンド"):
                    gr.Markdown(
                        value="""
                        <font size=2>
                        説明：複数の音声モデルを 1 つのモデルに結合し
                        （モデルパラメータの凸結合または線形結合）、
                        現実には存在しない新しい声質を作成する機能です。

                        注意事項：
                        1. 本機能は単一話者モデルのみをサポートします。
                        2. 複数話者モデルを無理に使用する場合、話者数がすべてのモデルで
                           一致している必要があります。同じ SpeakerID 同士が混合されます。
                        3. すべてのモデルの config.json 内の model フィールドが同一である必要があります。
                        4. 出力された混合モデルは、元のどの config.json でも利用可能ですが、
                           クラスタモデルは使用できません。
                        5. 複数モデルをまとめてアップロードする場合は、1 つのフォルダにまとめて
                           アップロードすることを推奨します。
                        6. 混合比率は 0–100 の範囲を推奨しますが、それ以外も指定可能です。
                           線形結合モードでは予期しない結果になることがあります。
                        7. 混合後のファイルはプロジェクトルートに output.pth という名前で保存されます。
                        8. 凸結合モードでは混合比率に Softmax を適用し、合計が 1 になるようにします。
                           線形結合モードでは正規化を行いません。
                        </font>
                        """
                    )
                    mix_model_path = gr.Files(
                        label="混合するモデルファイルを選択"
                    )
                    mix_model_upload_button = gr.UploadButton(
                        "混合するモデルファイルを選択 / 追加",
                        file_count="multiple",
                    )
                    mix_model_output1 = gr.Textbox(
                        label="混合比率（パーセント %）",
                        interactive=True,
                    )
                    mix_mode = gr.Radio(
                        choices=["凸結合", "線形結合"],
                        label="混合モード",
                        value="凸結合",
                        interactive=True,
                    )
                    mix_submit = gr.Button("声質ブレンドを開始", variant="primary")
                    mix_model_output2 = gr.Textbox(
                        label="出力メッセージ",
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

                with gr.TabItem("モデル圧縮ツール"):
                    gr.Markdown(
                        value="""
                        このツールはモデルの容量を圧縮します。
                        推論性能を **維持したまま**、約 600 MB の So-VITS モデルを
                        約 200 MB まで圧縮し、ディスク使用量を大きく削減できます。

                        **注意：圧縮したモデルは再学習できません。**
                        もう学習しないと決めたモデルのみ圧縮してください。
                    """
                    )
                    model_to_compress = gr.File(label="モデルをアップロード")
                    compress_model_btn = gr.Button(
                        "モデルを圧縮",
                        variant="primary",
                    )
                    compress_model_output = gr.Textbox(
                        label="出力メッセージ",
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
                    <font size=2> WebUI 設定</font>
                    """
                )
                debug_button = gr.Checkbox(
                    label=(
                        "デバッグモード。バグ報告を行う際はオンにしてください。"
                        "コンソールに詳細なエラーメッセージが表示されます。"
                    ),
                    value=debug,
                )
        # ローカルモデル一覧を更新
        local_model_refresh_btn.click(
            local_model_refresh_fn,
            outputs=local_model_selection,
        )
        # タブ切り替え時にローカルモードの有効 / 無効を設定
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

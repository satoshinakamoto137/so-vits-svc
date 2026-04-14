import io
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


# Simple in-memory log buffer for debugging live streaming.
_LIVE_LOGS = []


def live_log(msg: str):
    """
    Append a line to the live VC log buffer.
    """
    global _LIVE_LOGS
    _LIVE_LOGS.append(str(msg))
    # prevent unbounded growth
    if len(_LIVE_LOGS) > 500:
        _LIVE_LOGS = _LIVE_LOGS[-500:]


def get_live_logs() -> str:
    """
    Return logs as a single string.
    """
    return "\n".join(_LIVE_LOGS)


def clear_live_logs():
    """
    Clear the in-memory live logs.
    """
    _LIVE_LOGS.clear()


class SvcFilter:
    """
    Lightweight wrapper that treats an existing Svc instance
    (from inference.infer_tool.Svc) as an audio 'filter'.

    It uses the current model, speaker and basic VC parameters
    to transform incoming mono float32 audio blocks.
    """

    def __init__(self, svc_model, speaker: Optional[str] = None):
        self.svc_model = svc_model
        # Default parameters – these can be exposed in the UI if needed
        self.tran = 0
        self.cluster_ratio = 0.0
        self.auto_f0 = False
        self.noise_scale = 0.4
        self.f0_predictor = "pm"
        self.speaker = speaker
        # when True, bypass model and return input audio
        self.bypass = False

    def process(self, wav: np.ndarray, sr: int):
        """
        :param wav: mono float32 numpy array
        :param sr:  input sample rate
        :return: (processed_wav, processed_sr)
        """
        if self.bypass:
            # dry monitor
            live_log(f"SvcFilter.process bypassed (len={len(wav)}, sr={sr}).")
            return wav, sr

        if self.svc_model is None:
            live_log("SvcFilter.process called with no model loaded.")
            return wav, sr

        # pick a speaker if not provided
        speaker = self.speaker
        if speaker is None:
            try:
                speaker = list(self.svc_model.spk2id.keys())[0]
            except Exception:
                live_log("No speakers available in model.spk2id.")
                return wav, sr

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="wav")
        buf.seek(0)

        try:
            audio, _, _ = self.svc_model.infer(
                speaker=speaker,
                tran=self.tran,
                raw_path=buf,
                cluster_infer_ratio=self.cluster_ratio,
                auto_predict_f0=self.auto_f0,
                noice_scale=self.noise_scale,
                f0_filter=False,
                f0_predictor=self.f0_predictor,
            )
        except Exception as exc:  # pragma: no cover - debug path
            live_log(f"SvcFilter.process error: {exc}")
            return wav, sr
        return audio.cpu().numpy(), self.svc_model.target_sample


class LiveEngine:
    """
    Minimal real-time engine:
    - captures mono audio from an input device
    - runs it through a SvcFilter
    - plays it to an output device

    This is designed for Linux (PulseAudio/PipeWire) but relies only on
    sounddevice, so it should work anywhere sounddevice does.
    """

    def __init__(self, svc_filter: SvcFilter):
        self.filter = svc_filter
        self.stream: Optional[sd.Stream] = None
        self.running = False
        self.blocksize = 2048  # tune for latency vs CPU
         # number of channels actually used for the stream
        self.channels = 1
        self._logged_callback_started = False

    def start(self, input_device_index: int, output_device_index: int):
        if self.running:
            live_log("LiveEngine.start called but engine is already running.")
            return
        self.running = True

        # Inspect device capabilities to choose valid channel count
        try:
            in_info = sd.query_devices(input_device_index)
            out_info = sd.query_devices(output_device_index)
            live_log(f"Input device info: {in_info}")
            live_log(f"Output device info: {out_info}")
            max_in = in_info.get("max_input_channels", 0)
            max_out = out_info.get("max_output_channels", 0)
            channels = min(max_in, max_out)
            if channels <= 0:
                raise ValueError(
                    f"No common input/output channels (in={max_in}, out={max_out})"
                )
            # Keep it simple: use at most 2 channels
            self.channels = min(channels, 2)
        except Exception as exc:
            live_log(f"Failed to query devices/channels: {exc}")
            raise

        # Use model target sample rate if available, else default to 44100
        sr = getattr(self.filter.svc_model, "target_sample", 44100)

        live_log(
            f"Starting LiveEngine with sr={sr}, channels={self.channels}, in_dev={input_device_index}, out_dev={output_device_index}"
        )

        def callback(indata, outdata, frames, time, status):  # noqa: ARG001
            try:
                if status:
                    live_log(f"sounddevice status: {status}")

                if not self._logged_callback_started:
                    live_log(
                        f"LiveEngine callback first run: frames={frames}, in_shape={indata.shape}"
                    )
                    self._logged_callback_started = True

                if not self.running:
                    outdata[:] = 0
                    return

                # downmix input to mono regardless of device channel count
                mono = indata.mean(axis=1).astype(np.float32)
                y, y_sr = self.filter.process(mono, sr)

                # simple resample (if needed) using linear interpolation
                if y_sr != sr and len(y) > 1:
                    xp = np.linspace(0, 1, num=len(y), endpoint=False)
                    xnew = np.linspace(0, 1, num=frames, endpoint=False)
                    y = np.interp(xnew, xp, y)
                else:
                    if len(y) < frames:
                        y = np.pad(y, (0, frames - len(y)))
                    y = y[:frames]

                # expand mono back to the configured number of channels
                if self.channels == 1:
                    outdata[:] = y.reshape(-1, 1)
                else:
                    out_stereo = np.repeat(y.reshape(-1, 1), self.channels, axis=1)
                    outdata[:] = out_stereo
            except Exception as exc:  # pragma: no cover - debug path
                live_log(f"LiveEngine callback error: {exc}")
                try:
                    outdata[:] = 0
                except Exception:
                    pass

        self.stream = sd.Stream(
            device=(input_device_index, output_device_index),
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        )

        try:
            self.stream.start()
            live_log("LiveEngine stream started.")
        except Exception as exc:  # pragma: no cover - debug path
            live_log(f"LiveEngine.start error: {exc}")
            self.running = False
            self.stream = None
            raise

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        live_log("LiveEngine stream stopped.")

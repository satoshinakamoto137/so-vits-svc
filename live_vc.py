import io
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


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

    def process(self, wav: np.ndarray, sr: int):
        """
        :param wav: mono float32 numpy array
        :param sr:  input sample rate
        :return: (processed_wav, processed_sr)
        """
        if self.svc_model is None:
            return wav, sr

        # pick a speaker if not provided
        speaker = self.speaker
        if speaker is None:
            speaker = list(self.svc_model.spk2id.keys())[0]

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="wav")
        buf.seek(0)

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

    def start(self, input_device_index: int, output_device_index: int):
        if self.running:
            return
        self.running = True

        # Use model target sample rate if available, else default to 44100
        sr = getattr(self.filter.svc_model, "target_sample", 44100)

        def callback(indata, outdata, frames, time, status):  # noqa: ARG001
            if not self.running:
                outdata[:] = 0
                return

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

            outdata[:] = y.reshape(-1, 1)

        self.stream = sd.Stream(
            device=(input_device_index, output_device_index),
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


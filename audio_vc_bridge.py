#!/usr/bin/env python3
"""
Realtime mic → So-VITS-SVC → output bridge.

Architecture:
  - Input callback: push mono blocks into a queue.
  - Worker thread: pull blocks, run Svc inference, push processed blocks.
  - Output callback: pull processed blocks, write to output device.

Routing on Ubuntu (PulseAudio):
  - Use mic as --input-device
  - Use `pulse` (or `default`) as --output-device
  - Route that stream to a null sink like SoVITS-VC via pavucontrol
  - In OBS, capture "Monitor of SoVITS-VC".
"""

import argparse
import atexit
import io
import queue
import sys
import threading
import time
import traceback

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from inference.infer_tool import Svc

# global reference so we can always stop streams on process exit
_GLOBAL_ENGINE = None


def log(msg: str) -> None:
    print(msg, flush=True)


def list_devices() -> None:
    print("\nAvailable audio devices:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(
            f"{i}: {dev['name']} | "
            f"in={dev['max_input_channels']} out={dev['max_output_channels']} | "
            f"default_sr={dev['default_samplerate']}"
        )
    print("")


def normalize_speaker(model: Svc, sid: str | None) -> str:
    spk2id = getattr(model, "spk2id", None)
    if not spk2id:
        raise RuntimeError("Model config has no speakers (spk field missing).")
    keys = list(spk2id.keys())
    if sid is None:
        return keys[0]
    if sid in keys:
        return sid
    try:
        idx = int(sid)
        if 0 <= idx < len(keys):
            return keys[idx]
    except Exception:
        pass
    raise RuntimeError(f"Speaker '{sid}' not found. Available: {', '.join(keys)}")


class SvcFilter:
    """
    Lightweight wrapper that runs Svc.infer on a mono float32 block.
    """

    def __init__(
        self,
        svc_model: Svc,
        speaker: str,
        tran: int = 0,
        cluster_ratio: float = 0.0,
        auto_f0: bool = False,
        noise_scale: float = 0.4,
        f0_predictor: str = "harvest",
    ):
        self.svc_model = svc_model
        self.speaker = speaker
        self.tran = tran
        self.cluster_ratio = cluster_ratio
        self.auto_f0 = auto_f0
        self.noise_scale = noise_scale
        self.f0_predictor = f0_predictor

    def process_block(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        :param wav: mono float32 numpy array
        :param sr:  input sample rate
        :return: (processed_wav, processed_sr)
        """
        if self.svc_model is None:
            raise RuntimeError("SvcFilter called without a loaded model.")

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="wav")
        buf.seek(0)

        audio_out, _, _ = self.svc_model.infer(
            speaker=self.speaker,
            tran=self.tran,
            raw_path=buf,
            cluster_infer_ratio=self.cluster_ratio,
            auto_predict_f0=self.auto_f0,
            noice_scale=self.noise_scale,
            f0_filter=False,
            f0_predictor=self.f0_predictor,
        )

        y = audio_out.detach().cpu().numpy().astype(np.float32)
        return y, self.svc_model.target_sample


class AudioVCBridgeEngine:
    def __init__(
        self,
        input_device: int,
        output_device: int,
        sample_rate: int,
        blocksize: int,
        bypass: bool,
        svc_filter: SvcFilter | None,
        gain: float,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.bypass = bypass
        self.filter = svc_filter
        self.gain = gain

        self.running = False
        self.input_stream = None
        self.output_stream = None
        self.worker_thread = None

        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

    def input_callback(self, indata, frames, time_info, status):
        if status:
            log(f"[input status] {status}")

        if not self.running:
            return

        try:
            mono = indata.mean(axis=1).astype(np.float32).copy()
            if not self.input_queue.full():
                self.input_queue.put_nowait(mono)
        except Exception:
            log("[input callback error]\n" + traceback.format_exc())

    def output_callback(self, outdata, frames, time_info, status):
        if status:
            log(f"[output status] {status}")

        if not self.running:
            outdata.fill(0)
            return

        try:
            if self.output_queue.empty():
                outdata.fill(0)
                return

            y = self.output_queue.get_nowait()

            if len(y) < frames:
                y = np.pad(y, (0, frames - len(y)))
            elif len(y) > frames:
                y = y[:frames]

            outdata[:, 0] = y
        except Exception:
            outdata.fill(0)
            log("[output callback error]\n" + traceback.format_exc())

    def worker_loop(self):
        log("[worker] started")
        peak_counter = 0

        while self.running:
            try:
                x = self.input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # Apply gain first
                x_proc = (x * self.gain).astype(np.float32)

                if self.bypass or self.filter is None:
                    y = x_proc
                    y_sr = self.sample_rate
                else:
                    y, y_sr = self.filter.process_block(x_proc, self.sample_rate)

                # Resample if model sample rate != engine sample rate
                if y_sr != self.sample_rate and len(y) > 1:
                    x_old = np.linspace(0, 1, len(y), endpoint=False)
                    x_new = np.linspace(
                        0, 1, int(len(y) * self.sample_rate / y_sr), endpoint=False
                    )
                    y = np.interp(x_new, x_old, y).astype(np.float32)

                # Limit peaks to avoid clipping
                y = np.clip(y, -1.0, 1.0).astype(np.float32)

                if not self.output_queue.full():
                    self.output_queue.put_nowait(y)

                peak_counter += 1
                if peak_counter >= 20:
                    peak = float(np.max(np.abs(y))) if len(y) else 0.0
                    log(f"[worker] block ok | peak={peak:.3f}")
                    peak_counter = 0

            except Exception:
                log("[worker error]\n" + traceback.format_exc())

        log("[worker] stopped")

    def start(self):
        self.running = True

        log(
            f"Starting VC engine | in={self.input_device} out={self.output_device} "
            f"sr={self.sample_rate} blocksize={self.blocksize} bypass={self.bypass} gain={self.gain}"
        )

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        self.input_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.input_callback,
        )

        self.output_stream = sd.OutputStream(
            device=self.output_device,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.output_callback,
        )

        self.input_stream.start()
        self.output_stream.start()
        log("Streams started. Press Ctrl+C to stop.")

    def stop(self):
        self.running = False

        try:
            if self.input_stream is not None:
                self.input_stream.stop()
                self.input_stream.close()
        except Exception:
            log("[stop input error]\n" + traceback.format_exc())

        try:
            if self.output_stream is not None:
                self.output_stream.stop()
                self.output_stream.close()
        except Exception:
            log("[stop output error]\n" + traceback.format_exc())

        log("Engine stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Realtime mic -> So-VITS-SVC -> output bridge (Ubuntu/OBS friendly)."
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/G_0.pth",
        help="Path to So-VITS-SVC .pth model",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="models/confign.json",
        help="Path to So-VITS-SVC config.json",
    )
    parser.add_argument("--speaker", type=str, default=None, help="Speaker name or index from config")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index (use pulse/default for PulseAudio)")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Streaming sample rate (typically 44100 or 48000)",
    )
    parser.add_argument("--blocksize", type=int, default=2048, help="Audio block size")
    parser.add_argument(
        "--bypass",
        action="store_true",
        help="Bypass VC (mic → output direct, only gain applied)",
    )
    parser.add_argument(
        "--tran",
        type=int,
        default=0,
        help="Pitch shift in semitones (tran)",
    )
    parser.add_argument(
        "--cluster-ratio",
        type=float,
        default=0.0,
        help="Cluster ratio for SVC (0.0 disables)",
    )
    parser.add_argument(
        "--auto-f0",
        action="store_true",
        help="Enable automatic F0 prediction",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.4,
        help="Noise scale for SVC",
    )
    parser.add_argument(
        "--f0-predictor",
        type=str,
        default="harvest",
        help="F0 predictor to use (pm / harvest / dio / crepe / rmvpe / fcpe)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Gain multiplier applied before VC",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.input_device is None or args.output_device is None:
        print("Error: you must provide --input-device and --output-device")
        print("Use --list-devices first.")
        sys.exit(1)

    # Load SVC model unless in bypass-only mode
    svc_model = None
    svc_filter = None
    try:
        if not args.bypass:
            log(f"Loading SVC model: {args.model_path}")
            device = "cuda" if torch.cuda.is_available() else None
            # When device is None, Svc will pick automatically
            svc_model = Svc(
                args.model_path,
                args.config_path,
                device=device,
                cluster_model_path="",
                nsf_hifigan_enhance=False,
                shallow_diffusion=False,
                only_diffusion=False,
                spk_mix_enable=False,
                feature_retrieval=False,
            )
            speaker = normalize_speaker(svc_model, args.speaker)
            log(f"Using speaker: {speaker}")

            svc_filter = SvcFilter(
                svc_model=svc_model,
                speaker=speaker,
                tran=args.tran,
                cluster_ratio=args.cluster_ratio,
                auto_f0=args.auto_f0,
                noise_scale=args.noise_scale,
                f0_predictor=args.f0_predictor,
            )
        else:
            log("Bypass mode enabled: VC model will not be loaded.")
    except Exception as e:
        print(f"Failed to load So-VITS-SVC model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        in_info = sd.query_devices(args.input_device)
        out_info = sd.query_devices(args.output_device)

        print("Input device:", in_info["name"])
        print("Output device:", out_info["name"])
        print(f"Requested sample rate: {args.sample_rate}")
        print("")

        engine = AudioVCBridgeEngine(
            input_device=args.input_device,
            output_device=args.output_device,
            sample_rate=args.sample_rate,
            blocksize=args.blocksize,
            bypass=args.bypass,
            svc_filter=svc_filter,
            gain=args.gain,
        )

        # register global engine for cleanup on normal interpreter exit
        global _GLOBAL_ENGINE
        _GLOBAL_ENGINE = engine
        atexit.register(lambda: _GLOBAL_ENGINE and _GLOBAL_ENGINE.stop())

        engine.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        engine.stop()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        try:
            engine.stop()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

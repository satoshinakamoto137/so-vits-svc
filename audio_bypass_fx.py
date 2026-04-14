#!/usr/bin/env python3
import argparse
import atexit
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd


class SimpleDelayReverb:
    """
    Efecto simple tipo delay/reverb para probar el flujo.
    No busca sonar perfecto; busca demostrar que sí puedes procesar audio live.
    """

    def __init__(self, sample_rate: int, delay_ms: float = 120.0, feedback: float = 0.25, wet: float = 0.25):
        self.sample_rate = sample_rate
        self.delay_ms = delay_ms
        self.feedback = float(np.clip(feedback, 0.0, 0.95))
        self.wet = float(np.clip(wet, 0.0, 1.0))

        delay_samples = max(1, int(sample_rate * delay_ms / 1000.0))
        self.buffer = np.zeros(delay_samples, dtype=np.float32)
        self.index = 0

    def process_block(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=np.float32)

        for i, sample in enumerate(x):
            delayed = self.buffer[self.index]
            out = sample + delayed * self.wet
            self.buffer[self.index] = sample + delayed * self.feedback
            self.index = (self.index + 1) % len(self.buffer)
            y[i] = out

        return y


class AudioBypassEngine:
    def __init__(
        self,
        input_device: int | None,
        output_device: int | None,
        sample_rate: int,
        blocksize: int,
        gain: float,
        use_reverb: bool,
        delay_ms: float,
        feedback: float,
        wet: float,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.gain = gain
        self.use_reverb = use_reverb

        self.reverb = SimpleDelayReverb(
            sample_rate=sample_rate,
            delay_ms=delay_ms,
            feedback=feedback,
            wet=wet,
        )

        self.running = False
        self.input_stream = None
        self.output_stream = None
        self.worker_thread = None

        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)

    def log(self, msg: str) -> None:
        print(msg, flush=True)

    def input_callback(self, indata, frames, time_info, status):
        if status:
            self.log(f"[input status] {status}")

        if not self.running:
            return

        try:
            mono = indata[:, 0].astype(np.float32).copy()
            if not self.input_queue.full():
                self.input_queue.put_nowait(mono)
        except Exception as e:
            self.log(f"[input callback error] {e}")

    def output_callback(self, outdata, frames, time_info, status):
        if status:
            self.log(f"[output status] {status}")

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
        except Exception as e:
            self.log(f"[output callback error] {e}")
            outdata.fill(0)

    def worker_loop(self):
        self.log("[worker] started")
        peak_counter = 0

        while self.running:
            try:
                x = self.input_queue.get(timeout=0.2)

                # Ganancia
                y = x * self.gain

                # Efecto
                if self.use_reverb:
                    y = self.reverb.process_block(y)

                # Limitar para evitar clipping bruto
                y = np.clip(y, -1.0, 1.0).astype(np.float32)

                peak_counter += 1
                if peak_counter >= 20:
                    peak = float(np.max(np.abs(y))) if len(y) else 0.0
                    self.log(f"[worker] block ok | peak={peak:.3f}")
                    peak_counter = 0

                if not self.output_queue.full():
                    self.output_queue.put_nowait(y)

            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"[worker error] {e}")

        self.log("[worker] stopped")

    def start(self):
        self.running = True

        self.log(
            f"Starting audio engine | in={self.input_device} out={self.output_device} "
            f"sr={self.sample_rate} blocksize={self.blocksize} gain={self.gain} reverb={self.use_reverb}"
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
        self.log("Streams started. Press Ctrl+C to stop.")

    def stop(self):
        self.running = False

        try:
            if self.input_stream is not None:
                self.input_stream.stop()
                self.input_stream.close()
        except Exception as e:
            self.log(f"[stop input error] {e}")

        try:
            if self.output_stream is not None:
                self.output_stream.stop()
                self.output_stream.close()
        except Exception as e:
            self.log(f"[stop output error] {e}")

        self.log("Engine stopped.")


_GLOBAL_ENGINE = None

def list_devices():
    print("\nAvailable audio devices:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(
            f"{i}: {dev['name']} | "
            f"in={dev['max_input_channels']} out={dev['max_output_channels']} | "
            f"default_sr={dev['default_samplerate']}"
        )
    print("")


def main():
    parser = argparse.ArgumentParser(description="Simple realtime mic -> output bypass with gain and delay/reverb.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--blocksize", type=int, default=1024, help="Audio block size")
    parser.add_argument("--gain", type=float, default=1.2, help="Gain multiplier")
    parser.add_argument("--reverb", action="store_true", help="Enable simple delay/reverb")
    parser.add_argument("--delay-ms", type=float, default=120.0, help="Delay time in ms")
    parser.add_argument("--feedback", type=float, default=0.25, help="Delay feedback 0.0 to 0.95")
    parser.add_argument("--wet", type=float, default=0.25, help="Wet mix 0.0 to 1.0")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.input_device is None or args.output_device is None:
        print("Error: you must provide --input-device and --output-device")
        print("Use --list-devices first.")
        sys.exit(1)

    try:
        in_info = sd.query_devices(args.input_device)
        out_info = sd.query_devices(args.output_device)

        print("Input device:", in_info["name"])
        print("Output device:", out_info["name"])
        print(f"Requested sample rate: {args.sample_rate}")
        print("")

        engine = AudioBypassEngine(
            input_device=args.input_device,
            output_device=args.output_device,
            sample_rate=args.sample_rate,
            blocksize=args.blocksize,
            gain=args.gain,
            use_reverb=args.reverb,
            delay_ms=args.delay_ms,
            feedback=args.feedback,
            wet=args.wet,
        )

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

#!/usr/bin/env python3
"""Drive Force Touch trackpad vibration from pulses or an MSIG1 signal."""

import argparse
import math
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LIB_ROOT = REPO_ROOT / "lib"
for _p in reversed((LIB_ROOT, REPO_ROOT)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.bootstrap import maybe_reexec_venv

maybe_reexec_venv(__file__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map MSIG1 signal strength to Force Touch trackpad vibration, or pulse without stdin."
    )
    parser.add_argument("--pulse", type=int, help="Ignore stdin and vibrate N times.")
    parser.add_argument("--on-time", type=float, default=0.2, help="Seconds on per pulse (default: 0.2).")
    parser.add_argument("--off-time", type=float, default=0.2, help="Seconds between pulses (default: 0.2).")
    parser.add_argument("--frequency", type=float, default=200.0, help="Vibration carrier frequency, 100..4000 Hz (default: 200).")
    parser.add_argument("--level", type=float, default=1.0, help="Maximum vibration strength, 0..1 (default: 1).")
    parser.add_argument("--gain", type=float, default=1.0, help="Input RMS gain before clipping to 0..1 (default: 1).")
    parser.add_argument("--send-hz", type=float, default=50.0, help="Vibration update rate, 10..200 Hz (default: 50).")
    parser.add_argument("--raw", action="store_true", help="Read raw little-endian float32 input (requires --rate).")
    parser.add_argument("--rate", type=float, help="Sample rate in Hz for --raw input.")
    args = parser.parse_args(argv)
    for name in ("on_time", "off_time", "frequency", "level", "gain", "send_hz", "rate"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.pulse is not None and args.pulse < 0:
        parser.error("--pulse must be >= 0")
    if args.on_time < 0 or args.off_time < 0:
        parser.error("--on-time and --off-time must be >= 0")
    if not 100 <= args.frequency <= 4000:
        parser.error("--frequency must be between 100 and 4000 Hz")
    if not 0 <= args.level <= 1:
        parser.error("--level must be between 0 and 1")
    if args.gain < 0:
        parser.error("--gain must be >= 0")
    if not 10 <= args.send_hz <= 200:
        parser.error("--send-hz must be between 10 and 200 Hz")
    if args.rate is not None and args.rate <= 0:
        parser.error("--rate must be > 0")
    if args.raw and args.rate is None:
        parser.error("--raw requires --rate")
    return args


def signal_frames(reader, send_hz: float):
    """Yield (RMS, seconds) windows independent of pipe read boundaries."""
    import numpy as np

    size = max(1, int(round(reader.sample_rate / send_hz)))
    pending = np.empty(0, dtype=np.float32)
    for chunk in reader.iter_chunks(chunk_bytes=size * 4):
        pending = np.concatenate((pending, chunk))
        while pending.size >= size:
            frame, pending = pending[:size], pending[size:]
            clean = np.where(np.isfinite(frame), frame, 0).astype(np.float64)
            yield float(np.sqrt(np.mean(clean * clean))), size / reader.sample_rate
    if pending.size:
        clean = np.where(np.isfinite(pending), pending, 0).astype(np.float64)
        yield float(np.sqrt(np.mean(clean * clean))), pending.size / reader.sample_rate
    if reader.carry:
        raise ValueError("truncated float32 sample at end of input")


def play_for(controller, args, level: float, duration: float, deadline: float) -> float:
    """Pace finite bursts in real time, including for unpaced file input."""
    step = 1.0 / args.send_hz
    # After a stalled source, resume now instead of queuing overdue vibrations.
    deadline = max(deadline, time.monotonic())
    while duration > 1e-9:
        seconds = min(step, duration)
        controller.play(frequency=args.frequency, duration=seconds, level=level)
        deadline += seconds
        time.sleep(max(0.0, deadline - time.monotonic()))
        duration -= seconds
    return deadline


def main() -> int:
    args = parse_args()
    from lib.signal_stream import FloatSignalReader, StreamFormatError, is_tty_stdin
    from lib.trackpad import TrackpadController

    def stop(_sig, _frame):
        # Raising also interrupts a blocked stdin read, so the actuator is closed.
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.pulse == 0:
            return 0
        reader = None
        if args.pulse is None:
            if is_tty_stdin():
                raise ValueError("expects an MSIG1 stream on stdin, or use --pulse N")
            try:
                reader = FloatSignalReader.from_stdin(raw=args.raw, sample_rate=args.rate)
            except EOFError:
                return 0
            if not math.isfinite(reader.sample_rate) or reader.sample_rate <= 0:
                raise ValueError("input sample rate must be finite and > 0")
        with TrackpadController() as controller:
            deadline = time.monotonic()
            if args.pulse is not None:
                for i in range(args.pulse):
                    deadline = play_for(controller, args, args.level, args.on_time, deadline)
                    if i < args.pulse - 1:
                        time.sleep(args.off_time)
                        deadline = time.monotonic()
            else:
                for rms, duration in signal_frames(reader, args.send_hz):
                    level = min(1.0, rms * args.gain) * args.level
                    deadline = play_for(controller, args, level, duration, deadline)
        return 0
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, StreamFormatError, ValueError, OSError) as exc:
        print(f"trackpad-vibration: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())

"""Hardware-free checks: python -m unittest discover -s tests."""

import io
import os
import signal
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ["MSIG_SKIP_REEXEC"] = "1"

import trackpad_vibration as command
from lib.signal_stream import FloatSignalReader
from lib.trackpad import TrackpadController


class FragmentedInput(io.BytesIO):
    def read1(self, size=-1):
        return super().read(min(size, 3))


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CommandTests(unittest.TestCase):
    def reader(self, samples, rate=100, fragmented=False):
        stream = FragmentedInput if fragmented else io.BytesIO
        return FloatSignalReader(stream(np.asarray(samples, dtype="<f4").tobytes()), rate, bytearray())

    def run_command(self, argv, reader=None):
        controller = Mock()
        controller.__enter__ = Mock(return_value=controller)
        controller.__exit__ = Mock(return_value=False)
        clock = Clock()
        events = []
        controller.play.side_effect = lambda **kw: events.append((clock.now, kw))
        with (
            patch.object(command, "parse_args", return_value=command.parse_args(argv)),
            patch("lib.trackpad.TrackpadController", return_value=controller) as factory,
            patch("lib.signal_stream.is_tty_stdin", return_value=False),
            patch.object(FloatSignalReader, "from_stdin", return_value=reader) as read,
            patch.object(command.time, "monotonic", clock.monotonic),
            patch.object(command.time, "sleep", clock.sleep),
        ):
            result = command.main()
        return result, clock.now, events, controller, factory, read

    def test_requested_pulse_timing_ignores_stdin(self):
        result, duration, events, controller, _, read = self.run_command(
            ["--pulse", "15", "--on-time=0.2", "--off-time=0.2"]
        )
        self.assertEqual(result, 0)
        self.assertAlmostEqual(duration, 5.8)
        self.assertEqual(len(events), 150)
        self.assertAlmostEqual(events[10][0], 0.4)
        self.assertAlmostEqual(sum(e[1]["duration"] for e in events), 3.0)
        read.assert_not_called()
        controller.__exit__.assert_called_once()

    def test_zero_pulses_does_not_open_hardware(self):
        result, duration, events, _, factory, read = self.run_command(["--pulse=0"])
        self.assertEqual((result, duration, events), (0, 0, []))
        factory.assert_not_called()
        read.assert_not_called()

    def test_stream_gain_clipping_silence_and_partial_window(self):
        reader = self.reader([0, 0, 0.25, -0.25, 2, -2, 0.5])
        result, duration, events, controller, _, _ = self.run_command(
            ["--gain=2", "--level=0.5"], reader
        )
        self.assertEqual(result, 0)
        self.assertAlmostEqual(duration, 0.07)
        self.assertEqual([e[1]["level"] for e in events], [0, 0.25, 0.5, 0.5])
        self.assertAlmostEqual(events[-1][1]["duration"], 0.01)
        controller.__exit__.assert_called_once()

    def test_windows_do_not_depend_on_fragmented_reads(self):
        samples = [0, 0.25, -0.25, 1, -1, 0.5, 0]
        normal = list(command.signal_frames(self.reader(samples), 50))
        fragmented = list(command.signal_frames(self.reader(samples, fragmented=True), 50))
        self.assertEqual(normal, fragmented)

    def test_nonfinite_samples_are_silent(self):
        frames = list(command.signal_frames(self.reader([np.nan, np.inf, -np.inf, 0]), 50))
        self.assertEqual(frames, [(0, 0.02), (0, 0.02)])

    def test_truncated_sample_rejected(self):
        reader = FloatSignalReader(io.BytesIO(b"\x00\x00\x00"), 100, bytearray())
        with self.assertRaisesRegex(ValueError, "truncated"):
            list(command.signal_frames(reader, 50))

    def test_low_rate_input_uses_bounded_bursts(self):
        result, duration, events, _, _, _ = self.run_command([], self.reader([0.5], rate=2))
        self.assertEqual(result, 0)
        self.assertAlmostEqual(duration, 0.5)
        self.assertTrue(all(0 < e[1]["duration"] <= 0.1 for e in events))

    def test_rejects_invalid_options_before_hardware_access(self):
        for argv in (["--pulse=-1"], ["--level=2"], ["--frequency=0"],
                     ["--on-time=-1"], ["--off-time=inf"], ["--gain=nan"],
                     ["--send-hz=0"], ["--raw"], ["--raw", "--rate=0"]):
            with self.subTest(argv=argv), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    command.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_interrupt_closes_actuator_and_restores_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        controller = Mock()
        controller.__enter__ = Mock(return_value=controller)
        controller.__exit__ = Mock(return_value=False)
        controller.play.side_effect = KeyboardInterrupt
        with (
            patch.object(command, "parse_args", return_value=command.parse_args(["--pulse=1"])),
            patch("lib.trackpad.TrackpadController", return_value=controller),
        ):
            self.assertEqual(command.main(), 0)
        controller.__exit__.assert_called_once()
        self.assertEqual(handlers, {sig: signal.getsignal(sig) for sig in handlers})

    def test_hardware_failure_is_reported_without_traceback(self):
        with (
            patch.object(command, "parse_args", return_value=command.parse_args(["--pulse=1"])),
            patch("lib.trackpad.TrackpadController", side_effect=RuntimeError("no motor")),
            patch("sys.stderr", new=io.StringIO()) as err,
        ):
            self.assertEqual(command.main(), 1)
        self.assertEqual(err.getvalue(), "trackpad-vibration: no motor\n")


class HardwareTests(unittest.TestCase):
    def controller(self):
        controller = TrackpadController.__new__(TrackpadController)
        controller._mt = Mock()
        controller._cf = Mock()
        controller._motor = 1
        controller._devices = 2
        controller._cf.CFDataCreate.return_value = 3
        controller._cf.CFPropertyListCreateWithData.return_value = 4
        controller._mt.MTActuationCreateFromDictionary.return_value = 5
        controller._mt.MTActuationActuate.return_value = 0
        return controller

    def test_play_clamps_strength_and_releases_temporary_objects(self):
        controller = self.controller()
        controller.play(frequency=200, duration=0.02, level=2)
        controller._mt.MTActuationActuate.assert_called_once_with(5, 1, 0, 1, 0)
        self.assertEqual([c.args[0] for c in controller._cf.CFRelease.call_args_list], [3, 4, 5])

    def test_driver_failure_releases_actuation(self):
        controller = self.controller()
        controller._mt.MTActuationActuate.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "status -1"):
            controller.play(frequency=200, duration=0.02, level=0.5)
        controller._cf.CFRelease.assert_called_with(5)

    def test_silence_does_not_actuate(self):
        controller = self.controller()
        controller.play(frequency=200, duration=0.02, level=0)
        controller._mt.MTActuationActuate.assert_not_called()

    def test_close_is_idempotent(self):
        controller = self.controller()
        controller.close()
        controller.close()
        controller._mt.MTActuatorClose.assert_called_once_with(1)
        self.assertEqual([c.args[0] for c in controller._cf.CFRelease.call_args_list], [1, 2])

    def test_open_prefers_builtin_then_falls_back(self):
        controller = self.controller()
        controller._motor = None
        controller._mt.MTDeviceCreateList.return_value = 2
        controller._cf.CFArrayGetCount.return_value = 2
        controller._cf.CFArrayGetValueAtIndex.side_effect = [10, 20]
        controller._built_in = lambda device: device == 20

        def get_id(device, pointer):
            pointer._obj.value = device
            return 0

        controller._mt.MTDeviceGetDeviceID.side_effect = get_id
        controller._mt.MTActuatorCreateFromDeviceID.side_effect = [30, 40]
        controller._mt.MTActuatorOpen.side_effect = [-1, 0]
        controller._open()
        self.assertEqual([c.args[0] for c in controller._mt.MTActuatorCreateFromDeviceID.call_args_list], [20, 10])
        controller._cf.CFRelease.assert_called_once_with(30)
        self.assertEqual(controller._motor, 40)


if __name__ == "__main__":
    unittest.main()

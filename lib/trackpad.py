"""Force Touch actuator bindings, adapted from https://github.com/os6/nospeaker.

Uses the same private MultitouchSupport API and sine waveform dictionary.
See NOSPEAKER_LICENSE for the upstream MIT license.
"""

import ctypes as C
import math
import plistlib
import sys


class TrackpadController:
    """Open the first usable actuator, preferring the built-in trackpad."""

    def __init__(self) -> None:
        self._motor = None
        self._devices = None
        if sys.platform != "darwin":
            raise RuntimeError("trackpad vibration requires macOS and a Force Touch trackpad")
        try:
            self._mt = C.CDLL(
                "/System/Library/PrivateFrameworks/MultitouchSupport.framework/MultitouchSupport"
            )
            self._cf = C.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            self._bind()
        except (OSError, AttributeError) as exc:
            raise RuntimeError("required MultitouchSupport APIs are unavailable on this macOS version") from exc
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _bind(self) -> None:
        signatures = {
            "MTDeviceCreateList": ([], C.c_void_p),
            "MTDeviceGetDeviceID": ([C.c_void_p, C.POINTER(C.c_uint64)], C.c_int32),
            "MTActuatorCreateFromDeviceID": ([C.c_uint64], C.c_void_p),
            "MTActuatorOpen": ([C.c_void_p], C.c_int32),
            "MTActuatorClose": ([C.c_void_p], C.c_int32),
            "MTActuationCreateFromDictionary": ([C.c_void_p, C.c_int32], C.c_void_p),
            "MTActuationActuate": (
                [C.c_void_p, C.c_void_p, C.c_uint32, C.c_float, C.c_float], C.c_int32
            ),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self._mt, name)
            fn.argtypes, fn.restype = args, result
        self._built_in = getattr(self._mt, "MTDeviceIsBuiltIn", None)
        if self._built_in is not None:
            self._built_in.argtypes, self._built_in.restype = [C.c_void_p], C.c_bool
        signatures = {
            "CFArrayGetCount": ([C.c_void_p], C.c_long),
            "CFArrayGetValueAtIndex": ([C.c_void_p, C.c_long], C.c_void_p),
            "CFDataCreate": ([C.c_void_p, C.c_char_p, C.c_long], C.c_void_p),
            "CFPropertyListCreateWithData": (
                [C.c_void_p, C.c_void_p, C.c_ulong, C.c_void_p, C.c_void_p], C.c_void_p
            ),
            "CFRelease": ([C.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self._cf, name)
            fn.argtypes, fn.restype = args, result

    def _open(self) -> None:
        self._devices = self._mt.MTDeviceCreateList()
        if not self._devices:
            raise RuntimeError("no trackpad found; connect a Force Touch trackpad")
        devices = []
        for i in range(self._cf.CFArrayGetCount(self._devices)):
            device = self._cf.CFArrayGetValueAtIndex(self._devices, i)
            if not device:
                continue
            device_id = C.c_uint64()
            if self._mt.MTDeviceGetDeviceID(device, C.byref(device_id)) == 0:
                built_in = bool(self._built_in and self._built_in(device))
                devices.append((built_in, device_id.value))
        for _, device_id in sorted(devices, key=lambda item: not item[0]):
            motor = self._mt.MTActuatorCreateFromDeviceID(device_id)
            if not motor:
                continue
            if self._mt.MTActuatorOpen(motor) == 0:
                self._motor = motor
                return
            self._cf.CFRelease(motor)
        raise RuntimeError("no usable haptic motor found; requires a Force Touch MacBook or Magic Trackpad 2 or newer")

    def play(self, *, frequency: float, duration: float, level: float) -> None:
        """Play a bounded sine burst; it expires even if the input pipe stalls."""
        if not self._motor:
            raise RuntimeError("trackpad actuator is closed")
        if not all(math.isfinite(v) for v in (frequency, duration, level)):
            raise ValueError("vibration parameters must be finite")
        if not 100 <= frequency <= 4000 or not 0 < duration <= 0.1:
            raise ValueError("frequency must be 100..4000 Hz and duration must be 0..0.1 seconds")
        level = max(0.0, min(1.0, level))
        if level == 0:
            return
        ms = duration * 1000.0
        shape = {
            "BaseWaveform": {"Type": "Gaussian", "Amplitude": 0.0, "DurationMS": ms},
            "Tones": [{"Type": "Sine", "Amplitude": 1.0, "DurationMS": ms,
                       "DelayMS": 0.0, "FrequencykHz": frequency / 1000.0}],
        }
        payload = plistlib.dumps(shape)
        data = self._cf.CFDataCreate(None, payload, len(payload))
        if not data:
            raise RuntimeError("could not allocate trackpad waveform data")
        try:
            dictionary = self._cf.CFPropertyListCreateWithData(None, data, 0, None, None)
        finally:
            self._cf.CFRelease(data)
        if not dictionary:
            raise RuntimeError("could not create trackpad waveform dictionary")
        try:
            actuation = self._mt.MTActuationCreateFromDictionary(dictionary, 0)
        finally:
            self._cf.CFRelease(dictionary)
        if not actuation:
            raise RuntimeError("could not create trackpad actuation")
        try:
            result = self._mt.MTActuationActuate(actuation, self._motor, 0, level, 0.0)
            if result != 0:
                raise RuntimeError(f"trackpad actuation failed (status {result})")
        finally:
            self._cf.CFRelease(actuation)

    def close(self) -> None:
        if self._motor:
            self._mt.MTActuatorClose(self._motor)
            self._cf.CFRelease(self._motor)
            self._motor = None
        if self._devices:
            self._cf.CFRelease(self._devices)
            self._devices = None

    def __enter__(self) -> "TrackpadController":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

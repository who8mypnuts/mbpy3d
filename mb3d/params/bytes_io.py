"""Little endian field access on a byte buffer (the ``Rd`` helper)."""

from __future__ import annotations

import struct


class ByteReader:
    def __init__(self, data: bytes):
        self.d = data

    def u8(self, o: int) -> int:
        return self.d[o]

    def i8(self, o: int) -> int:
        return struct.unpack_from("<b", self.d, o)[0]

    def u16(self, o: int) -> int:
        return struct.unpack_from("<H", self.d, o)[0]

    def i16(self, o: int) -> int:
        return struct.unpack_from("<h", self.d, o)[0]

    def i32(self, o: int) -> int:
        return struct.unpack_from("<i", self.d, o)[0]

    def u32(self, o: int) -> int:
        return struct.unpack_from("<I", self.d, o)[0]

    def f32(self, o: int) -> float:
        return struct.unpack_from("<f", self.d, o)[0]

    def f64(self, o: int) -> float:
        return struct.unpack_from("<d", self.d, o)[0]

    def rgb(self, o: int) -> tuple[int, int, int]:
        return (self.d[o], self.d[o + 1], self.d[o + 2])

    def d7b(self, o: int) -> float:
        """``Double7B``: the 7 high bytes of a double."""
        return struct.unpack("<d", b"\0" + self.d[o:o + 7])[0]

    def str(self, o: int, n: int) -> str:
        s = self.d[o:o + n]
        end = s.find(b"\0")
        if end >= 0:
            s = s[:end]
        return s.decode("latin-1").strip()


class ByteWriter:
    def __init__(self, data: bytearray):
        self.d = data

    def u8(self, o: int, v: int) -> None:
        self.d[o] = v & 0xFF

    def u16(self, o: int, v: int) -> None:
        struct.pack_into("<H", self.d, o, v & 0xFFFF)

    def i16(self, o: int, v: int) -> None:
        struct.pack_into("<h", self.d, o, max(-32768, min(32767, int(v))))

    def i32(self, o: int, v: int) -> None:
        struct.pack_into("<i", self.d, o, int(v))

    def u32(self, o: int, v: int) -> None:
        struct.pack_into("<I", self.d, o, v & 0xFFFFFFFF)

    def f32(self, o: int, v: float) -> None:
        struct.pack_into("<f", self.d, o, float(v))

    def f64(self, o: int, v: float) -> None:
        struct.pack_into("<d", self.d, o, float(v))

    def rgb(self, o: int, c) -> None:
        self.d[o:o + 3] = bytes(int(x) & 0xFF for x in c)

    def d7b(self, o: int, v: float) -> None:
        self.d[o:o + 7] = struct.pack("<d", float(v))[1:8]

    def str(self, o: int, n: int, s: str) -> None:
        b = s.encode("latin-1", "replace")[:n]
        self.d[o:o + n] = b + b"\0" * (n - len(b))


def short_float(w: int) -> float:
    """``ShortFloatToSingle``: mantissa (shortint) * 10^(exponent - 1)."""
    m = struct.unpack("<b", bytes([w & 0xFF]))[0]
    e = struct.unpack("<b", bytes([(w >> 8) & 0xFF]))[0]
    e = max(-25, min(25, e))
    return float(m) * 10.0 ** (e - 1)


def to_short_float(v: float) -> int:
    """``SingleToShortFloat``: the closest m * 10^(e - 1) with byte m and e."""
    import math

    if not math.isfinite(v) or abs(v) < 1e-45:
        return 0
    if abs(v) > 1e38:
        return 99 | (38 << 8)
    s, e = float(v), 0
    while abs(s) >= 9.95:
        s *= 0.1
        e += 1
    while abs(s) <= 0.995:
        s *= 10.0
        e -= 1
    m = int(round(s * 10.0))
    return (m & 0xFF) | ((e & 0xFF) << 8)

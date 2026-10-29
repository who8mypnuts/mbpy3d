"""MB3D text parameters: ``Mandelbulb3Dv18{...}`` <-> raw header bytes."""

from __future__ import annotations

import struct

from .scene import ADDON_SIZE, HEADER_SIZE


def three_bytes_to_4_chars(b: bytes) -> str:
    i = b[0] | (b[1] << 8) | (b[2] << 16)
    out = []
    for _ in range(4):
        c = i & 0x3F
        c += 46 if c < 12 else 53 if c < 38 else 59
        out.append(chr(c))
        i >>= 6
    return "".join(out)


def four_chars_to_3_bytes(s: bytes) -> bytes | None:
    if len(s) < 4:
        return None
    i = 0
    for j, c in enumerate(s[:4]):
        if 97 <= c <= 122:
            v = c - 59
        elif 65 <= c <= 90:
            v = c - 53
        elif 46 <= c <= 57:
            v = c - 46
        else:
            return None
        i |= v << (j * 6)
    return bytes([i & 0xFF, (i >> 8) & 0xFF, (i >> 16) & 0xFF])


def find_text_params(data: bytes) -> str | None:
    text = data.decode("latin-1")
    i = text.find("Mandelbulb3Dv")
    return text[i:] if i >= 0 else None


def raw_from_text(text: str) -> tuple[bytes, str]:
    """``GetHeaderFromText``: returns (header + addon bytes, title)."""
    t = text.lstrip()
    open_ = t.find("{")
    if open_ < 0:
        raise ValueError("text parameters: missing '{'")
    num = ""
    for c in t[:open_]:
        if c == "." or c.isdigit():
            num += c
        else:
            num = ""
    try:
        ver = float(num) if num else 14.0
    except ValueError:
        ver = 13.0
    close = t.find("}", open_)
    if close < 0:
        raise ValueError("text parameters: missing '}'")
    body = t[open_ + 1:close].replace("\u2026", "...")
    body_b = bytes(c for c in body.encode("latin-1", "replace") if c > 32 and c != 0x85)
    raw = bytearray()
    for k in range(0, len(body_b) - 3, 4):
        b = four_chars_to_3_bytes(body_b[k:k + 4])
        if b is None:
            raise ValueError(f"text parameters: bad characters {body_b[k:k + 4]!r}")
        raw += b
    if ver < 16.0:
        raise ValueError(f"text parameters of version {ver} are not supported")
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"text parameters too short ({len(raw)} of {HEADER_SIZE} header bytes)")
    w, h = struct.unpack_from("<ii", raw, 4)
    if not (1 <= w < 32768 and 1 <= h < 32768):
        raise ValueError("text parameters: invalid image size")
    raw += bytes(HEADER_SIZE + ADDON_SIZE - len(raw)) if len(raw) < HEADER_SIZE + ADDON_SIZE else b""
    raw = raw[:HEADER_SIZE + ADDON_SIZE]
    rest = t[close + 1:]
    title = "Mandelbulb3D"
    i = rest.find("{Titel: ")
    if i >= 0:
        r = rest[i + 8:]
        end = min([k for k in (r.find("}"), r.find("\n"), r.find("\r")) if k >= 0] or [len(r)])
        title = r[:end][:48]
    return bytes(raw), title


def raw_to_text(raw: bytes, title: str = "Mandelbulb3D") -> str:
    """``MakeTextparas``: the parameters as MB3D text."""
    raw = bytearray(raw) + bytes(max(0, HEADER_SIZE + ADDON_SIZE - len(raw)))
    a = HEADER_SIZE
    raw[a] = 16
    if raw[a + 1] & 3 == 1:
        n = 1
    else:
        n = -1
        for i in range(5, -1, -1):
            if struct.unpack_from("<i", raw, a + 8 + i * 188)[0] != 0:
                n = i
                break
    raw[a + 4] = (n + 1) & 0xFF
    out = ["Mandelbulb3Dv18{\r\n"]
    k = 0
    for g in range(0, HEADER_SIZE, 3):
        out.append(three_bytes_to_4_chars(bytes(raw[g:g + 3])))
        k += 1
        if k % 20 == 0:
            out.append("\r\n")
    k = 0
    groups = ((n + 1) * 188 + 10) // 3
    for i in range(groups):
        g = bytes(raw[a + i * 3 + j] if a + i * 3 + j < len(raw) else 0 for j in range(3))
        out.append(three_bytes_to_4_chars(g))
        k += 1
        if k % 20 == 0:
            out.append("\r\n")
    out.append("}\r\n{Titel: " + title + "}\r\n")
    return "".join(out)

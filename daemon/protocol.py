"""S3Q1 v1: RGB request, planar RGBA mask response.

Request: 'S3Q1', u32 JSON length, JSON, u32 data length, HWC RGB bytes.
Reply: 'S3R1', i32 status, u32 W/H/C/JSON length, JSON, u32 data length, CHW bytes.
Frames are top row first. Status 0 succeeds; errors carry no image payload.
"""

import json
import socket
import struct

PORT = 47823
MAX_JSON = 65536
MAX_REPLY_JSON = 32 * 1024 * 1024
MAX_PIXELS = 16777216
MAX_EDGE = 16384
REPLY = struct.Struct("<4siIIII")


class ProtocolError(ValueError):
    pass


def read_exact(conn, count):
    out = bytearray(count)
    view = memoryview(out)
    offset = 0
    while offset < count:
        n = conn.recv_into(view[offset:])
        if not n:
            raise ConnectionError("Connection closed before a complete packet arrived")
        offset += n
    return out


def dimensions(header):
    w, h = header.get("width"), header.get("height")
    if (
        type(w) is not int
        or type(h) is not int
        or not 0 < w <= MAX_EDGE
        or not 0 < h <= MAX_EDGE
        or w * h > MAX_PIXELS
    ):
        raise ProtocolError("Invalid image dimensions; maximum 16 megapixels")
    return w, h


def read_request(conn):
    magic, n = struct.unpack("<4sI", read_exact(conn, 8))
    if magic != b"S3Q1" or not 0 < n <= MAX_JSON:
        raise ProtocolError("Invalid S3Q1 header")
    try:
        header = json.loads(read_exact(conn, n))
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("Invalid JSON") from exc
    if not isinstance(header, dict):
        raise ProtocolError("JSON header must be an object")
    cmd = header.get("cmd")
    if cmd not in ("infer", "info", "shutdown", "capture_begin", "capture_finish", "capture_abort", "capture_frame"):
        raise ProtocolError("Unsupported command")
    expected = 0
    if cmd == "infer":
        w, h = dimensions(header)
        expected = w * h * 3 * 4
    n = struct.unpack("<I", read_exact(conn, 4))[0]
    if n != expected:
        raise ProtocolError("RGB payload length does not match dimensions")
    return header, read_exact(conn, n)


def pack_reply(status, metadata, w=0, h=0, channels=0, payload=b""):
    body = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(body) > MAX_REPLY_JSON:
        raise ProtocolError("Reply metadata too large")
    return (
        REPLY.pack(b"S3R1", status, w, h, channels, len(body))
        + body
        + struct.pack("<I", len(payload))
        + payload
    )


def request(header, payload=b"", port=PORT, timeout=600):
    body = json.dumps(header, allow_nan=False).encode("utf-8")
    if len(body) > MAX_JSON:
        raise ProtocolError("Request metadata too large")
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        conn.sendall(
            b"S3Q1"
            + struct.pack("<I", len(body))
            + body
            + struct.pack("<I", len(payload))
            + payload
        )
        magic, status, w, h, c, n = REPLY.unpack(read_exact(conn, REPLY.size))
        if magic != b"S3R1" or n > MAX_REPLY_JSON:
            raise ProtocolError("Invalid S3R1 reply")
        metadata = json.loads(read_exact(conn, n))
        size = struct.unpack("<I", read_exact(conn, 4))[0]
        if status:
            if size:
                raise ProtocolError("Error reply contains pixels")
            raise RuntimeError(str(metadata))
        if header.get("cmd") == "infer":
            ew, eh = dimensions(header)
            if (w, h, c, size) != (ew, eh, 4, ew * eh * 16):
                raise ProtocolError("Reply image shape does not match request")
        elif (w, h, c, size) != (0, 0, 0, 0):
            raise ProtocolError("Control reply contains an unexpected image")
        return metadata, read_exact(conn, size)

"""Connection server adapted from MarigoldV2-Nuke (MIT)."""
import os
import socket
import threading
import time
import logging
from protocol import PORT, ProtocolError, dimensions, pack_reply, read_request
LOG = logging.getLogger("sam3")

class Server:
    def __init__(self, engine, port=PORT, idle_timeout=300):
        self.engine = engine
        self.port = port
        self.idle_timeout = idle_timeout
        self.stop = threading.Event()
        self.slots = threading.BoundedSemaphore(8)
        self.last_activity = time.monotonic()
        self.active = 0
        self.state_lock = threading.Lock()

    def handle(self, conn):
        try:
            conn.settimeout(600)
            with conn:
                while not self.stop.is_set():
                    try:
                        header, data = read_request(conn)
                    except (ConnectionError, socket.timeout):
                        return
                    except ProtocolError as exc:
                        conn.sendall(pack_reply(2, {"error": str(exc)}))
                        return  # framing cannot be trusted after an invalid request
                    with self.state_lock:
                        self.active += 1
                    try:
                        cmd = header["cmd"]
                        if cmd == "infer":
                            meta, payload = self.engine.infer(header, data)
                            w, h = dimensions(header)
                            conn.sendall(pack_reply(0, meta, w, h, 4, payload))
                            LOG.info("Frame %dx%d: %.2fs", w, h, meta["elapsed_seconds"])
                        elif cmd == "info":
                            conn.sendall(pack_reply(0, self.engine.info()))
                        elif cmd.startswith('capture_'):
                            conn.sendall(pack_reply(0, self.engine.control(header)))
                        elif cmd.startswith('video_'):
                            conn.sendall(pack_reply(0, self.engine.video.control(header)))
                        else:
                            conn.sendall(pack_reply(0, {"state": "stopping"}))
                            self.stop.set()
                            return
                    except Exception as exc:
                        LOG.exception("Request failed")
                        message = str(exc)[:10000]
                        if "out of memory" in message.lower():
                            message = (
                                "GPU memory exhausted. Reduce inference long edge, close other GPU workloads, "
                                "then restart the engine. Model loading also needs VRAM. " + message
                            )
                        conn.sendall(pack_reply(1, {"error": message}))
                    finally:
                        with self.state_lock:
                            self.active -= 1
                            self.last_activity = time.monotonic()
        except OSError:
            LOG.debug("Client disconnected", exc_info=True)
        finally:
            self.slots.release()

    def serve(self):
        with socket.socket() as listener:
            if os.name == "nt":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", self.port))
            self.port = listener.getsockname()[1]
            listener.listen(8)
            listener.settimeout(0.5)
            LOG.info("Listening on 127.0.0.1:%d", self.port)
            while not self.stop.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    with self.state_lock:
                        idle = (
                            not self.active and not self.engine.video.running()
                            and time.monotonic() - self.last_activity > self.idle_timeout
                        )
                    if self.idle_timeout and idle:
                        return
                    continue
                if not self.slots.acquire(blocking=False):
                    conn.close()
                    continue
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

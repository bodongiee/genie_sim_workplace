# =============================================================================
# H.264 video encoder using PyAV
# =============================================================================

import numpy as np

class H264Encoder:
    def __init__(self, width: int, height: int, fps: int = 30, bitrate: int = 4_000_000, gop_size: int = 15):
        import av
        from fractions import Fraction

        for codec_name in ("h264_nvenc", "libx264"):
            try:
                self._codec = av.codec.Codec(codec_name, "w")
                print(f"[H264Encoder] Using encoder: {codec_name}")
                break
            except Exception:
                continue
        else:
            raise RuntimeError('\033[31m' + "[ERROR] No Codec available" +'\033[0m')

        self._ctx = av.codec.CodecContext.create(self._codec, "w")
        self._ctx.width = width
        self._ctx.height = height
        self._ctx.time_base = Fraction(1, fps)
        self._ctx.framerate = Fraction(fps, 1)
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.bit_rate = bitrate
        self._ctx.gop_size = gop_size

        if "nvenc" in self._codec.name:
            self._ctx.options = {"preset": "p3", "tune": "ull", "zerolatency": "1", "rc": "cbr","repeat_headers": "1",}
        else:
            self._ctx.options = {"preset": "ultrafast", "tune": "zerolatency", }
            self._ctx.flags &= ~av.codec.context.Flags.GLOBAL_HEADER

        self._ctx.open()
        self._frame_id = 0
        self._av = av
        print('\033[42m\033[37m' + f"[H264Encoder] Ready: {width}x{height} @ {fps}fps, " f"{bitrate / 1e6:.1f} Mbps, codec={self._codec.name}" + '\033[0m]')

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value: int):
        self._frame_id = value

    def encode(self, rgba: np.ndarray) -> list[bytes]:
        h, w = rgba.shape[:2]
        frame = self._av.VideoFrame(w, h, "rgba") # RGBA
        frame.planes[0].update(rgba.tobytes())  
        frame.pts = self._frame_id
        self._frame_id += 1

        yuv_frame = frame.reformat(format="yuv420p") # RGBA to YUV
        return [bytes(pkt) for pkt in self._ctx.encode(yuv_frame) if bytes(pkt)] # YUV to H264

    def flush(self) -> list[bytes]:
        return [bytes(pkt) for pkt in self._ctx.encode(None) if bytes(pkt)]

    def close(self):
        try:
            self.flush()
        except Exception:
            pass
        self._ctx.close()

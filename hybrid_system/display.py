from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class DisplayEvent:
    kind: str
    key: int | None = None


class GLFWDisplay:
    def __init__(
        self,
        title: str,
        size: tuple[int, int],
        fullscreen: bool = True,
        fps: float = 30.0,
    ) -> None:
        import glfw

        self._glfw = glfw
        self._running = False
        self._size = size
        self._fps = fps

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        monitor = glfw.get_primary_monitor() if fullscreen else None
        if monitor is not None:
            mode = glfw.get_video_mode(monitor)
            win_w, win_h = mode.size.width, mode.size.height
        else:
            win_w, win_h = size

        self._window = glfw.create_window(win_w, win_h, title, monitor, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_window_close_callback(self._window, self._on_close)
        if fullscreen:
            glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

        self._program = self._build_program()
        self._vao, self._vbo = self._build_quad()
        self._texture = self._build_texture(size[0], size[1])
        from OpenGL.GL import glGetUniformLocation
        self._u_tex = glGetUniformLocation(self._program, b"uTex")
        self._running = True

    def _shader_src(self) -> tuple[str, str]:
        vertex = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTex;
        out vec2 vTex;
        void main() {
            vTex = aTex;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
        """
        fragment = """
        #version 330 core
        in vec2 vTex;
        uniform sampler2D uTex;
        out vec4 FragColor;
        void main() {
            FragColor = texture(uTex, vTex);
        }
        """
        return vertex, fragment

    def _build_program(self) -> int:
        from OpenGL.GL import GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_VERTEX_SHADER
        from OpenGL.GL import glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog, glGetShaderiv
        from OpenGL.GL.shaders import compileProgram, compileShader

        vertex, fragment = self._shader_src()
        vs = compileShader(vertex, GL_VERTEX_SHADER)
        fs = compileShader(fragment, GL_FRAGMENT_SHADER)
        prog = compileProgram(vs, fs)
        return prog

    def _build_quad(self) -> tuple[int, int]:
        from OpenGL.GL import (
            GL_ARRAY_BUFFER,
            GL_FLOAT,
            GL_STATIC_DRAW,
            glBindBuffer,
            glBindVertexArray,
            glBufferData,
            glEnableVertexAttribArray,
            glGenBuffers,
            glGenVertexArrays,
            glVertexAttribPointer,
        )
        import ctypes as _ct

        vertices = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                -1.0, -1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                -1.0, 1.0, 0.0, 1.0,
            ],
            dtype=np.float32,
        )
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 16, _ct.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, 16, _ct.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao, vbo

    def _build_texture(self, width: int, height: int) -> int:
        from OpenGL.GL import (
            GL_CLAMP_TO_EDGE,
            GL_LINEAR,
            GL_RGB,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T,
            GL_UNSIGNED_BYTE,
            GL_BGR,
            glBindTexture,
            glGenTextures,
            glTexImage2D,
            glTexParameteri,
        )

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            width,
            height,
            0,
            GL_BGR,
            GL_UNSIGNED_BYTE,
            None,
        )
        return tex

    def _on_key(self, window, key, scancode, action, mods) -> None:
        import glfw

        if action == glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)
            self._running = False

    def _on_close(self, window) -> None:
        self._running = False

    def show(self, frame: np.ndarray) -> None:
        if not self._running:
            return
        if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_LINEAR)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        from OpenGL.GL import (
            GL_COLOR_BUFFER_BIT,
            GL_BGR,
            GL_TEXTURE_2D,
            GL_TEXTURE0,
            GL_UNSIGNED_BYTE,
            glActiveTexture,
            glBindTexture,
            glBindVertexArray,
            glClear,
            glDrawArrays,
            glTexSubImage2D,
            glUseProgram,
            glUniform1i,
        )

        glUseProgram(self._program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._texture)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            frame.shape[1],
            frame.shape[0],
            GL_BGR,
            GL_UNSIGNED_BYTE,
            frame,
        )
        glUniform1i(self._u_tex, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindVertexArray(self._vao)
        glDrawArrays(4, 0, 6)
        self._glfw.swap_buffers(self._window)

    def pump(self, fps: int = 60) -> bool:
        self._glfw.poll_events()
        if self._glfw.window_should_close(self._window):
            self._running = False
        return self._running

    def close(self) -> None:
        self._running = False
        try:
            from OpenGL.GL import (
                glDeleteBuffers,
                glDeleteProgram,
                glDeleteTextures,
                glDeleteVertexArrays,
            )

            glDeleteTextures([self._texture])
            glDeleteBuffers([self._vbo])
            glDeleteVertexArrays([self._vao])
            glDeleteProgram(self._program)
        except Exception:
            pass
        try:
            self._glfw.destroy_window(self._window)
        except Exception:
            pass
        try:
            self._glfw.terminate()
        except Exception:
            pass


class SDL2Display:
    SDL_INIT_VIDEO = 0x20
    SDL_WINDOW_SHOWN = 0x00000004
    SDL_WINDOW_FULLSCREEN_DESKTOP = 0x00001000 | 0x00000001
    SDL_RENDERER_ACCELERATED = 0x00000002
    SDL_RENDERER_PRESENTVSYNC = 0x00000004
    SDL_TEXTUREACCESS_STREAMING = 1
    SDL_PIXELFORMAT_RGB24 = 0x17101803
    SDL_QUIT = 0x100

    def __init__(self, title: str, size: tuple[int, int], fullscreen: bool = True, fps: float = 30.0) -> None:
        lib_name = ctypes.util.find_library("SDL2") or "libSDL2-2.0.so.0"
        self._lib = ctypes.CDLL(lib_name)
        self._load_api()
        self._running = False
        self._size = size
        self._fps = fps

        if self._lib.SDL_Init(self.SDL_INIT_VIDEO) != 0:
            raise RuntimeError(self._get_error())

        flags = self.SDL_WINDOW_SHOWN
        if fullscreen:
            flags |= self.SDL_WINDOW_FULLSCREEN_DESKTOP

        self._window = self._lib.SDL_CreateWindow(
            title.encode("utf-8"),
            self._window_pos_undefined(),
            self._window_pos_undefined(),
            size[0],
            size[1],
            flags,
        )
        if not self._window:
            raise RuntimeError(self._get_error())

        self._renderer = self._lib.SDL_CreateRenderer(
            self._window,
            -1,
            self.SDL_RENDERER_ACCELERATED | self.SDL_RENDERER_PRESENTVSYNC,
        )
        if not self._renderer:
            raise RuntimeError(self._get_error())

        self._texture = self._lib.SDL_CreateTexture(
            self._renderer,
            self.SDL_PIXELFORMAT_RGB24,
            self.SDL_TEXTUREACCESS_STREAMING,
            size[0],
            size[1],
        )
        if not self._texture:
            raise RuntimeError(self._get_error())

        if fullscreen:
            self._lib.SDL_ShowCursor(0)

        self._running = True
        self._event_buf = ctypes.create_string_buffer(64)

    def _load_api(self) -> None:
        lib = self._lib
        lib.SDL_GetError.restype = ctypes.c_char_p
        lib.SDL_Init.argtypes = [ctypes.c_uint32]
        lib.SDL_Init.restype = ctypes.c_int
        lib.SDL_Quit.argtypes = []
        lib.SDL_Quit.restype = None
        lib.SDL_CreateWindow.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
        lib.SDL_CreateWindow.restype = ctypes.c_void_p
        lib.SDL_DestroyWindow.argtypes = [ctypes.c_void_p]
        lib.SDL_DestroyWindow.restype = None
        lib.SDL_CreateRenderer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
        lib.SDL_CreateRenderer.restype = ctypes.c_void_p
        lib.SDL_DestroyRenderer.argtypes = [ctypes.c_void_p]
        lib.SDL_DestroyRenderer.restype = None
        lib.SDL_CreateTexture.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.SDL_CreateTexture.restype = ctypes.c_void_p
        lib.SDL_DestroyTexture.argtypes = [ctypes.c_void_p]
        lib.SDL_DestroyTexture.restype = None
        lib.SDL_UpdateTexture.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.SDL_UpdateTexture.restype = ctypes.c_int
        lib.SDL_RenderClear.argtypes = [ctypes.c_void_p]
        lib.SDL_RenderClear.restype = ctypes.c_int
        lib.SDL_RenderCopy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.SDL_RenderCopy.restype = ctypes.c_int
        lib.SDL_RenderPresent.argtypes = [ctypes.c_void_p]
        lib.SDL_RenderPresent.restype = None
        lib.SDL_PollEvent.argtypes = [ctypes.c_void_p]
        lib.SDL_PollEvent.restype = ctypes.c_int
        lib.SDL_SetWindowFullscreen.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.SDL_SetWindowFullscreen.restype = ctypes.c_int
        lib.SDL_ShowCursor.argtypes = [ctypes.c_int]
        lib.SDL_ShowCursor.restype = ctypes.c_int
        lib.SDL_DestroyTexture.argtypes = [ctypes.c_void_p]
        lib.SDL_DestroyTexture.restype = None
        lib.SDL_Delay.argtypes = [ctypes.c_uint32]
        lib.SDL_Delay.restype = None

    @staticmethod
    def _window_pos_undefined() -> int:
        return 0x1FFF0000

    def _get_error(self) -> str:
        err = self._lib.SDL_GetError()
        return err.decode("utf-8", errors="ignore") if err else "SDL error"

    def show(self, frame: np.ndarray) -> None:
        if not self._running:
            return
        if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        pitch = frame.shape[1] * 3
        ptr = frame.ctypes.data_as(ctypes.c_void_p)
        if self._lib.SDL_UpdateTexture(self._texture, None, ptr, pitch) != 0:
            self._running = False
            return
        self._lib.SDL_RenderClear(self._renderer)
        self._lib.SDL_RenderCopy(self._renderer, self._texture, None, None)
        self._lib.SDL_RenderPresent(self._renderer)

    def pump(self, fps: int = 60) -> bool:
        while self._lib.SDL_PollEvent(self._event_buf):
            event_type = ctypes.c_uint32.from_buffer(self._event_buf).value
            if event_type == self.SDL_QUIT:
                self._running = False
                break
        return self._running

    def close(self) -> None:
        self._running = False
        try:
            if getattr(self, "_texture", None):
                self._lib.SDL_DestroyTexture(self._texture)
        except Exception:
            pass
        try:
            if getattr(self, "_renderer", None):
                self._lib.SDL_DestroyRenderer(self._renderer)
        except Exception:
            pass
        try:
            if getattr(self, "_window", None):
                self._lib.SDL_DestroyWindow(self._window)
        except Exception:
            pass
        try:
            self._lib.SDL_Quit()
        except Exception:
            pass


class FFplayDisplay:
    def __init__(
        self,
        title: str,
        size: tuple[int, int],
        fullscreen: bool = True,
        fps: float = 30.0,
    ) -> None:
        if shutil.which("ffplay") is None:
            raise RuntimeError("ffplay is not available")

        self._title = title
        self._size = size
        self._fullscreen = fullscreen
        self._running = True
        cmd = [
                "ffplay",
                "-loglevel",
                "error",
                "-hide_banner",
                "-nostdin",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-framedrop",
                "-sync",
                "video",
                "-an",
                "-sn",
                "-dn",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{size[0]}x{size[1]}",
                "-framerate",
                f"{max(1.0, float(fps)):.3f}",
                "-window_title",
                title,
                "-autoexit",
        ]
        if fullscreen:
            cmd.append("-fs")
        cmd.append("pipe:0")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        time.sleep(0.2)

    def show(self, frame: np.ndarray) -> None:
        if not self._running or self._proc.poll() is not None:
            self._running = False
            return
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(memoryview(frame).cast("B"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._running = False

    def pump(self, fps: int = 60) -> bool:
        if self._proc.poll() is not None:
            self._running = False
        return self._running

    def close(self) -> None:
        self._running = False
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


class OpenCVDisplay:
    def __init__(
        self,
        title: str,
        size: tuple[int, int],
        fullscreen: bool = True,
        fps: float = 30.0,
    ) -> None:
        self._title = title
        self._size = size
        self._fullscreen = fullscreen
        self._fps = fps
        self._fullscreen_applied = False
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.moveWindow(self._title, 0, 0)
            cv2.setWindowProperty(self._title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self._title, size[0], size[1])

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self._title, frame)
        if self._fullscreen and not self._fullscreen_applied:
            cv2.moveWindow(self._title, 0, 0)
            cv2.setWindowProperty(self._title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self._fullscreen_applied = True

    def pump(self, fps: int = 60) -> bool:
        delay = max(1, int(1000 / max(1, fps)))
        key = cv2.waitKey(delay) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._title)
        except Exception:
            pass


def create_display(
    title: str,
    size: tuple[int, int],
    fullscreen: bool = True,
    fps: float = 30.0,
    backend: str = "auto",
) -> Any:
    backend = (backend or "auto").lower()
    if backend in ("auto", "sdl2"):
        try:
            return SDL2Display(title, size, fullscreen=fullscreen, fps=fps)
        except Exception:
            if backend != "auto":
                raise
    if backend in ("auto", "glfw", "opengl", "egl"):
        try:
            return GLFWDisplay(title, size, fullscreen=fullscreen, fps=fps)
        except Exception:
            if backend not in ("auto", "egl"):
                raise
    if backend in ("auto", "ffplay") and shutil.which("ffplay") is not None:
        return FFplayDisplay(title, size, fullscreen=fullscreen, fps=fps)
    if backend in ("auto", "opencv"):
        return OpenCVDisplay(title, size, fullscreen=fullscreen, fps=fps)
    raise ValueError(f"Unsupported display backend: {backend}")

"""Off-screen OpenGL rendering backends supporting both ModernGL and PyOpenGL."""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RenderBackend(ABC):
    """Abstract off-screen OpenGL 3.3 Core rendering backend."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Identifier of the active OpenGL binding ('moderngl' or 'pyopengl')."""

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return GPU vendor, renderer, OpenGL version, and backend metadata."""

    @abstractmethod
    def clear(
        self,
        r: float = 0.0,
        g: float = 0.0,
        b: float = 0.0,
        a: float = 1.0,
        depth: float = 1.0,
    ) -> None:
        """Clear the off-screen color and depth buffers."""

    @abstractmethod
    def set_state(
        self,
        depth_test: bool = False,
        depth_write: bool = True,
        blend: bool = False,
    ) -> None:
        """Configure OpenGL pipeline state for depth testing and alpha blending."""

    @abstractmethod
    def draw_arrays(
        self,
        vertex_shader: str,
        fragment_shader: str,
        attributes: List[Tuple[str, int, str]],
        vertex_data: np.ndarray,
        uniforms: Optional[Dict[str, Any]] = None,
        textures: Optional[Dict[str, Tuple[np.ndarray, str]]] = None,
        indices: Optional[np.ndarray] = None,
        instance_attributes: Optional[List[Tuple[str, int, str]]] = None,
        instance_data: Optional[np.ndarray] = None,
        instance_count: int = 1,
    ) -> None:
        """Compile shaders, bind buffers/uniforms/textures, and issue an off-screen draw call.

        Args:
            vertex_shader: GLSL 330 core vertex shader source.
            fragment_shader: GLSL 330 core fragment shader source.
            attributes: Per-vertex layout list of (name, num_floats, moderngl_fmt).
            vertex_data: Contiguous float32 array of per-vertex attributes.
            uniforms: Mapping of uniform name to scalar, tuple, or float32 matrix/vector.
            textures: Mapping of sampler uniform name to (rgba_uint8_array, filter_mode).
            indices: Optional uint32 index buffer for indexed rendering (IBO/EBO).
            instance_attributes: Optional per-instance layout list of (name, num_floats, fmt).
            instance_data: Optional contiguous float32 array of per-instance attributes.
            instance_count: Number of instances to render when using instanced rendering.
        """

    @abstractmethod
    def finish(self) -> None:
        """Block until all queued GPU commands in the current pipeline complete."""

    @abstractmethod
    def read_pixels(self) -> np.ndarray:
        """Read the off-screen color buffer as a top-down (H, W, 4) uint8 RGBA array."""

    @abstractmethod
    def release(self) -> None:
        """Release FBO and OpenGL context resources."""


class ModernGLBackend(RenderBackend):
    """Off-screen OpenGL backend powered by ModernGL standalone context + FBO."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        import moderngl

        self._mgl = moderngl
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception:
            # Fallback for Linux headless environments using EGL
            self.ctx = moderngl.create_standalone_context(backend="egl", require=330)

        self.color_tex = self.ctx.texture((width, height), components=4, dtype="f1")
        self.depth_rb = self.ctx.depth_renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex],
            depth_attachment=self.depth_rb,
        )
        self.fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self._program_cache: Dict[Tuple[str, str], Any] = {}

    @property
    def backend_name(self) -> str:
        return "moderngl"

    def get_device_info(self) -> Dict[str, Any]:
        info = self.ctx.info
        return {
            "backend": "moderngl",
            "backend_version": getattr(self._mgl, "__version__", "unknown"),
            "gl_vendor": info.get("GL_VENDOR", "Unknown"),
            "gl_renderer": info.get("GL_RENDERER", "Unknown"),
            "gl_version": info.get("GL_VERSION", "Unknown"),
            "resolution": f"{self.width}x{self.height}",
        }

    def clear(
        self,
        r: float = 0.0,
        g: float = 0.0,
        b: float = 0.0,
        a: float = 1.0,
        depth: float = 1.0,
    ) -> None:
        self.fbo.use()
        self.fbo.depth_mask = True
        self.fbo.clear(red=r, green=g, blue=b, alpha=a, depth=depth)

    def set_state(
        self,
        depth_test: bool = False,
        depth_write: bool = True,
        blend: bool = False,
    ) -> None:
        flags = self._mgl.NOTHING
        if depth_test:
            flags |= self._mgl.DEPTH_TEST
        if blend:
            flags |= self._mgl.BLEND
        self.ctx.enable_only(flags)
        if blend:
            self.ctx.blend_func = (self._mgl.SRC_ALPHA, self._mgl.ONE_MINUS_SRC_ALPHA)
        self.fbo.depth_mask = bool(depth_write)

    def draw_arrays(
        self,
        vertex_shader: str,
        fragment_shader: str,
        attributes: List[Tuple[str, int, str]],
        vertex_data: np.ndarray,
        uniforms: Optional[Dict[str, Any]] = None,
        textures: Optional[Dict[str, Tuple[np.ndarray, str]]] = None,
        indices: Optional[np.ndarray] = None,
        instance_attributes: Optional[List[Tuple[str, int, str]]] = None,
        instance_data: Optional[np.ndarray] = None,
        instance_count: int = 1,
    ) -> None:
        self.fbo.use()
        cache_key = (vertex_shader, fragment_shader)
        prog = self._program_cache.get(cache_key)
        if prog is None:
            prog = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader,
            )
            self._program_cache[cache_key] = prog

        vbo_bytes = np.ascontiguousarray(vertex_data, dtype=np.float32).tobytes()
        vbo = self.ctx.buffer(vbo_bytes)
        fmt_str = " ".join(attr[2] for attr in attributes)
        attr_names = [attr[0] for attr in attributes]
        vao_content = [(vbo, fmt_str, *attr_names)]

        inst_vbo = None
        if instance_attributes and instance_data is not None:
            inst_bytes = np.ascontiguousarray(instance_data, dtype=np.float32).tobytes()
            inst_vbo = self.ctx.buffer(inst_bytes)
            inst_fmt = " ".join(attr[2] for attr in instance_attributes) + "/i"
            inst_names = [attr[0] for attr in instance_attributes]
            vao_content.append((inst_vbo, inst_fmt, *inst_names))

        ibo = None
        if indices is not None:
            ibo_bytes = np.ascontiguousarray(indices, dtype=np.uint32).tobytes()
            ibo = self.ctx.buffer(ibo_bytes)

        vao = self.ctx.vertex_array(prog, vao_content, index_buffer=ibo)

        created_textures = []
        if textures:
            for unit, (sampler_name, (tex_arr, filter_mode)) in enumerate(textures.items()):
                tex_rgba = np.ascontiguousarray(tex_arr, dtype=np.uint8)
                h, w, c = tex_rgba.shape
                tex = self.ctx.texture((w, h), c, tex_rgba.tobytes())
                if filter_mode == "nearest":
                    tex.filter = (self._mgl.NEAREST, self._mgl.NEAREST)
                else:
                    tex.filter = (self._mgl.LINEAR, self._mgl.LINEAR)
                tex.repeat_x = True
                tex.repeat_y = True
                tex.use(location=unit)
                if sampler_name in prog:
                    prog[sampler_name].value = unit
                created_textures.append(tex)

        if uniforms:
            for u_name, u_val in uniforms.items():
                if u_name not in prog:
                    continue
                if isinstance(u_val, np.ndarray):
                    prog[u_name].write(np.ascontiguousarray(u_val, dtype=np.float32).tobytes())
                elif isinstance(u_val, (tuple, list)):
                    prog[u_name].value = tuple(u_val)
                else:
                    prog[u_name].value = u_val

        vao.render(mode=self._mgl.TRIANGLES, instances=instance_count)

        # Clean up per-draw GPU buffer/texture objects; cached shader programs are freed in release()
        vao.release()
        vbo.release()
        if inst_vbo is not None:
            inst_vbo.release()
        if ibo is not None:
            ibo.release()
        for tex in created_textures:
            tex.release()

    def finish(self) -> None:
        self.ctx.finish()

    def read_pixels(self) -> np.ndarray:
        raw = self.fbo.read(components=4, dtype="f1")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 4))
        # Flip vertically from OpenGL bottom-left origin to top-left image origin
        return np.flipud(arr).copy()

    def release(self) -> None:
        for prog in self._program_cache.values():
            try:
                prog.release()
            except Exception:
                pass
        self._program_cache.clear()
        self.fbo.release()
        self.color_tex.release()
        self.depth_rb.release()
        self.ctx.release()


class PyOpenGLBackend(RenderBackend):
    """Off-screen OpenGL backend powered by PyOpenGL (OpenGL.GL) + standalone glcontext."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        import glcontext
        import OpenGL
        from OpenGL import GL

        self._gl = GL
        self._opengl_version = getattr(OpenGL, "__version__", "unknown")
        self._ctx = glcontext.default_backend()(mode="standalone", glversion=330)

        # Create off-screen FBO with RGBA8 color texture and DEPTH24 renderbuffer
        self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

        self._color_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self._color_tex,
            0,
        )

        self._depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._depth_rb)
        GL.glRenderbufferStorage(
            GL.GL_RENDERBUFFER,
            GL.GL_DEPTH_COMPONENT24,
            width,
            height,
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER,
            self._depth_rb,
        )

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"PyOpenGL FBO incomplete: status=0x{status:04X}")

        GL.glViewport(0, 0, width, height)
        self._program_cache: Dict[Tuple[str, str], int] = {}

    @property
    def backend_name(self) -> str:
        return "pyopengl"

    def _get_gl_string(self, enum_val: int) -> str:
        raw = self._gl.glGetString(enum_val)
        if raw is None:
            return "Unknown"
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def get_device_info(self) -> Dict[str, Any]:
        GL = self._gl
        return {
            "backend": "pyopengl",
            "backend_version": self._opengl_version,
            "gl_vendor": self._get_gl_string(GL.GL_VENDOR),
            "gl_renderer": self._get_gl_string(GL.GL_RENDERER),
            "gl_version": self._get_gl_string(GL.GL_VERSION),
            "resolution": f"{self.width}x{self.height}",
        }

    def clear(
        self,
        r: float = 0.0,
        g: float = 0.0,
        b: float = 0.0,
        a: float = 1.0,
        depth: float = 1.0,
    ) -> None:
        GL = self._gl
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glClearColor(r, g, b, a)
        GL.glClearDepth(depth)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    def set_state(
        self,
        depth_test: bool = False,
        depth_write: bool = True,
        blend: bool = False,
    ) -> None:
        GL = self._gl
        if depth_test:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LESS)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glDepthMask(GL.GL_TRUE if depth_write else GL.GL_FALSE)

        if blend:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        else:
            GL.glDisable(GL.GL_BLEND)

    def _compile_shader(self, source: str, shader_type: int) -> int:
        GL = self._gl
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        if not ok:
            log = GL.glGetShaderInfoLog(shader)
            if isinstance(log, bytes):
                log = log.decode("utf-8", errors="replace")
            raise RuntimeError(f"Shader compilation failed:\n{log}")
        return shader

    def draw_arrays(
        self,
        vertex_shader: str,
        fragment_shader: str,
        attributes: List[Tuple[str, int, str]],
        vertex_data: np.ndarray,
        uniforms: Optional[Dict[str, Any]] = None,
        textures: Optional[Dict[str, Tuple[np.ndarray, str]]] = None,
        indices: Optional[np.ndarray] = None,
        instance_attributes: Optional[List[Tuple[str, int, str]]] = None,
        instance_data: Optional[np.ndarray] = None,
        instance_count: int = 1,
    ) -> None:
        GL = self._gl
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

        cache_key = (vertex_shader, fragment_shader)
        prog = self._program_cache.get(cache_key)
        if prog is None:
            vs = self._compile_shader(vertex_shader, GL.GL_VERTEX_SHADER)
            fs = self._compile_shader(fragment_shader, GL.GL_FRAGMENT_SHADER)
            prog = GL.glCreateProgram()
            GL.glAttachShader(prog, vs)
            GL.glAttachShader(prog, fs)
            GL.glLinkProgram(prog)
            if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
                log = GL.glGetProgramInfoLog(prog)
                if isinstance(log, bytes):
                    log = log.decode("utf-8", errors="replace")
                raise RuntimeError(f"Program link failed:\n{log}")
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)
            self._program_cache[cache_key] = prog
        GL.glUseProgram(prog)

        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)

        vbo_arr = np.ascontiguousarray(vertex_data, dtype=np.float32)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vbo_arr.nbytes, vbo_arr, GL.GL_STATIC_DRAW)

        total_floats = sum(attr[1] for attr in attributes)
        stride = total_floats * 4
        offset = 0
        for attr_name, count, _ in attributes:
            loc = GL.glGetAttribLocation(prog, attr_name)
            if loc >= 0:
                GL.glEnableVertexAttribArray(loc)
                GL.glVertexAttribPointer(
                    loc,
                    count,
                    GL.GL_FLOAT,
                    GL.GL_FALSE,
                    stride,
                    ctypes.c_void_p(offset),
                )
            offset += count * 4

        inst_vbo = None
        if instance_attributes and instance_data is not None:
            inst_arr = np.ascontiguousarray(instance_data, dtype=np.float32)
            inst_vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, inst_vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, inst_arr.nbytes, inst_arr, GL.GL_STATIC_DRAW
            )
            inst_total_floats = sum(attr[1] for attr in instance_attributes)
            inst_stride = inst_total_floats * 4
            inst_offset = 0
            for attr_name, count, _ in instance_attributes:
                loc = GL.glGetAttribLocation(prog, attr_name)
                if loc >= 0:
                    GL.glEnableVertexAttribArray(loc)
                    GL.glVertexAttribPointer(
                        loc,
                        count,
                        GL.GL_FLOAT,
                        GL.GL_FALSE,
                        inst_stride,
                        ctypes.c_void_p(inst_offset),
                    )
                    GL.glVertexAttribDivisor(loc, 1)
                inst_offset += count * 4

        ibo = None
        num_indices = 0
        if indices is not None:
            idx_arr = np.ascontiguousarray(indices, dtype=np.uint32)
            num_indices = idx_arr.size
            ibo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER, idx_arr.nbytes, idx_arr, GL.GL_STATIC_DRAW
            )

        created_textures = []
        if textures:
            for unit, (sampler_name, (tex_arr, filter_mode)) in enumerate(textures.items()):
                tex_rgba = np.ascontiguousarray(tex_arr, dtype=np.uint8)
                h, w, _ = tex_rgba.shape
                tex_id = GL.glGenTextures(1)
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA8,
                    w,
                    h,
                    0,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    tex_rgba,
                )
                gl_filter = GL.GL_NEAREST if filter_mode == "nearest" else GL.GL_LINEAR
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, gl_filter)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, gl_filter)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
                loc = GL.glGetUniformLocation(prog, sampler_name)
                if loc >= 0:
                    GL.glUniform1i(loc, unit)
                created_textures.append(tex_id)

        if uniforms:
            for u_name, u_val in uniforms.items():
                loc = GL.glGetUniformLocation(prog, u_name)
                if loc < 0:
                    continue
                if isinstance(u_val, np.ndarray):
                    arr = np.ascontiguousarray(u_val, dtype=np.float32)
                    if arr.shape == (4, 4) or arr.size == 16:
                        GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, arr)
                    elif arr.size == 4:
                        GL.glUniform4fv(loc, 1, arr)
                    elif arr.size == 3:
                        GL.glUniform3fv(loc, 1, arr)
                    elif arr.size == 2:
                        GL.glUniform2fv(loc, 1, arr)
                elif isinstance(u_val, (tuple, list)):
                    if len(u_val) == 2:
                        GL.glUniform2f(loc, float(u_val[0]), float(u_val[1]))
                    elif len(u_val) == 3:
                        GL.glUniform3f(
                            loc, float(u_val[0]), float(u_val[1]), float(u_val[2])
                        )
                    elif len(u_val) == 4:
                        GL.glUniform4f(
                            loc,
                            float(u_val[0]),
                            float(u_val[1]),
                            float(u_val[2]),
                            float(u_val[3]),
                        )
                elif isinstance(u_val, int):
                    GL.glUniform1i(loc, u_val)
                elif isinstance(u_val, float):
                    GL.glUniform1f(loc, u_val)

        vertex_count = vbo_arr.size // total_floats
        if indices is not None:
            if instance_count > 1:
                GL.glDrawElementsInstanced(
                    GL.GL_TRIANGLES,
                    num_indices,
                    GL.GL_UNSIGNED_INT,
                    ctypes.c_void_p(0),
                    instance_count,
                )
            else:
                GL.glDrawElements(
                    GL.GL_TRIANGLES,
                    num_indices,
                    GL.GL_UNSIGNED_INT,
                    ctypes.c_void_p(0),
                )
        else:
            if instance_count > 1:
                GL.glDrawArraysInstanced(
                    GL.GL_TRIANGLES, 0, vertex_count, instance_count
                )
            else:
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, vertex_count)

        GL.glBindVertexArray(0)
        GL.glDeleteVertexArrays(1, [vao])
        GL.glDeleteBuffers(1, [vbo])
        if inst_vbo is not None:
            GL.glDeleteBuffers(1, [inst_vbo])
        if ibo is not None:
            GL.glDeleteBuffers(1, [ibo])
        if created_textures:
            GL.glDeleteTextures(len(created_textures), created_textures)

    def finish(self) -> None:
        self._gl.glFinish()

    def read_pixels(self) -> np.ndarray:
        GL = self._gl
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        raw = GL.glReadPixels(
            0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE
        )
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 4))
        return np.flipud(arr).copy()

    def release(self) -> None:
        GL = self._gl
        try:
            for prog in self._program_cache.values():
                GL.glDeleteProgram(prog)
            self._program_cache.clear()
            GL.glDeleteFramebuffers(1, [self._fbo])
            GL.glDeleteTextures(1, [self._color_tex])
            GL.glDeleteRenderbuffers(1, [self._depth_rb])
            self._ctx.destroy()
        except Exception:
            pass


def create_backend(backend: str = "auto", width: int = 512, height: int = 512) -> RenderBackend:
    """Factory function to create an off-screen OpenGL backend ('moderngl', 'pyopengl', or 'auto')."""
    choice = backend.lower().strip()
    if choice == "moderngl":
        return ModernGLBackend(width, height)
    if choice == "pyopengl":
        return PyOpenGLBackend(width, height)
    if choice == "auto":
        try:
            return ModernGLBackend(width, height)
        except Exception:
            return PyOpenGLBackend(width, height)
    raise ValueError(f"Unsupported backend '{backend}'. Choose from: auto, moderngl, pyopengl.")

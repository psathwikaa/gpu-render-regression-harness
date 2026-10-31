"""Collection of 8 deterministic off-screen OpenGL 3.3 test scenes."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backends import RenderBackend


@dataclass(frozen=True)
class SceneMetadata:
    """Descriptive metadata for an OpenGL test scene."""

    scene_id: str
    title: str
    category: str
    description: str
    features: Tuple[str, ...]
    regression_description: str


def _perspective_matrix(
    fov_deg: float, aspect: float, z_near: float, z_far: float
) -> np.ndarray:
    """Create a column-major 4x4 perspective projection matrix for OpenGL."""
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (z_far + z_near) / (z_near - z_far)
    mat[2, 3] = -1.0
    mat[3, 2] = (2.0 * z_far * z_near) / (z_near - z_far)
    return mat


def _look_at_matrix(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> np.ndarray:
    """Create a column-major 4x4 view matrix for OpenGL."""
    eye_v = np.array(eye, dtype=np.float64)
    target_v = np.array(target, dtype=np.float64)
    up_v = np.array(up, dtype=np.float64)

    fwd = target_v - eye_v
    fwd /= np.linalg.norm(fwd)
    side = np.cross(fwd, up_v)
    side /= np.linalg.norm(side)
    up_ortho = np.cross(side, fwd)

    mat = np.eye(4, dtype=np.float32)
    mat[0, 0:3] = side
    mat[1, 0:3] = up_ortho
    mat[2, 0:3] = -fwd
    # Column-major storage for GLSL uniform mat4
    col_mat = mat.T.copy()
    col_mat[3, 0] = -float(np.dot(side, eye_v))
    col_mat[3, 1] = -float(np.dot(up_ortho, eye_v))
    col_mat[3, 2] = float(np.dot(fwd, eye_v))
    return col_mat


class TestScene(ABC):
    """Base class for an off-screen OpenGL regression test scene."""

    @property
    @abstractmethod
    def metadata(self) -> SceneMetadata:
        """Return scene metadata."""

    @abstractmethod
    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        """Execute all state changes and draw calls to render the scene into backend's FBO."""


class RgbTriangleScene(TestScene):
    """Scene 1: Classic vertex-interpolated barycentric RGB triangle & frame."""

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="01_rgb_triangle",
            title="Barycentric RGB Triangle",
            category="Triangles & Vertex Interpolation",
            description=(
                "Renders a primary RGB triangle with smooth barycentric color interpolation "
                "surrounded by three corner accent triangles to verify vertex attribute "
                "interpolation and clip-space rasterization."
            ),
            features=("GL_TRIANGLES", "VBO", "VAO", "Smooth Varying Interpolation"),
            regression_description="Perturbs top vertex position and swaps green/blue vertex color channels.",
        )

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.07, 0.08, 0.12, 1.0)
        backend.set_state(depth_test=False, depth_write=False, blend=False)

        vs = """
        #version 330 core
        in vec2 in_pos;
        in vec3 in_color;
        out vec3 v_color;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_color = in_color;
        }
        """
        fs = """
        #version 330 core
        in vec3 v_color;
        out vec4 frag_color;
        void main() {
            frag_color = vec4(v_color, 1.0);
        }
        """

        top_x = 0.14 if inject_regression else 0.0
        g_col = (0.15, 0.35, 0.95) if inject_regression else (0.10, 0.92, 0.45)

        vertices = np.array(
            [
                # Outer dark frame triangle (subtle backdrop)
                [0.0, 0.86, 0.16, 0.19, 0.28],
                [-0.84, -0.74, 0.14, 0.17, 0.25],
                [0.84, -0.74, 0.18, 0.21, 0.30],
                # Primary RGB barycentric triangle
                [top_x, 0.72, 0.98, 0.22, 0.32],
                [-0.70, -0.62, g_col[0], g_col[1], g_col[2]],
                [0.70, -0.62, 0.18, 0.45, 0.98],
                # Top-left accent triangle
                [-0.78, 0.78, 0.95, 0.75, 0.20],
                [-0.88, 0.48, 0.90, 0.35, 0.15],
                [-0.52, 0.65, 0.98, 0.88, 0.35],
                # Top-right accent triangle
                [0.78, 0.78, 0.25, 0.88, 0.95],
                [0.52, 0.65, 0.45, 0.95, 0.85],
                [0.88, 0.48, 0.15, 0.55, 0.92],
            ],
            dtype=np.float32,
        )

        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_pos", 2, "2f"), ("in_color", 3, "3f")],
            vertex_data=vertices,
        )


class IndexedQuadUvScene(TestScene):
    """Scene 2: Indexed geometry (IBO/EBO) with procedural UV grid & concentric rings."""

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="02_indexed_quad_uv",
            title="Indexed Quad Mesh & Procedural UV Pattern",
            category="Indexed Geometry (EBO/IBO)",
            description=(
                "Renders a multi-quad indexed mesh using an Element Buffer Object (GL_UNSIGNED_INT) "
                "with shared vertices and a fragment shader that synthesizes a high-precision UV "
                "grid and interference rings."
            ),
            features=("glDrawElements", "Index Buffer Object", "UV Coordinates", "Uniforms"),
            regression_description="Shifts UV frequency scale uniform and inverts diagonal quad tint.",
        )

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.05, 0.06, 0.09, 1.0)
        backend.set_state(depth_test=False, depth_write=False, blend=False)

        vs = """
        #version 330 core
        in vec2 in_pos;
        in vec2 in_uv;
        in vec3 in_tint;
        out vec2 v_uv;
        out vec3 v_tint;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_uv = in_uv;
            v_tint = in_tint;
        }
        """
        fs = """
        #version 330 core
        in vec2 v_uv;
        in vec3 v_tint;
        uniform float u_freq;
        out vec4 frag_color;
        void main() {
            vec2 grid = abs(fract(v_uv * u_freq - 0.5) - 0.5) / 0.08;
            float line = clamp(min(grid.x, grid.y), 0.0, 1.0);
            float dist = length(v_uv - vec2(0.5, 0.5));
            float ring = 0.5 + 0.5 * cos(dist * 36.0);
            vec3 base = mix(vec3(0.12, 0.15, 0.22), v_tint, 0.75 * ring + 0.25);
            vec3 col = mix(vec3(0.95, 0.97, 1.0), base, line);
            frag_color = vec4(col, 1.0);
        }
        """

        # 4 quads arranged in a 2x2 layout using shared central vertices (9 vertices total, 24 indices)
        vertices = np.array(
            [
                # x, y, u, v, r, g, b
                [-0.82, 0.82, 0.0, 1.0, 0.95, 0.35, 0.45],   # 0: top-left
                [0.00, 0.82, 0.5, 1.0, 0.95, 0.75, 0.25],    # 1: top-mid
                [0.82, 0.82, 1.0, 1.0, 0.25, 0.85, 0.55],    # 2: top-right
                [-0.82, 0.00, 0.0, 0.5, 0.65, 0.35, 0.95],   # 3: mid-left
                [0.00, 0.00, 0.5, 0.5, 0.95, 0.95, 0.95],    # 4: center
                [0.82, 0.00, 1.0, 0.5, 0.20, 0.75, 0.95],    # 5: mid-right
                [-0.82, -0.82, 0.0, 0.0, 0.30, 0.50, 0.98],  # 6: bot-left
                [0.00, -0.82, 0.5, 0.0, 0.85, 0.30, 0.80],   # 7: bot-mid
                [0.82, -0.82, 1.0, 0.0, 0.98, 0.55, 0.20],   # 8: bot-right
            ],
            dtype=np.float32,
        )

        indices = np.array(
            [
                0, 3, 4, 0, 4, 1,  # Top-Left quad
                1, 4, 5, 1, 5, 2,  # Top-Right quad
                3, 6, 7, 3, 7, 4,  # Bottom-Left quad
                4, 7, 8, 4, 8, 5,  # Bottom-Right quad
            ],
            dtype=np.uint32,
        )

        freq = 4.5 if inject_regression else 8.0
        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_pos", 2, "2f"), ("in_uv", 2, "2f"), ("in_tint", 3, "3f")],
            vertex_data=vertices,
            indices=indices,
            uniforms={"u_freq": freq},
        )


class CheckerTexture2DScene(TestScene):
    """Scene 3: 2D RGBA Texture upload, UV mapping, and bilinear/nearest sampling."""

    def __init__(self) -> None:
        self._cached_tex: Optional[np.ndarray] = None

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="03_checker_texture_2d",
            title="2D Procedural Texture & UV Mapping",
            category="Textures & Filtering",
            description=(
                "Uploads a procedurally generated 128x128 RGBA texture (multi-color checkerboard "
                "with concentric calibration rings) to GL_TEXTURE_2D and samples it across "
                "primary and inset textured quads."
            ),
            features=("GL_TEXTURE_2D", "Sampler2D", "UV Tiling", "Texture Filtering"),
            regression_description="Applies a UV phase offset and corrupts texture channel swizzle.",
        )

    @staticmethod
    def _build_calibration_texture(size: int = 128) -> np.ndarray:
        """Generate a deterministic 128x128 RGBA8 calibration texture."""
        y, x = np.mgrid[0:size, 0:size].astype(np.float32)
        u = x / float(size)
        v = y / float(size)

        # 8x8 checkerboard
        cx = (np.floor(u * 8.0).astype(np.int32)) & 1
        cy = (np.floor(v * 8.0).astype(np.int32)) & 1
        checker = (cx ^ cy).astype(np.float32)

        # Radial rings around center
        dx = u - 0.5
        dy = v - 0.5
        r = np.sqrt(dx * dx + dy * dy)
        ring_mask = ((r > 0.18) & (r < 0.24)) | ((r > 0.34) & (r < 0.39))

        tex = np.zeros((size, size, 4), dtype=np.uint8)
        # Base dark/bright checker with subtle gradient
        r_ch = np.where(checker > 0.5, 225.0 * (0.4 + 0.6 * u), 38.0)
        g_ch = np.where(checker > 0.5, 215.0 * (0.4 + 0.6 * v), 46.0)
        b_ch = np.where(checker > 0.5, 240.0 * (0.8 - 0.4 * u), 68.0)

        # Highlight calibration rings in vibrant amber/cyan
        r_ch = np.where(ring_mask, 250.0, r_ch)
        g_ch = np.where(ring_mask, 165.0, g_ch)
        b_ch = np.where(ring_mask, 40.0, b_ch)

        # Crisp crosshair lines at center
        cross = (np.abs(x - size * 0.5) < 1.5) | (np.abs(y - size * 0.5) < 1.5)
        r_ch = np.where(cross, 50.0, r_ch)
        g_ch = np.where(cross, 245.0, g_ch)
        b_ch = np.where(cross, 195.0, b_ch)

        tex[..., 0] = np.clip(r_ch, 0, 255).astype(np.uint8)
        tex[..., 1] = np.clip(g_ch, 0, 255).astype(np.uint8)
        tex[..., 2] = np.clip(b_ch, 0, 255).astype(np.uint8)
        tex[..., 3] = 255
        return tex

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.08, 0.09, 0.13, 1.0)
        backend.set_state(depth_test=False, depth_write=False, blend=False)

        vs = """
        #version 330 core
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_uv = in_uv;
        }
        """
        fs = """
        #version 330 core
        in vec2 v_uv;
        uniform sampler2D u_tex;
        uniform vec2 u_uv_offset;
        uniform int u_swizzle_bug;
        out vec4 frag_color;
        void main() {
            vec4 sample_col = texture(u_tex, v_uv + u_uv_offset);
            if (u_swizzle_bug == 1) {
                sample_col.rgb = sample_col.bgr;
            }
            frag_color = sample_col;
        }
        """

        # Main diamond-tilted quad + two corner preview quads (with 2x UV tiling)
        vertices = np.array(
            [
                # Main center quad (slightly rotated diamond/square)
                [-0.68, 0.58, 0.0, 1.0],
                [-0.58, -0.68, 0.0, 0.0],
                [0.68, -0.58, 1.0, 0.0],
                [-0.68, 0.58, 0.0, 1.0],
                [0.68, -0.58, 1.0, 0.0],
                [0.58, 0.68, 1.0, 1.0],
                # Top-right tiled mini-quad (2x2 repeat)
                [0.45, 0.88, 0.0, 2.0],
                [0.45, 0.45, 0.0, 0.0],
                [0.88, 0.45, 2.0, 0.0],
                [0.45, 0.88, 0.0, 2.0],
                [0.88, 0.45, 2.0, 0.0],
                [0.88, 0.88, 2.0, 2.0],
            ],
            dtype=np.float32,
        )

        if self._cached_tex is None:
            self._cached_tex = self._build_calibration_texture(128)
        tex_arr = self._cached_tex
        uv_offset = (0.12, -0.08) if inject_regression else (0.0, 0.0)
        swizzle_bug = 1 if inject_regression else 0

        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_pos", 2, "2f"), ("in_uv", 2, "2f")],
            vertex_data=vertices,
            textures={"u_tex": (tex_arr, "linear")},
            uniforms={"u_uv_offset": uv_offset, "u_swizzle_bug": swizzle_bug},
        )


class AlphaBlendingGlassScene(TestScene):
    """Scene 4: Multi-layer translucent alpha blending (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)."""

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="04_alpha_blending_glass",
            title="Multi-Layer Translucent Alpha Blending",
            category="Blending & Compositing",
            description=(
                "Renders a high-contrast background stripe pattern followed by 5 overlapping "
                "translucent glass panes using standard source-over alpha blending "
                "(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) to verify blend equation precision."
            ),
            features=("GL_BLEND", "GL_SRC_ALPHA", "GL_ONE_MINUS_SRC_ALPHA", "Order-Dependent Alpha"),
            regression_description="Disables hardware alpha blending, causing translucent layers to overwrite opaques.",
        )

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.06, 0.07, 0.10, 1.0)
        # If inject_regression is True, intentionally disable blending to simulate a pipeline state bug
        backend.set_state(depth_test=False, depth_write=False, blend=not inject_regression)

        vs = """
        #version 330 core
        in vec2 in_pos;
        in vec4 in_color;
        out vec4 v_color;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_color = in_color;
        }
        """
        fs = """
        #version 330 core
        in vec4 v_color;
        out vec4 frag_color;
        void main() {
            frag_color = v_color;
        }
        """

        quads: List[List[float]] = []

        def add_quad(
            x0: float,
            y0: float,
            x1: float,
            y1: float,
            r: float,
            g: float,
            b: float,
            a: float,
        ) -> None:
            quads.extend(
                [
                    [x0, y1, r, g, b, a],
                    [x0, y0, r, g, b, a],
                    [x1, y0, r, g, b, a],
                    [x0, y1, r, g, b, a],
                    [x1, y0, r, g, b, a],
                    [x1, y1, r, g, b, a],
                ]
            )

        # Opaque background reference bars
        for i in range(6):
            y_bot = -0.80 + i * 0.28
            y_top = y_bot + 0.14
            add_quad(-0.85, y_bot, 0.85, y_top, 0.22, 0.25, 0.34, 1.0)

        # Overlapping translucent panes (back-to-front)
        add_quad(-0.72, -0.45, 0.18, 0.65, 0.95, 0.20, 0.32, 0.58)   # Crimson glass
        add_quad(-0.28, -0.68, 0.68, 0.28, 0.15, 0.88, 0.52, 0.55)   # Emerald glass
        add_quad(-0.15, -0.15, 0.76, 0.76, 0.22, 0.52, 0.98, 0.58)   # Cobalt glass
        add_quad(-0.55, -0.72, 0.35, -0.05, 0.98, 0.78, 0.18, 0.52)  # Amber glass
        add_quad(-0.35, -0.35, 0.35, 0.35, 0.96, 0.96, 1.00, 0.35)   # Center frost pane

        vertices = np.array(quads, dtype=np.float32)
        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_pos", 2, "2f"), ("in_color", 4, "4f")],
            vertex_data=vertices,
        )


class DepthTestIntersectionScene(TestScene):
    """Scene 5: 3D Intersecting Triangles testing hardware Z-buffer occlusion."""

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="05_depth_test_intersection",
            title="3D Depth Buffer Intersecting Planes",
            category="Depth Testing (Z-Buffer)",
            description=(
                "Renders three inclined 3D triangles that pierce directly through one another "
                "across the Z=0 plane with GL_DEPTH_TEST (GL_LESS) enabled, verifying per-pixel "
                "24-bit hardware depth buffer occlusion along diagonal intersection seams."
            ),
            features=("GL_DEPTH_TEST", "GL_LESS", "24-bit Depth Buffer", "3D Clip-Space Z"),
            regression_description="Disables GL_DEPTH_TEST so draw-order painter's algorithm replaces Z-occlusion.",
        )

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.05, 0.06, 0.09, 1.0, depth=1.0)
        # When inject_regression is True, disable depth testing so occlusion fails
        backend.set_state(depth_test=not inject_regression, depth_write=True, blend=False)

        vs = """
        #version 330 core
        in vec3 in_pos;
        in vec3 in_color;
        out vec3 v_color;
        out float v_depth;
        void main() {
            gl_Position = vec4(in_pos, 1.0);
            v_color = in_color;
            v_depth = in_pos.z;
        }
        """
        fs = """
        #version 330 core
        in vec3 v_color;
        in float v_depth;
        out vec4 frag_color;
        void main() {
            // Subtle depth cue shading based on clip-space Z (-1 near, +1 far)
            float shade = 1.0 - 0.25 * (v_depth + 0.6);
            frag_color = vec4(v_color * shade, 1.0);
        }
        """

        vertices = np.array(
            [
                # Plane A (Coral-Red): tilted left-near (z=-0.55) to right-far (z=+0.55)
                [-0.82, 0.72, -0.55, 0.96, 0.26, 0.34],
                [-0.82, -0.72, -0.55, 0.92, 0.18, 0.28],
                [0.82, 0.00, 0.55, 0.98, 0.55, 0.38],
                # Plane B (Cyan-Blue): tilted right-near (z=-0.55) to left-far (z=+0.55)
                [0.82, 0.72, -0.55, 0.18, 0.68, 0.98],
                [-0.82, 0.00, 0.55, 0.12, 0.42, 0.90],
                [0.82, -0.72, -0.55, 0.28, 0.88, 0.95],
                # Plane C (Gold-Green): tilted bottom-near (z=-0.60) to top-far (z=+0.60)
                [0.00, 0.85, 0.60, 0.25, 0.85, 0.45],
                [-0.68, -0.78, -0.60, 0.88, 0.92, 0.25],
                [0.68, -0.78, -0.60, 0.45, 0.95, 0.55],
            ],
            dtype=np.float32,
        )

        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_pos", 3, "3f"), ("in_color", 3, "3f")],
            vertex_data=vertices,
        )


class DepthAndBlendingCompositeScene(TestScene):
    """Scene 6: Combined opaque 3D depth-buffered geometry + translucent depth-tested overlays."""

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="06_depth_and_blending_composite",
            title="Multi-Pass Depth & Translucent Composite",
            category="Depth + Blending Pipeline State",
            description=(
                "Executes a two-pass pipeline: Pass 1 renders opaque 3D pillars with depth "
                "testing and depth writes enabled; Pass 2 renders semi-transparent glass bands "
                "at mid-depth with depth testing enabled and depth writes disabled."
            ),
            features=("Multi-Pass Rendering", "glDepthMask", "GL_DEPTH_TEST + GL_BLEND"),
            regression_description="Shifts mid-depth translucent band Z behind the back pillar.",
        )

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.06, 0.08, 0.12, 1.0, depth=1.0)

        vs = """
        #version 330 core
        in vec3 in_pos;
        in vec4 in_color;
        out vec4 v_color;
        void main() {
            gl_Position = vec4(in_pos, 1.0);
            v_color = in_color;
        }
        """
        fs = """
        #version 330 core
        in vec4 v_color;
        out vec4 frag_color;
        void main() {
            frag_color = v_color;
        }
        """

        def make_box_3d(
            x0: float,
            y0: float,
            x1: float,
            y1: float,
            z: float,
            c_top: Tuple[float, float, float, float],
            c_bot: Tuple[float, float, float, float],
        ) -> List[List[float]]:
            return [
                [x0, y1, z, *c_top],
                [x0, y0, z, *c_bot],
                [x1, y0, z, *c_bot],
                [x0, y1, z, *c_top],
                [x1, y0, z, *c_bot],
                [x1, y1, z, *c_top],
            ]

        # Pass 1: Opaque pillars submitted out-of-order in Z to test hardware depth buffer
        opaque_verts: List[List[float]] = []
        # Near-left pillar (z = -0.45, closest to camera)
        opaque_verts.extend(
            make_box_3d(
                -0.65,
                -0.75,
                -0.25,
                0.75,
                -0.45,
                (0.95, 0.42, 0.25, 1.0),
                (0.75, 0.22, 0.15, 1.0),
            )
        )
        # Far-center diagonal pillar (z = +0.40, farthest from camera)
        opaque_verts.extend(
            make_box_3d(
                -0.45,
                -0.60,
                0.45,
                0.60,
                0.40,
                (0.25, 0.55, 0.92, 1.0),
                (0.15, 0.30, 0.68, 1.0),
            )
        )
        # Mid-right pillar (z = +0.05)
        opaque_verts.extend(
            make_box_3d(
                0.20,
                -0.75,
                0.65,
                0.75,
                0.05,
                (0.22, 0.88, 0.58, 1.0),
                (0.12, 0.58, 0.36, 1.0),
            )
        )

        backend.set_state(depth_test=True, depth_write=True, blend=False)
        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_pos", 3, "3f"), ("in_color", 4, "4f")],
            vertex_data=np.array(opaque_verts, dtype=np.float32),
        )

        # Pass 2: Translucent glass ribbons at intermediate depths
        # Ribbon 1 at z = -0.15 (behind near-left pillar at -0.45, but in front of mid-right +0.05 and far +0.40)
        z_ribbon1 = 0.65 if inject_regression else -0.15
        # Ribbon 2 at z = +0.20 (behind near-left and mid-right, in front of far-center +0.40)
        z_ribbon2 = 0.20

        glass_verts: List[List[float]] = []
        glass_verts.extend(
            make_box_3d(
                -0.82,
                -0.48,
                0.82,
                0.08,
                z_ribbon2,
                (0.95, 0.25, 0.85, 0.55),
                (0.75, 0.15, 0.65, 0.55),
            )
        )
        glass_verts.extend(
            make_box_3d(
                -0.82,
                -0.08,
                0.82,
                0.48,
                z_ribbon1,
                (0.98, 0.88, 0.25, 0.55),
                (0.95, 0.65, 0.15, 0.55),
            )
        )

        backend.set_state(depth_test=True, depth_write=False, blend=True)
        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_pos", 3, "3f"), ("in_color", 4, "4f")],
            vertex_data=np.array(glass_verts, dtype=np.float32),
        )


class InstancedPrimitivesGridScene(TestScene):
    """Scene 7: Hardware Instanced Rendering (36 geometric rosettes in 1 draw call)."""

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="07_instanced_primitives_grid",
            title="Hardware Instanced 6x6 Geometric Grid",
            category="Instanced Rendering",
            description=(
                "Renders 36 geometric 6-pointed star rosettes in a single hardware-instanced "
                "draw call (glDrawArraysInstanced) using per-instance translation, scale, "
                "rotation angle, and color attributes."
            ),
            features=("glDrawArraysInstanced", "VertexAttribDivisor", "Per-Instance Transforms"),
            regression_description="Corrupts per-instance rotation angle and scale progression.",
        )

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.05, 0.06, 0.09, 1.0)
        backend.set_state(depth_test=False, depth_write=False, blend=False)

        vs = """
        #version 330 core
        in vec2 in_local_pos;
        in float in_hub_weight;
        in vec2 in_offset;
        in vec2 in_scale_rot;
        in vec3 in_inst_color;
        out vec3 v_color;
        void main() {
            float scale = in_scale_rot.x;
            float angle = in_scale_rot.y;
            float c = cos(angle);
            float s = sin(angle);
            vec2 rotated = vec2(
                in_local_pos.x * c - in_local_pos.y * s,
                in_local_pos.x * s + in_local_pos.y * c
            );
            vec2 world = in_offset + rotated * scale;
            gl_Position = vec4(world, 0.0, 1.0);
            v_color = mix(in_inst_color, vec3(0.98, 0.98, 1.0), in_hub_weight * 0.55);
        }
        """
        fs = """
        #version 330 core
        in vec3 v_color;
        out vec4 frag_color;
        void main() {
            frag_color = vec4(v_color, 1.0);
        }
        """

        # Base geometry: 6-pointed star composed of 12 triangles radiating from center
        num_spikes = 12
        base_verts: List[List[float]] = []
        for i in range(num_spikes):
            a0 = (2.0 * math.pi * i) / num_spikes
            a1 = (2.0 * math.pi * (i + 1)) / num_spikes
            r0 = 1.0 if (i % 2 == 0) else 0.48
            r1 = 1.0 if ((i + 1) % 2 == 0) else 0.48
            base_verts.append([0.0, 0.0, 1.0])
            base_verts.append([r0 * math.cos(a0), r0 * math.sin(a0), 0.0])
            base_verts.append([r1 * math.cos(a1), r1 * math.sin(a1), 0.0])

        vertex_data = np.array(base_verts, dtype=np.float32)

        # 6x6 grid of instances (36 total)
        grid_n = 6
        instances: List[List[float]] = []
        for row in range(grid_n):
            for col in range(grid_n):
                u = (col + 0.5) / grid_n
                v = (row + 0.5) / grid_n
                ox = -0.76 + (col / (grid_n - 1)) * 1.52
                oy = -0.76 + (row / (grid_n - 1)) * 1.52
                idx = row * grid_n + col
                scale = 0.115 - 0.02 * math.sin(idx * 0.4)
                rot = idx * (0.85 if inject_regression else 0.22)
                if inject_regression and (idx % 3 == 0):
                    scale *= 0.45

                r_col = 0.25 + 0.70 * u
                g_col = 0.30 + 0.65 * v
                b_col = 0.92 - 0.55 * u * v
                instances.append([ox, oy, scale, rot, r_col, g_col, b_col])

        instance_data = np.array(instances, dtype=np.float32)

        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[("in_local_pos", 2, "2f"), ("in_hub_weight", 1, "1f")],
            vertex_data=vertex_data,
            instance_attributes=[
                ("in_offset", 2, "2f"),
                ("in_scale_rot", 2, "2f"),
                ("in_inst_color", 3, "3f"),
            ],
            instance_data=instance_data,
            instance_count=len(instances),
        )


class PhongShadedSphereScene(TestScene):
    """Scene 8: 3D Tessellated UV Sphere with Blinn-Phong + Rim Lighting and Depth Buffer."""

    def __init__(self) -> None:
        self._cached_mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def metadata(self) -> SceneMetadata:
        return SceneMetadata(
            scene_id="08_phong_shaded_sphere",
            title="3D Blinn-Phong Lit Sphere & Orbiting Ring",
            category="3D Shading & Lighting",
            description=(
                "Renders a tessellated 3D sphere and tilted orbital ring with perspective "
                "projection, surface normals, two directional lights (warm key + cool fill), "
                "Blinn-Phong specular highlights, and Fresnel rim lighting."
            ),
            features=("3D Perspective MVP", "Vertex Normals", "Blinn-Phong Specular", "GL_DEPTH_TEST"),
            regression_description="Inverts light direction vector and zeroes out specular highlights.",
        )

    @staticmethod
    def _build_sphere_and_ring_mesh(
        lat_steps: int = 28, lon_steps: int = 36
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate indexed vertex buffer (pos3, norm3, col3) and uint32 index buffer."""
        verts: List[List[float]] = []
        indices: List[int] = []

        # 1. UV Sphere (radius = 0.85)
        radius = 0.85
        for i in range(lat_steps + 1):
            theta = math.pi * i / lat_steps
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            v_coord = i / float(lat_steps)

            for j in range(lon_steps + 1):
                phi = 2.0 * math.pi * j / lon_steps
                nx = sin_t * math.cos(phi)
                ny = cos_t
                nz = sin_t * math.sin(phi)
                px = radius * nx
                py = radius * ny
                pz = radius * nz

                # Subtle latitude band color modulation
                band = 0.5 + 0.5 * math.cos(v_coord * math.pi * 6.0)
                cr = 0.22 + 0.20 * band
                cg = 0.52 + 0.25 * (1.0 - v_coord)
                cb = 0.88 - 0.15 * band
                verts.append([px, py, pz, nx, ny, nz, cr, cg, cb])

        stride = lon_steps + 1
        for i in range(lat_steps):
            for j in range(lon_steps):
                v0 = i * stride + j
                v1 = v0 + stride
                v2 = v0 + 1
                v3 = v1 + 1
                indices.extend([v0, v1, v2, v2, v1, v3])

        # 2. Tilted orbital ring around the sphere to test 3D depth occlusion with the sphere
        base_idx = len(verts)
        ring_segments = 48
        r_inner = 1.08
        r_outer = 1.38
        tilt = math.radians(26.0)
        cos_tilt = math.cos(tilt)
        sin_tilt = math.sin(tilt)

        for k in range(ring_segments + 1):
            ang = 2.0 * math.pi * k / ring_segments
            ca = math.cos(ang)
            sa = math.sin(ang)
            for r_val, c_tint in (
                (r_inner, (0.95, 0.65, 0.22)),
                (r_outer, (0.92, 0.35, 0.28)),
            ):
                x0 = r_val * ca
                y0 = 0.0
                z0 = r_val * sa
                # Rotate around X axis by tilt
                rx = x0
                ry = y0 * cos_tilt - z0 * sin_tilt
                rz = y0 * sin_tilt + z0 * cos_tilt
                # Normal points along tilted Y axis + slight radial component
                nx = 0.25 * ca
                ny = cos_tilt
                nz = sin_tilt
                n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
                verts.append(
                    [rx, ry, rz, nx / n_len, ny / n_len, nz / n_len, *c_tint]
                )

        for k in range(ring_segments):
            i0 = base_idx + k * 2
            i1 = i0 + 1
            i2 = i0 + 2
            i3 = i0 + 3
            indices.extend([i0, i1, i2, i2, i1, i3])

        return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def render(self, backend: RenderBackend, inject_regression: bool = False) -> None:
        backend.clear(0.05, 0.06, 0.10, 1.0, depth=1.0)
        backend.set_state(depth_test=True, depth_write=True, blend=False)

        vs = """
        #version 330 core
        in vec3 in_pos;
        in vec3 in_normal;
        in vec3 in_color;
        uniform mat4 u_mvp;
        out vec3 v_world_pos;
        out vec3 v_normal;
        out vec3 v_color;
        void main() {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            v_world_pos = in_pos;
            v_normal = normalize(in_normal);
            v_color = in_color;
        }
        """
        fs = """
        #version 330 core
        in vec3 v_world_pos;
        in vec3 v_normal;
        in vec3 v_color;
        uniform vec3 u_light_dir;
        uniform vec3 u_view_pos;
        uniform float u_spec_strength;
        out vec4 frag_color;
        void main() {
            vec3 N = normalize(v_normal);
            vec3 L = normalize(u_light_dir);
            vec3 V = normalize(u_view_pos - v_world_pos);
            vec3 H = normalize(L + V);

            // Key directional light (warm)
            float diff_key = max(dot(N, L), 0.0);
            float spec = pow(max(dot(N, H), 0.0), 40.0) * u_spec_strength;

            // Fill light from opposite lower-left (cool)
            vec3 L_fill = normalize(vec3(-0.6, -0.4, 0.5));
            float diff_fill = max(dot(N, L_fill), 0.0) * 0.35;

            // Fresnel rim lighting
            float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.38;

            vec3 ambient = 0.16 * v_color;
            vec3 diffuse = (diff_key * vec3(1.0, 0.96, 0.90) + diff_fill * vec3(0.45, 0.70, 1.0)) * v_color;
            vec3 specular = spec * vec3(1.0, 0.98, 0.92);
            vec3 rim_col = rim * vec3(0.35, 0.75, 1.0);

            vec3 final_rgb = clamp(ambient + diffuse + specular + rim_col, 0.0, 1.0);
            frag_color = vec4(final_rgb, 1.0);
        }
        """

        if self._cached_mesh is None:
            self._cached_mesh = self._build_sphere_and_ring_mesh()
        verts, indices = self._cached_mesh
        eye = (0.0, 0.55, 2.85)
        proj = _perspective_matrix(42.0, 1.0, 0.1, 15.0)
        view = _look_at_matrix(eye, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        # Column-major matrix multiplication: MVP = P * V => in numpy row-view of column-major: V @ P
        mvp = np.matmul(view, proj).astype(np.float32)

        light_dir = (-0.65, -0.75, -0.50) if inject_regression else (0.65, 0.75, 0.60)
        spec_strength = 0.0 if inject_regression else 0.75

        backend.draw_arrays(
            vertex_shader=vs,
            fragment_shader=fs,
            attributes=[
                ("in_pos", 3, "3f"),
                ("in_normal", 3, "3f"),
                ("in_color", 3, "3f"),
            ],
            vertex_data=verts,
            indices=indices,
            uniforms={
                "u_mvp": mvp,
                "u_light_dir": light_dir,
                "u_view_pos": eye,
                "u_spec_strength": spec_strength,
            },
        )


ALL_SCENES: List[TestScene] = [
    RgbTriangleScene(),
    IndexedQuadUvScene(),
    CheckerTexture2DScene(),
    AlphaBlendingGlassScene(),
    DepthTestIntersectionScene(),
    DepthAndBlendingCompositeScene(),
    InstancedPrimitivesGridScene(),
    PhongShadedSphereScene(),
]


def get_scenes(scene_filter: Optional[List[str]] = None) -> List[TestScene]:
    """Return all registered OpenGL test scenes or filter by scene_id substring/exact match."""
    if not scene_filter:
        return list(ALL_SCENES)
    normalized = [s.strip().lower() for s in scene_filter if s.strip()]
    if not normalized or "all" in normalized:
        return list(ALL_SCENES)

    selected: List[TestScene] = []
    for scene in ALL_SCENES:
        sid = scene.metadata.scene_id.lower()
        if any(f == sid or f in sid for f in normalized):
            selected.append(scene)
    return selected

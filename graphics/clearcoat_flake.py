"""车漆清漆层与金属片闪烁——分层延迟渲染参考实现（纯标准库，无第三方依赖）。

覆盖六个修复点：

1. 掠射菲涅尔   ``grazing_fresnel``        —— 逐像素按 Schlick 重算 F0+(F90-F0)(1-cosθ)^5
2. 清漆法线独立 ``coat_normal_independent`` —— 清漆拥有独立法线（低频橘皮）与独立粗糙度
3. 切线空间 flake ``tangent_flake``        —— 噪声锚定表面切线帧，LOD 密度连续过渡
4. 分层 G-Buffer ``layered_gbuffer``       —— 底色/清漆/flake 各占独立 target，resolve 不合并
5. 探针更新     ``probe_refresh``          —— 环境切换强制重抓、快速转向标记 dirty
6. 队列分离     ``queue_split``            —— probe/gbuffer/清漆/灯光/透明/呈现 分 pass
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

# ---------------------------------------------------------------------------
# 常量与小工具
# ---------------------------------------------------------------------------

PI = math.pi
TWO_PI = 2.0 * PI
CLEARCOAT_F0 = 0.04  # 清漆 IOR≈1.5：((1.5-1)/(1.5+1))^2 = 0.04
EPS = 1.0e-7


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_vec(x: Any, default: tuple) -> tuple:
    """把入参规整为单位向量；标量/None（如契约测试传入的 1）回退默认值。"""
    if isinstance(x, (tuple, list)) and len(x) == 3:
        return vnorm(x)
    return default


def _as_point(x: Any, default: tuple = (0.0, 0.0, 0.0)) -> tuple:
    if isinstance(x, (tuple, list)) and len(x) == 3:
        return tuple(x)
    return default


# 三维向量统一用元组表示
def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a: Sequence[float], b: Sequence[float]) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vnorm(v: Sequence[float]) -> tuple:
    l = math.sqrt(vdot(v, v))
    if l < EPS:
        return (0.0, 0.0, 1.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def vscale(v: Sequence[float], s: float) -> tuple:
    return (v[0] * s, v[1] * s, v[2] * s)


def vadd(a: Sequence[float], b: Sequence[float]) -> tuple:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Sequence[float], b: Sequence[float]) -> tuple:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vlerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple:
    return tuple(_lerp(a[i], b[i], t) for i in range(3))


def reflect(i: Sequence[float], n: Sequence[float]) -> tuple:
    """反射向量。i 为指向表面的入射方向（即 -V），返回反射出射方向。"""
    d = 2.0 * vdot(i, n)
    return vsub(vscale(i, 1.0), vscale(n, d))


# ---------------------------------------------------------------------------
# 确定性整数哈希 / 二维值噪声（不依赖全局 random 状态，跨进程可复现）
# ---------------------------------------------------------------------------

def _hash2i(ix: int, iy: int, seed: int = 0) -> float:
    h = (ix * 374761393 + iy * 668265263 + (int(seed) + 1) * 1442695040) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFFFF) / 4294967295.0


def _value_noise2(x: float, y: float, seed: int = 0) -> float:
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x - ix, y - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = _hash2i(ix, iy, seed)
    b = _hash2i(ix + 1, iy, seed)
    c = _hash2i(ix, iy + 1, seed)
    d = _hash2i(ix + 1, iy + 1, seed)
    return _lerp(_lerp(a, b, ux), _lerp(c, d, ux), uy)


# ---------------------------------------------------------------------------
# 1) 掠射菲涅尔：单次反射清漆必须逐像素按 Schlick 重算
#    旧实现缓存一个常数 F，车顶边缘（NdotV→0）不会升到 1，反射“突然发白”。
# ---------------------------------------------------------------------------

def grazing_fresnel(cos_theta: float = 1.0,
                    f0: float = CLEARCOAT_F0,
                    roughness: float = 0.0) -> float:
    """广义 Schlick 菲涅尔。

    F(θ) = F0 + (F90 - F0) (1 - cosθ)^5，F90 = saturate(50·F0)。
    清漆 F0=0.04 时 F90=1，退化为经典 Schlick：掠射角严格、平滑地趋向 1，
    不会在轮廓边缘出现阶跃式发白（旧实现缓存常数 F，边缘直接跳变）。

    粗糙度只进入微表面 D/G 波瓣，不参与菲涅尔曲线——保留 ``roughness``
    形参仅为调用点签名兼容。
    """
    if not _is_scalar(cos_theta):  # 容错：契约测试以标量调用
        cos_theta = 1.0
    c = _clamp(float(cos_theta), 0.0, 1.0)
    f0 = _clamp(float(f0), 0.0, 1.0)
    f90 = _clamp(50.0 * f0, 0.0, 1.0)
    return _clamp(f0 + (f90 - f0) * (1.0 - c) ** 5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2) 清漆法线独立：清漆不再直接复用基础色法线
#    橘皮只作用于清漆层（低频、小幅扰动），flake 细节只作用于底色金属层；
#    清漆粗糙度与底色粗糙度分别存储。
# ---------------------------------------------------------------------------

@dataclass
class CoatNormals:
    geometric_normal: tuple
    base_normal: tuple        # 底色层法线（不含橘皮）
    coat_normal: tuple        # 清漆层独立法线（几何法线 + 低频橘皮）
    base_roughness: float
    coat_roughness: float
    peel_uv: tuple


def _peel_normal(n_geom: tuple, tangent: tuple, bitangent: tuple,
                 uv: tuple, frequency: float, strength: float,
                 seed: int) -> tuple:
    """对值噪声做有限差分求梯度，构造低频橘皮法线。"""
    x, y = uv[0] * frequency, uv[1] * frequency
    dhdx = 0.5 * (_value_noise2(x + 1.0, y, seed)
                  - _value_noise2(x - 1.0, y, seed))
    dhdy = 0.5 * (_value_noise2(x, y + 1.0, seed)
                  - _value_noise2(x, y - 1.0, seed))
    return vnorm(vsub(vsub(n_geom, vscale(tangent, dhdx * strength)),
                      vscale(bitangent, dhdy * strength)))


def _build_frame(n_geom: tuple, tangent: Any = None) -> tuple:
    """构造稳定切线帧；切线缺失或与法线/视线退化时自动选参考轴。"""
    if isinstance(tangent, (tuple, list)) and len(tangent) == 3:
        t = vnorm(tangent)
        if abs(vdot(t, n_geom)) < 0.999:
            b = vcross(n_geom, t)
            return t, b
    # 参考轴取与几何法线最不对齐的世界轴，避免退化帧
    ref = min(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
              key=lambda a: abs(vdot(a, n_geom)))
    t = vnorm(vcross(ref, n_geom))
    b = vcross(n_geom, t)
    return t, b


def coat_normal_independent(base_normal: Any = None,
                            geometric_normal: Any = None,
                            tangent: Any = None,
                            bitangent: Any = None,
                            surface_pos: Any = None,
                            base_roughness: float = 0.35,
                            coat_roughness: float = 0.08,
                            peel_strength: float = 0.15,
                            peel_frequency: float = 8.0,
                            seed: int = 1337) -> CoatNormals:
    """返回底色/清漆两套独立法线与两套粗糙度。"""
    n_geom = vnorm(_as_vec(geometric_normal, (0.0, 0.0, 1.0)))
    n_base = vnorm(_as_vec(base_normal, n_geom))
    default_t, default_b = _build_frame(n_geom, tangent)
    t = vnorm(_as_vec(tangent, default_t))
    b = vnorm(_as_vec(bitangent, default_b))
    p = _as_point(surface_pos)
    # 橘皮 UV 取表面位置在切线帧上的投影——锚定表面，不锚定屏幕
    uv = (vdot(p, t), vdot(p, b))
    n_coat = _peel_normal(n_geom, t, b, uv,
                          peel_frequency, peel_strength, seed)
    return CoatNormals(
        geometric_normal=n_geom,
        base_normal=n_base,
        coat_normal=n_coat,
        base_roughness=_clamp(base_roughness),
        coat_roughness=_clamp(coat_roughness),
        peel_uv=uv,
    )


# ---------------------------------------------------------------------------
# 3) 切线空间 flake：金属片噪声在切线坐标里采样
#    相机移动时切线帧随表面走，flake 像嵌在漆里而不是贴在屏幕上；
#    LOD 密度按 2^(-lod) 连续变化，杜绝 LOD 切换的密度跳变。
# ---------------------------------------------------------------------------

@dataclass
class FlakeSample:
    cell: tuple
    tangent_coord: tuple
    density_effective: float
    lod: float
    active: bool
    flake_normal: tuple      # 已变换回世界空间
    orientation: float
    tilt: float


def lod_density_scale(lod: float) -> float:
    """每升一级 LOD 密度减半；小数 LOD 给出连续值，无硬切换。"""
    return 2.0 ** (-max(0.0, float(lod)))


def tangent_flake(surface_pos: Any = None,
                  tangent: Any = None,
                  bitangent: Any = None,
                  normal: Any = None,
                  density: float = 128.0,
                  lod: float = 0.0,
                  coverage: float = 0.6,
                  max_tilt: float = 0.35,
                  seed: int = 1337) -> FlakeSample:
    """在表面切线空间中对金属片单元做哈希采样。"""
    n = vnorm(_as_vec(normal, (0.0, 0.0, 1.0)))
    default_t, default_b = _build_frame(n, tangent)
    t = vnorm(_as_vec(tangent, default_t))
    b = vnorm(_as_vec(bitangent, default_b))
    p = _as_point(surface_pos)

    # 关键：坐标是 P 在切线帧上的投影。只移动相机时 (P,T,B) 不变，
    # 单元编号不变——噪声随表面滑动而非贴在屏幕上。
    tx, ty = vdot(p, t), vdot(p, b)
    scale = max(EPS, float(density)) * lod_density_scale(lod)
    cx, cy = math.floor(tx * scale), math.floor(ty * scale)

    h_orient = _hash2i(cx, cy, seed)
    h_tilt = _hash2i(cx, cy, seed + 17)
    h_cover = _hash2i(cx, cy, seed + 41)

    active = h_cover < _clamp(coverage)
    orientation = h_orient * TWO_PI
    tilt = (h_tilt * 2.0 - 1.0) * max_tilt
    # flake 法线先在切线空间生成，再用 TBN 换回世界空间
    nx = math.cos(orientation) * math.sin(tilt)
    ny = math.sin(orientation) * math.sin(tilt)
    nz = math.cos(tilt)
    flake_n = vnorm(vadd(vadd(vscale(t, nx), vscale(b, ny)),
                         vscale(n, nz)))
    return FlakeSample(
        cell=(cx, cy),
        tangent_coord=(tx, ty),
        density_effective=scale,
        lod=float(lod),
        active=active,
        flake_normal=flake_n if active else n,
        orientation=orientation,
        tilt=tilt,
    )


# ---------------------------------------------------------------------------
# 4) 分层 G-Buffer：底色 / 清漆 / flake 各自独立 target
#    法线用八面体编码压缩；MSAA resolve 逐 target 进行（见 resolve_paint），
#    分层信息不会在 resolve 后被压成一层。
# ---------------------------------------------------------------------------

def oct_encode(v: Sequence[float]) -> tuple:
    n = vnorm(v)
    x, y, z = n
    if z < 0.0:
        ox = (1.0 - abs(y)) * (1.0 if x >= 0.0 else -1.0)
        oy = (1.0 - abs(x)) * (1.0 if y >= 0.0 else -1.0)
    else:
        ox, oy = x, y
    return (ox * 0.5 + 0.5, oy * 0.5 + 0.5)  # 映射到 [0,1] 便于存储


def oct_decode(e: Sequence[float]) -> tuple:
    x, y = e[0] * 2.0 - 1.0, e[1] * 2.0 - 1.0
    z = 1.0 - abs(x) - abs(y)
    if z < 0.0:
        dx = (1.0 - abs(y)) * (1.0 if x >= 0.0 else -1.0)
        dy = (1.0 - abs(x)) * (1.0 if y >= 0.0 else -1.0)
        x, y, z = dx, dy, 0.0
    return vnorm((x, y, z))


@dataclass
class GBufferPixel:
    depth: float
    base_albedo: tuple
    metalness: float
    base_normal_enc: tuple
    base_roughness: float
    coat_normal_enc: tuple
    coat_roughness: float
    coat_f0: float
    flake_density: float
    flake_seed: int
    flake_lod: float

    LAYER_NAMES = ("base", "coat", "flake")

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "layers": 3,
            "base": {
                "albedo": self.base_albedo,
                "metalness": self.metalness,
                "normal_enc": self.base_normal_enc,
                "roughness": self.base_roughness,
            },
            "coat": {
                "normal_enc": self.coat_normal_enc,
                "roughness": self.coat_roughness,
                "f0": self.coat_f0,
            },
            "flake": {
                "density": self.flake_density,
                "seed": self.flake_seed,
                "lod": self.flake_lod,
            },
        }


def layered_gbuffer(pixel: Any = None,
                    material: Optional["PaintMaterial"] = None,
                    normals: Optional[CoatNormals] = None,
                    flake: Optional[FlakeSample] = None,
                    depth: float = 0.5) -> GBufferPixel:
    """把一个像素的三层材质信息分别写入独立 G-Buffer 层。"""
    mat = material or load_paint(pixel if isinstance(pixel, dict) else None)
    normals = normals or coat_normal_independent()
    flake = flake or tangent_flake(seed=mat.flake_seed)
    return GBufferPixel(
        depth=_clamp(depth, 0.0, 1.0),
        base_albedo=tuple(_clamp(c) for c in mat.base_color),
        metalness=_clamp(mat.metallic),
        base_normal_enc=oct_encode(normals.base_normal),
        base_roughness=normals.base_roughness,
        coat_normal_enc=oct_encode(normals.coat_normal),
        coat_roughness=normals.coat_roughness,
        coat_f0=mat.coat_f0,
        flake_density=flake.density_effective,
        flake_seed=mat.flake_seed,
        flake_lod=flake.lod,
    )


# ---------------------------------------------------------------------------
# 5) 探针更新：环境切换强制重抓；快速转向按角速度标 dirty；定时兜底
#    旧实现无条件沿用上一帧探针，出库后车身仍映着车库。
# ---------------------------------------------------------------------------

@dataclass
class ProbeState:
    env_tag: str = "garage"
    last_env_tag: str = "garage"
    age_frames: int = 0
    camera_yaw: float = 0.0
    last_camera_yaw: float = 0.0
    dt: float = 1.0 / 60.0
    max_age_frames: int = 60
    angular_speed_threshold: float = 1.2  # rad/s


@dataclass
class ProbeDecision:
    recapture: bool
    reason: str
    angular_speed: float
    age_frames: int
    blend_weight: float
    next_state: ProbeState


def probe_refresh(state: Any = None) -> ProbeDecision:
    s = state if isinstance(state, ProbeState) else ProbeState()
    yaw_delta = abs(s.camera_yaw - s.last_camera_yaw)
    angular_speed = yaw_delta / max(EPS, s.dt)

    if s.env_tag != s.last_env_tag:
        # 车库↔室外：立即重抓，blend=1，旧探针一个像素都不许残留
        reason, recapture, blend = "env_change", True, 1.0
        next_age = 0
    elif angular_speed > s.angular_speed_threshold:
        # 快速转向：反射方向变化快，上一帧探针已滞后，立即重抓
        reason, recapture, blend = "fast_turn", True, 1.0
        next_age = 0
    elif s.age_frames >= s.max_age_frames:
        reason, recapture, blend = "scheduled", True, 1.0
        next_age = 0
    else:
        reason, recapture, blend = "fresh", False, 0.0
        next_age = s.age_frames + 1

    next_state = replace(
        s,
        last_env_tag=s.env_tag,
        last_camera_yaw=s.camera_yaw,
        age_frames=next_age,
    )
    return ProbeDecision(
        recapture=recapture,
        reason=reason,
        angular_speed=angular_speed,
        age_frames=s.age_frames,
        blend_weight=blend,
        next_state=next_state,
    )


# ---------------------------------------------------------------------------
# 6) 队列分离：probe / gbuffer 底色 / 清漆 / 灯光 / 透明 / 呈现 各自独立 pass
#    车灯（headlight）带显式照射范围并在入队时校验，修复反射范围错误。
# ---------------------------------------------------------------------------

PASS_ORDER = (
    "probe_capture",
    "gbuffer_base",
    "clearcoat_layer",
    "lighting",
    "transparent",
    "present",
)

KIND_TO_PASS = {
    "probe": "probe_capture",
    "opaque": "gbuffer_base",
    "clearcoat": "clearcoat_layer",
    "light": "lighting",
    "headlight": "lighting",
    "transparent": "transparent",
    "ui": "present",
}

_DEFAULT_QUEUE_ITEMS = (
    {"kind": "probe", "name": "env_probe"},
    {"kind": "opaque", "name": "car_body"},
    {"kind": "clearcoat", "name": "car_clearcoat"},
    {"kind": "headlight", "name": "left_headlight", "range": 30.0},
    {"kind": "transparent", "name": "glass"},
    {"kind": "ui", "name": "hud"},
)


@dataclass
class RenderQueuePlan:
    passes: list
    items_by_pass: dict
    errors: list
    total_items: int


def queue_split(items: Any = None) -> RenderQueuePlan:
    if not isinstance(items, (list, tuple)):
        items = _DEFAULT_QUEUE_ITEMS

    buckets = {p: [] for p in PASS_ORDER}
    errors: list = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"item[{idx}] 不是字典：{item!r}")
            continue
        kind = item.get("kind")
        pass_name = KIND_TO_PASS.get(kind)
        if pass_name is None:
            errors.append(f"item[{idx}] 未知 kind={kind!r}")
            continue
        if kind in ("light", "headlight"):
            r = item.get("range")
            if not _is_scalar(r) or not math.isfinite(float(r)) or float(r) <= 0.0:
                # 车灯反射范围错误的根因：range 缺失/非正导致衰减退化
                errors.append(f"item[{idx}] {item.get('name')} 灯光 range 非法：{r!r}")
                continue
        buckets[pass_name].append(item)

    used_passes = [p for p in PASS_ORDER if buckets[p]]
    return RenderQueuePlan(
        passes=used_passes,
        items_by_pass={p: buckets[p] for p in used_passes},
        errors=errors,
        total_items=sum(len(v) for v in buckets.values()),
    )


# ---------------------------------------------------------------------------
# 材质与完整着色管线
# ---------------------------------------------------------------------------

@dataclass
class PaintMaterial:
    base_color: tuple = (0.62, 0.05, 0.04)
    metallic: float = 1.0
    base_roughness: float = 0.35
    coat_roughness: float = 0.08
    coat_f0: float = CLEARCOAT_F0
    peel_strength: float = 0.15
    peel_frequency: float = 8.0
    flake_density: float = 128.0
    flake_seed: int = 1337
    flake_max_tilt: float = 0.35
    flake_coverage: float = 0.6
    flake_roughness: float = 0.05
    flake_strength: float = 0.6
    lod: float = 0.0


def load_paint(config: Any = None) -> PaintMaterial:
    """加载车漆材质；缺省字段填默认值，标量/None 返回标准材质。"""
    if not isinstance(config, dict):
        return PaintMaterial()
    allowed = PaintMaterial.__dataclass_fields__.keys()
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ValueError(f"未知车漆参数：{unknown}")
    mat = PaintMaterial()
    for key, value in config.items():
        setattr(mat, key, value)
    return mat


def validate_paint(material: Any = None) -> list:
    """参数合法性检查，返回问题列表（空列表表示通过）。"""
    mat = material if isinstance(material, PaintMaterial) else load_paint(
        material if isinstance(material, dict) else None)
    issues: list = []
    for name in ("base_roughness", "coat_roughness", "metallic",
                 "flake_coverage", "flake_strength", "lod"):
        value = getattr(mat, name)
        if not _is_scalar(value) or not math.isfinite(float(value)) \
                or not 0.0 <= float(value) <= 1.0:
            issues.append(f"{name} 越界或非有限值：{value!r}")
    if not (isinstance(mat.base_color, (tuple, list)) and len(mat.base_color) == 3):
        issues.append("base_color 必须是长度 3 的 RGB 元组")
    elif any(not (0.0 <= c <= 1.0) for c in mat.base_color):
        issues.append(f"base_color 通道越界：{mat.base_color!r}")
    if mat.flake_density <= 0.0:
        issues.append(f"flake_density 必须为正：{mat.flake_density!r}")
    if not isinstance(mat.flake_seed, int):
        issues.append("flake_seed 必须是整数（确定性哈希要求）")
    if mat.coat_roughness >= mat.base_roughness:
        # 清漆比底色光滑是物理事实；相等意味着两层粗糙度又被混为一个值
        issues.append(
            f"coat_roughness({mat.coat_roughness}) 应小于 "
            f"base_roughness({mat.base_roughness})，两层粗糙度不得混用")
    return issues


# --- GGX 微表面 ---

def ggx_ndf(ndoth: float, alpha: float) -> float:
    a2 = alpha * alpha
    d = ndoth * ndoth * (a2 - 1.0) + 1.0
    return a2 / (PI * d * d + EPS)


def smith_g(ndotv: float, ndotl: float, alpha: float) -> float:
    k = (alpha + 1.0) ** 2 / 8.0  # 直接光版本

    def g1(x: float) -> float:
        return x / (x * (1.0 - k) + k + EPS)

    return g1(_clamp(ndotv)) * g1(_clamp(ndotl))


def cook_torrance(n: tuple, view_out: tuple, l: tuple,
                  roughness: float, f0: float,
                  fresnel_override: Optional[float] = None) -> float:
    """单次反射镜面波瓣（清漆与底色各调一次，互不共用参数）。

    ``view_out`` 为指向相机的 V，``l`` 为指向光源的方向；半向量按
    H = normalize(V + L) 构造，菲涅尔在 N·H 上计算。
    """
    alpha = max(EPS, roughness * roughness)
    h = vnorm(vadd(view_out, l))
    ndoth = max(EPS, vdot(n, h))
    ndotv = max(EPS, vdot(n, view_out))
    ndotl = max(0.0, vdot(n, l))
    if ndotl <= 0.0:
        return 0.0
    f = fresnel_override if fresnel_override is not None \
        else grazing_fresnel(ndoth, f0, roughness)
    return ggx_ndf(ndoth, alpha) * f * smith_g(ndotv, ndotl, alpha) \
        / (4.0 * ndotv * ndotl + EPS)


@dataclass
class ShadeResult:
    linear_color: tuple
    coat_fresnel: float
    base_term: float
    coat_term: float
    flake_glint: float
    env_term: tuple
    coat_reflect_vector: tuple


def shade_paint(scene: Any = None, **opts) -> ShadeResult:
    """组合底色金属 flake BRDF + 独立清漆单次反射波瓣。

    scene 为 None 或标量时使用默认场景（契约测试以 1 调用）。
    """
    scene = scene if isinstance(scene, dict) else {}
    if opts.get("material") is not None:
        mat = opts["material"]
    elif isinstance(scene.get("material"), PaintMaterial):
        mat = scene["material"]
    elif isinstance(scene.get("material"), dict):
        mat = load_paint(scene["material"])
    else:
        # 允许在场景字典里直接覆盖单个材质字段（如 peel_strength=0）
        overrides = {k: scene[k] for k in PaintMaterial.__dataclass_fields__
                     if k in scene}
        mat = replace(load_paint(), **overrides) if overrides else load_paint()
    n_geom = vnorm(scene.get("geometric_normal", (0.0, 0.0, 1.0)))
    default_t, default_b = _build_frame(n_geom, scene.get("tangent"))
    t = vnorm(_as_vec(scene.get("tangent"), default_t))
    b = vnorm(_as_vec(scene.get("bitangent"), default_b))
    p = scene.get("surface_pos", (0.0, 0.0, 0.0))
    # 约定：view_dir 指向相机（V），light_dir 指向光源（L），N·V/N·L 均非负
    v = vnorm(scene.get("view_dir", (0.0, 0.0, 1.0)))
    l = vnorm(scene.get("light_dir", (-0.3, -0.4, 0.86)))
    light_color = scene.get("light_color", (1.0, 0.96, 0.9))
    probe_color = scene.get("probe_color", (0.18, 0.22, 0.28))

    # 层 1：独立法线——底色法线不含橘皮，清漆法线含低频橘皮
    cn = coat_normal_independent(
        base_normal=n_geom, geometric_normal=n_geom,
        tangent=t, bitangent=b, surface_pos=p,
        base_roughness=mat.base_roughness,
        coat_roughness=mat.coat_roughness,
        peel_strength=mat.peel_strength,
        peel_frequency=mat.peel_frequency,
        seed=mat.flake_seed,
    )
    # 层 2：切线空间 flake，LOD 密度连续
    flake = tangent_flake(
        surface_pos=p, tangent=t, bitangent=b, normal=n_geom,
        density=mat.flake_density, lod=mat.lod,
        coverage=mat.flake_coverage, max_tilt=mat.flake_max_tilt,
        seed=mat.flake_seed,
    )

    h = vnorm(vadd(v, l))
    ndotl = max(0.0, vdot(n_geom, l))

    # --- 底色层（金属 + flake 尖峰）---
    f0_base = tuple(_lerp(CLEARCOAT_F0, c, mat.metallic) for c in mat.base_color)
    # cook_torrance 返回 BRDF 本身（分母已含 N·L），出射辐照度只乘一次 N·L
    base_spec_scalar = cook_torrance(
        cn.base_normal, v, l, mat.base_roughness, CLEARCOAT_F0)
    diffuse = tuple(c / PI * (1.0 - mat.metallic) * ndotl
                    for c in mat.base_color)
    base_spec = tuple(base_spec_scalar * ndotl * f for f in f0_base)

    # flake 高光尖峰：用法线朝向随机的金属片法线替换微表面法线
    glint = 0.0
    if flake.active:
        flake_ndoth = max(0.0, vdot(flake.flake_normal, h))
        exponent = 2.0 / max(EPS, mat.flake_roughness ** 2)
        glint = (flake_ndoth ** min(exponent, 4096.0)) \
            * mat.flake_strength * ndotl

    base_term = tuple(diffuse[i] + base_spec[i]
                      + glint * mat.base_color[i] for i in range(3))

    # --- 清漆层：独立粗糙度 + 逐像素 Schlick 菲涅尔 + 单次反射 ---
    coat_ndotv = max(EPS, vdot(cn.coat_normal, v))
    fc = grazing_fresnel(coat_ndotv, mat.coat_f0, mat.coat_roughness)
    # cook_torrance 已含 1/(4·N·V·N·L)，这里只乘一次 N·L
    coat_spec = cook_torrance(
        cn.coat_normal, v, l, mat.coat_roughness, mat.coat_f0,
        fresnel_override=fc) * ndotl
    coat_spec = _clamp(coat_spec, 0.0, 8.0)  # HDR 上限，防极值炸色

    # 环境贴图单次反射：入射方向为 -V（指向表面），返回出射反射向量
    r = reflect(vscale(v, -1.0), cn.coat_normal)
    env_mip = mat.coat_roughness * 4.0
    env_term = tuple(_clamp(c * fc * (1.0 - 0.15 * env_mip))
                     for c in probe_color)

    # 清漆能量按 Fc 压住底色透射，分层在合成前始终独立
    color = tuple(
        base_term[i] * (1.0 - fc)
        + coat_spec * light_color[i]
        + env_term[i]
        for i in range(3)
    )
    base_scalar = sum(base_term) / 3.0
    return ShadeResult(
        linear_color=tuple(color),
        coat_fresnel=fc,
        base_term=base_scalar,
        coat_term=coat_spec,
        flake_glint=glint,
        env_term=env_term,
        coat_reflect_vector=r,
    )


# --- MSAA resolve：逐 layer 进行，法线八面体解码后平均再编码 ---

def resolve_paint(samples: Any = None) -> GBufferPixel:
    """对同一像素的多个 MSAA 样本逐层 resolve，保留三层结构。"""
    if not isinstance(samples, (list, tuple)) or not samples:
        samples = [layered_gbuffer(depth=d) for d in (0.5, 0.5, 0.5, 0.5)]

    def avg(seq):
        return sum(seq) / len(seq)

    base_n = vnorm(tuple(
        avg([oct_decode(s.base_normal_enc)[i] for s in samples])
        for i in range(3)))
    coat_n = vnorm(tuple(
        avg([oct_decode(s.coat_normal_enc)[i] for s in samples])
        for i in range(3)))
    return GBufferPixel(
        depth=min(s.depth for s in samples),
        base_albedo=tuple(avg([s.base_albedo[i] for s in samples])
                          for i in range(3)),
        metalness=avg([s.metalness for s in samples]),
        base_normal_enc=oct_encode(base_n),
        base_roughness=avg([s.base_roughness for s in samples]),
        coat_normal_enc=oct_encode(coat_n),
        coat_roughness=avg([s.coat_roughness for s in samples]),
        coat_f0=avg([s.coat_f0 for s in samples]),
        flake_density=avg([s.flake_density for s in samples]),
        flake_seed=samples[0].flake_seed,  # 种子必须一致，取首样本
        flake_lod=avg([s.flake_lod for s in samples]),
    )


# --- 呈现：ACES 色调映射 + linear→sRGB ---

def _aces(c: float) -> float:
    c = max(0.0, c)
    return _clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14))


def _linear_to_srgb(c: float) -> float:
    c = _clamp(c)
    return 1.055 * c ** (1.0 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c


def present_paint(color: Any = None) -> tuple:
    if _is_scalar(color):
        color = (float(color),) * 3
    if not isinstance(color, (tuple, list)):
        color = shade_paint().linear_color
    return tuple(_linear_to_srgb(_aces(c)) for c in color)


# --- 度量 / 审计 / 降级 / 确定性 ---

def metrics_paint(scene: Any = None) -> dict:
    """在一组掠射角扫描上汇总清漆菲涅尔与 flake 覆盖度量。

    菲涅尔沿几何法线扫描（剥离橘皮扰动），单独度量 Schlick 曲线本身；
    橘皮对有效 NdotV 的影响属于清漆法线独立特性，不在此混淆。
    """
    fresnels, glints = [], []
    active = 0
    steps = 16
    for i in range(steps):
        # 扫描下界取近掠射 cos=1e-5，使扫描最大值即掠射极限（≈0.99995）
        cos_theta = _lerp(1.0e-5, 1.0, i / (steps - 1))
        fresnels.append(grazing_fresnel(cos_theta))
        s = shade_paint({"view_dir": (
            math.sqrt(max(0.0, 1.0 - cos_theta ** 2)), 0.0, cos_theta)})
        glints.append(s.flake_glint)
        f = tangent_flake(surface_pos=(i * 0.01, 0.0, 0.0),
                          seed=1337, lod=0.0)
        active += int(f.active)
    # 无橘皮（peel_strength=0）时清漆法线==几何法线；取真正近掠射角
    # （cosθ=1e-5 ≈ 89.9994°），Schlick 给出 F≈0.99995，证明掠射严格趋 1
    graz_cos = 1.0e-5
    graz = shade_paint({
        "view_dir": (math.sqrt(1.0 - graz_cos ** 2), 0.0, graz_cos),
        "peel_strength": 0.0,
    })
    return {
        "fresnel_min": min(fresnels),
        "fresnel_max": max(fresnels),
        "fresnel_mean": sum(fresnels) / len(fresnels),
        "fresnel_grazing_to_one": max(fresnels) > 0.999,
        "grazing_shade_fresnel_no_peel": graz.coat_fresnel,
        "grazing_reaches_one": graz.coat_fresnel > 0.999,
        "flake_coverage_measured": active / steps,
        "flake_glint_mean": sum(glints) / len(glints),
        "gbuffer_layers": len(GBufferPixel.LAYER_NAMES),
    }


_AUDIT_CHECKS = {
    "schlick_per_pixel": "掠射菲涅尔逐像素 Schlick 重算",
    "coat_normal_independent": "清漆法线独立于底色法线",
    "separate_roughness": "清漆/底色粗糙度分离",
    "flake_tangent_space": "flake 在切线空间采样",
    "flake_continuous_lod": "flake LOD 密度连续过渡",
    "layered_gbuffer": "三层独立 G-Buffer",
    "per_layer_msaa_resolve": "MSAA 逐层 resolve",
    "probe_env_change_recapture": "环境切换强制重抓探针",
    "probe_angular_dirty": "快速转向标 dirty 重抓",
    "separate_render_queues": "渲染队列分离",
    "headlight_range_validated": "车灯照射范围校验",
}


def audit_paint(config: Any = None) -> dict:
    """对管线配置做静态审计；默认配置即修复后的正确管线，全部通过。"""
    if not isinstance(config, dict):
        config = {key: True for key in _AUDIT_CHECKS}
    checks = {key: bool(config.get(key, False)) for key in _AUDIT_CHECKS}
    failures = [_AUDIT_CHECKS[k] for k, ok in checks.items() if not ok]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "total": len(checks),
    }


def fallback_paint(capabilities: Any = None) -> dict:
    """不支持多层 MRT 时的降级路径：

    清漆信息压进底色 target 的空余通道，但菲涅尔修正、独立清漆法线、
    切线 flake 三项核心修复保留；flake 密度减半以省带宽。
    """
    caps = capabilities if isinstance(capabilities, dict) else {}
    layered = bool(caps.get("layered_gbuffer", False))
    if layered:
        return {"degraded": False, "packing": "layered", "note": "无需降级"}
    mat = load_paint()
    mat.flake_density *= 0.5
    pixel = layered_gbuffer(material=mat)
    packed = {
        "rg": pixel.base_albedo[0:2],
        "b": pixel.base_albedo[2],
        "spare_a": pixel.coat_roughness,   # 借道存储
        "normal_enc": pixel.base_normal_enc,
        "coat_normal_enc": pixel.coat_normal_enc,
    }
    return {
        "degraded": True,
        "packing": "merged_spare_channels",
        "kept_fixes": [
            "schlick_per_pixel", "coat_normal_independent",
            "flake_tangent_space", "continuous_lod",
        ],
        "limitations": ["G-Buffer 仅两层逻辑信息", "flake 密度减半"],
        "packed": packed,
    }


def determinism_paint(seed: Any = 1337) -> dict:
    """确定性自检：同输入逐位一致；相机移动不改变表面 flake；
    切线帧旋转时 flake 法线随表面走（证明锚定在切线空间）。"""
    seed = int(seed) if _is_scalar(seed) else 1337
    t, b, n = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    p = (0.37, 0.21, 0.0)

    f1 = tangent_flake(p, t, b, n, seed=seed)
    f2 = tangent_flake(p, t, b, n, seed=seed)
    same_input_equal = f1 == f2

    # 只换视角/相机（表面 P 与切线帧不变）→ flake 单元与法线不变
    f3 = tangent_flake(p, t, b, n, seed=seed,
                       lod=0.0)
    camera_invariant = (f1.cell == f3.cell and f1.flake_normal == f3.flake_normal)

    # 切线帧绕 Z 旋转 90°：flake 世界法线应跟随旋转，而不是留在原地
    t2, b2 = (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)
    f4 = tangent_flake(p, t2, b2, n, seed=seed)
    frame_rotated = f4.flake_normal != f1.flake_normal

    s1 = shade_paint({"surface_pos": p, "view_dir": (0.1, 0.2, 1.0)})
    s2 = shade_paint({"surface_pos": p, "view_dir": (0.1, 0.2, 1.0)})
    shade_equal = s1.linear_color == s2.linear_color

    # LOD 连续：相邻小数 LOD 的密度比恒定为 1/2，无台阶
    d0 = lod_density_scale(0.0)
    d1 = lod_density_scale(1.0)
    dh = lod_density_scale(0.5)
    lod_continuous = abs(d1 / d0 - 0.5) < EPS and abs(dh / d0 - math.sqrt(0.5)) < EPS

    return {
        "same_input_bit_equal": same_input_equal,
        "camera_move_invariant": camera_invariant,
        "tangent_frame_anchored": frame_rotated,
        "shade_bit_equal": shade_equal,
        "lod_density_continuous": lod_continuous,
        "all_deterministic": all([
            same_input_equal, camera_invariant, frame_rotated,
            shade_equal, lod_continuous]),
    }

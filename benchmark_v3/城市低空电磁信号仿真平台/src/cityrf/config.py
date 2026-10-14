"""Strict SI-unit configuration; single-port TX and uniform RX arrays in v1."""
from typing import Literal
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Component(Spec):
    # 平台原生 7 种 + v2 基带新增 6 种（gfsk/16qam/64qam/ook/fhss/lfm），共 13 种
    kind: Literal["tone", "noise", "bpsk", "qpsk", "fsk", "ofdm", "chirp",
                  "gfsk", "16qam", "64qam", "ook", "fhss", "lfm"] = "qpsk"
    weight: float = Field(default=1, gt=0)
    offset_hz: float = 0
    bandwidth_hz: float = Field(default=100000, gt=0)
    symbol_rate_hz: float = Field(default=25000, gt=0)
    phase_rad: float = 0
    pulse_period_s: float | None = Field(default=None, gt=0)
    duty_cycle: float = Field(default=1, gt=0, le=1)
    pulse_start_s: float = 0
    chirp_period_s: float = Field(default=.001, gt=0)


class Antenna(Spec):
    geometry_type: Literal["cross","rectangular"] = "cross"
    num_rows: int | None = Field(default=None, ge=1, le=16)
    num_cols: int | None = Field(default=None, ge=1, le=16)
    horizontal_elements: int | None = Field(default=None, ge=2, le=16)
    vertical_elements: int | None = Field(default=None, ge=2, le=16)
    vertical_spacing_wavelengths: float = Field(default=.5, gt=0, le=2)
    horizontal_spacing_wavelengths: float = Field(default=.5, gt=0, le=2)
    pattern: Literal["iso"] = "iso"
    polarization: Literal["V"] = "V"
    gain_error_std_db: float = Field(default=0, ge=0, le=6)
    phase_error_std_deg: float = Field(default=0, ge=0, le=45)

    @model_validator(mode="before")
    @classmethod
    def migrate_linear(cls, value):
        if not isinstance(value,dict):
            return value
        old={"elements","spacing_wavelengths"} & value.keys()
        new={"num_rows","num_cols","vertical_spacing_wavelengths","horizontal_spacing_wavelengths"} & value.keys()
        if old and (new or value.get("geometry_type")=="cross"):
            raise ValueError("cannot mix legacy linear and planar array fields")
        if old:
            value=dict(value)
            value["geometry_type"]="rectangular"
            value["num_rows"]=1
            value["num_cols"]=value.pop("elements",4)
            spacing=value.pop("spacing_wavelengths",.5)
            value["vertical_spacing_wavelengths"]=spacing
            value["horizontal_spacing_wavelengths"]=spacing
        value=dict(value)
        rectangle=any(value.get(k) is not None for k in ("num_rows","num_cols"))
        arms=any(value.get(k) is not None for k in ("horizontal_elements","vertical_elements"))
        kind=value.get("geometry_type","rectangular" if rectangle else "cross")
        value["geometry_type"]=kind
        if kind=="cross":
            if rectangle:
                raise ValueError("cross array uses arm counts, not rectangle rows/columns")
            if value.get("horizontal_elements") is None:
                value["horizontal_elements"]=8
            if value.get("vertical_elements") is None:
                value["vertical_elements"]=8
        elif kind=="rectangular":
            if arms:
                raise ValueError("rectangular array cannot define cross arms")
            if value.get("num_rows") is None:
                value["num_rows"]=2
            if value.get("num_cols") is None:
                value["num_cols"]=2
        return value

    @model_validator(mode="after")
    def size_bound(self):
        if self.geometry_type=="cross" and (self.horizontal_elements%2 or self.vertical_elements%2):
            raise ValueError("cross arms must have even counts; no shared central element")
        if self.elements>16:
            raise ValueError("array supports at most 16 elements")
        return self

    @property
    def elements(self):
        return self.horizontal_elements+self.vertical_elements if self.geometry_type=="cross" else self.num_rows*self.num_cols


class Emitter(Spec):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
    position_m: tuple[float,float,float]
    orientation_rad: tuple[float,float,float] = (0,0,0)
    platform_id: str | None = None
    role: Literal["communication", "interference", "background"] = "communication"
    # 上限 130 dBm：卡 5 的窗外强源是 1 MW 级监视雷达（+90 dBm 量级），
    # 80 dBm 的旧上限会把它卡在完全线性区、阻塞效应做不出来。
    power_dbm: float = Field(default=0, ge=-100, le=130)
    rf_offset_hz: float = 0
    # 本发射源的绝对载频。None = 与场景 carrier_frequency_hz 同频。
    # 非 None 时：自由空间信道按该频率算波长/路径损耗，接收机在天线口看到
    # (carrier_hz - 场景载频) 的频移 ⇒ 可表示「窗外异载频强源」（卡 5/卡 8）。
    # 注意：基带离散表示的频率是模 fs 的，真正起作用的是它在通带外的位置
    # （被通道滤波器抑制）与它对前端 IMD3/软限幅的贡献。
    carrier_hz: float | None = Field(default=None, ge=1e8, le=1e11)
    velocity_mps: tuple[float,float,float] = (0,0,0)
    # 预设信号源：填 v3 辐射体名（如 "Bluetooth_FHSS"）时，generate 用该辐射体生成真实时域信号；
    # None 表示自由模式，用 components 里的调制与参数。
    preset: str | None = None
    components: list[Component] = Field(default_factory=lambda: [Component()], min_length=1, max_length=16)


class Frontend(Spec):
    noise_figure_db: float = Field(default=5, ge=0, le=40)
    temperature_k: float = Field(default=290, gt=0)
    noise_enabled: bool = True
    colored_noise_enabled: bool = False
    colored_noise_db: float = Field(default=0, ge=-40, le=40)
    colored_bandwidth_hz: float = Field(default=200000, gt=0)
    impulse_noise_enabled: bool = False
    impulse_rate_hz: float = Field(default=200, gt=0)
    impulse_power_db: float = Field(default=20, ge=0, le=60)
    impulse_width_us: float = Field(default=5, gt=0, le=1000)
    phase_noise_enabled: bool = False
    phase_noise_deg: float = Field(default=1, ge=0, le=30)
    correlated_noise_enabled: bool = False
    noise_correlation: float = Field(default=0.3, ge=0, le=1)
    imd3_enabled: bool = False
    iip3_dbm: float = Field(default=10, ge=-20, le=40)
    gain_db: float = Field(default=0, ge=-60, le=100)
    soft_limit_input_dbm: float | None = None
    filter_taps: int = Field(default=129, ge=17, le=1025)

    @model_validator(mode="after")
    def odd_taps(self):
        if self.filter_taps % 2 == 0:
            raise ValueError("filter_taps must be odd")
        return self


class Receiver(Spec):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
    position_m: tuple[float,float,float]
    orientation_rad: tuple[float,float,float] = (0,0,0)
    antenna: Antenna = Field(default_factory=Antenna)
    tuning_offset_hz: float = 0
    bandwidth_hz: float = Field(default=800000, gt=0)
    frontend: Frontend = Field(default_factory=Frontend)


class Propagation(Spec):
    backend: Literal["free_space", "tapped", "sionna_rt"] = "free_space"
    rt_variant: Literal["llvm_ad_mono_polarized", "cuda_ad_mono_polarized"] = "llvm_ad_mono_polarized"
    scene: str = "munich"   # "munich" | "empty" | 自定义 Mitsuba 场景文件路径（如 priors/hangzhou_geometry/scene.xml）
    max_depth: int = Field(default=3, ge=0, le=10)
    samples_per_src: int = Field(default=100000, ge=100, le=10000000)
    max_num_paths_per_src: int = Field(default=100000, ge=100, le=1000000)
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    tap_delays_s: list[float] = Field(default_factory=lambda: [0])
    tap_gains_real: list[float] = Field(default_factory=lambda: [1])
    tap_gains_imag: list[float] = Field(default_factory=lambda: [0])

    @model_validator(mode="after")
    def taps_valid(self):
        if not (len(self.tap_delays_s) == len(self.tap_gains_real) == len(self.tap_gains_imag)) or not self.tap_delays_s:
            raise ValueError("tap lengths must match and not be empty")
        if any(t < 0 or t > .001 for t in self.tap_delays_s):
            raise ValueError("tap delays must be between 0 and 1 ms")
        return self


class Task(Spec):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
    receiver_ids: list[str] = Field(min_length=1)
    desired_emitter_ids: list[str] = Field(min_length=1)
    protected_center_offset_hz: float = 0
    protected_bandwidth_hz: float = Field(default=100000, gt=0)
    instruction: str = "Analyze interference using the public IQ observations. Distinguish paths from physical emitters."


class Scenario(Spec):
    schema_version: Literal["1.2"] = "1.2"
    name: str = "static-city"
    seed: int = Field(default=42, ge=0, le=2147483647)
    duration_s: float = Field(default=.05, gt=0, le=.5)
    sample_rate_hz: float = Field(default=1e6, ge=10000, le=40e6)
    carrier_frequency_hz: float = Field(default=2.4e9, ge=1e8, le=1e11)
    max_working_mb: int = Field(default=2048, ge=64, le=65536)
    emitters: list[Emitter] = Field(min_length=1, max_length=32)
    receivers: list[Receiver] = Field(min_length=1, max_length=16)
    propagation: Propagation = Field(default_factory=Propagation)
    tasks: list[Task] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_v1(cls, value):
        if isinstance(value,dict) and value.get("schema_version") in ("1.0","1.1"):
            value=dict(value)
            previous=value["schema_version"]
            value["schema_version"]="1.2"
            receivers=[]
            for item in value.get("receivers",[]):
                if isinstance(item,dict):
                    item=dict(item)
                    antenna=dict(item.get("antenna",{}))
                    if previous=="1.0" and not {"num_rows","num_cols"} & antenna.keys():
                        antenna.setdefault("elements",4)
                    elif previous=="1.1":
                        antenna.setdefault("geometry_type","rectangular")
                    item["antenna"]=antenna
                receivers.append(item)
            value["receivers"]=receivers
        return value

    @model_validator(mode="after")
    def contracts(self):
        ids = [x.id for x in self.emitters + self.receivers]
        if len(ids) != len(set(ids)):
            raise ValueError("device IDs must be globally unique")
        if len({t.id for t in self.tasks}) != len(self.tasks):
            raise ValueError("task IDs must be unique")
        if abs(self.duration_s*self.sample_rate_hz-round(self.duration_s*self.sample_rate_hz)) > 1e-6:
            raise ValueError("duration must contain an integer number of samples")
        if self.num_samples < 1:
            raise ValueError("capture must contain at least one sample")
        for tx in self.emitters:
            for c in tx.components:
                half = 0 if c.kind == "tone" and c.pulse_period_s is None else c.bandwidth_hz/2
                if abs(c.offset_hz)+half >= .48*self.sample_rate_hz:
                    raise ValueError("waveform exceeds Nyquist guard band")
                if c.symbol_rate_hz > self.sample_rate_hz/2:
                    raise ValueError("symbol rate too high")
                if c.duty_cycle != 1 and c.pulse_period_s is None:
                    raise ValueError("pulse duty requires period")
        for rx in self.receivers:
            if abs(rx.tuning_offset_hz)+rx.bandwidth_hz/2 >= self.sample_rate_hz/2:
                raise ValueError("receiver passband exceeds Nyquist")
        if self.propagation.backend == "sionna_rt":
            if any(r.antenna != self.receivers[0].antenna for r in self.receivers):
                raise ValueError("Sionna v1 adapter requires identical receiver array specs")
        for task in self.tasks:
            if len(set(task.receiver_ids)) != len(task.receiver_ids) or len(set(task.desired_emitter_ids)) != len(task.desired_emitter_ids):
                raise ValueError("task receiver/source IDs must be unique")
            if not set(task.receiver_ids) <= {r.id for r in self.receivers}:
                raise ValueError("unknown task receiver")
            if not set(task.desired_emitter_ids) <= {t.id for t in self.emitters}:
                raise ValueError("unknown desired emitter")
        for tx in self.emitters:
            for rx in self.receivers:
                if math.dist(tx.position_m, rx.position_m) < .1:
                    raise ValueError("TX/RX separation must be >= 0.1 m")
        return self

    @property
    def num_samples(self):
        return round(self.duration_s*self.sample_rate_hz)

"""Craft-local flux transducer: pump, quantify, reverse, and meter rear-hemi energy.

Sits between internal craft systems and the rear-hemi void bubble. Unlike
global curvature-flux bookkeeping, the transducer keeps an **explicit local
ledger** of three co-stored channels:

  1. Electrostatic — dual charge envelopes / capacitive gap
  2. Twist — angular-momentum / Hopf-flux flywheel accumulation
  3. Geometric tension — rubber-band packing of the stretched void (η, rear bias)

**Bidirectional control (motor–generator + gearbox):**

- **Store / READY** — actively *pre-charges* the dual envelopes and *pre-twists*
  the rear-hemi flux flywheel (parametric pump), not just passive collection.
- **Release** — reverts flux channels and meters intensity so energy dumps as a
  directional slingshot impulse rather than isotropic dissipation.

Analog only — effective-metric control, not literal vacuum engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TransducerConfig:
    """Hardware / control parameters for the flux transducer."""

    capacity: float = 1.0
    """Maximum total stored energy (normalized units)."""
    w_electrostatic: float = 0.40
    w_twist: float = 0.35
    w_geometric: float = 0.25
    """Relative channel weights for total ledger (should sum ~1)."""
    store_efficiency: float = 0.92
    release_efficiency: float = 0.88
    """Fraction of sensed / pumped power that actually banks / dumps."""
    intensity: float = 1.0
    """Throttle 0–1: release dump hardness (also scales pump drive if unset)."""
    max_intensity: float = 1.0
    reverse_gain: float = 1.35
    """Channel-reversion amplification of directed forward dump."""
    leak_rate: float = 0.02
    """Slow coast leakage when idle."""
    sense_gain: float = 1.0
    """Scale on instantaneous power read from fields."""
    twist_coupling: float = 0.55
    """How strongly shell vibration velocity feeds the twist channel."""
    geometric_coupling: float = 0.70
    """How strongly void order η feeds geometric tension."""
    hopf_channel_gain: float = 0.45
    """Strength of flux-channel polarity field for Hopfion modulation."""
    axis: str = "x"

    # --- Active pumping (store phase): pre-charge + pre-twist ---
    enable_precharge: bool = True
    enable_pretwist: bool = True
    precharge_rate: float = 0.55
    """Electrostatic energy injected per unit time at full pump (locked)."""
    pretwist_rate: float = 0.50
    """Twist energy injected per unit time at full pump (locked)."""
    geometric_pump_rate: float = 0.12
    """Small geometric-tension co-pump from rear packing stretch."""
    pump_efficiency: float = 0.90
    """Motor efficiency of active inject (ledger fraction actually banked)."""
    pump_intensity: float = 1.0
    """Pump throttle 0–1 (independent of release intensity when set)."""
    target_energy: float = 0.95
    """Fraction of capacity at which the rear hemi is 'ready'."""
    target_electrostatic: float | None = None
    """Optional absolute ES target; default = target_energy * capacity * w_es/wsum."""
    target_twist: float | None = None
    """Optional absolute twist target; default from weights."""
    pump_requires_lock: bool = True
    """If True, active pump gates on ψ ≥ psi_pump_threshold."""
    psi_pump_threshold: float = 0.55
    charge_boost_gain: float = 0.85
    """Maps pre-charge drive → dual-shell |σ| / |E| field boost."""
    pretwist_gain: float = 0.90
    """Maps pre-twist drive → azimuthal velocity / vibration boost."""
    phase_assist_gain: float = 0.25
    """How much active pumping pulls ψ toward lock (store only)."""
    ready_hold_leak: float = 0.005
    """Tiny leakage while holding READY (transducer top-up counters this)."""
    topup_rate: float = 0.35
    """Active top-up rate while READY to hold ledger at target."""


@dataclass
class EnergyLedger:
    """Explicit three-channel energy bookkeeping (craft-local)."""

    electrostatic: float = 0.0
    twist: float = 0.0
    geometric: float = 0.0

    @property
    def total(self) -> float:
        return float(self.electrostatic + self.twist + self.geometric)

    def as_dict(self) -> dict[str, float]:
        return {
            "electrostatic": float(self.electrostatic),
            "twist": float(self.twist),
            "geometric": float(self.geometric),
            "total": self.total,
        }

    def clip_capacity(self, capacity: float) -> EnergyLedger:
        tot = self.total
        if tot <= capacity or tot <= 1e-15:
            return self
        scale = capacity / tot
        return EnergyLedger(
            electrostatic=self.electrostatic * scale,
            twist=self.twist * scale,
            geometric=self.geometric * scale,
        )

    def scale(self, factor: float) -> EnergyLedger:
        return EnergyLedger(
            electrostatic=self.electrostatic * factor,
            twist=self.twist * factor,
            geometric=self.geometric * factor,
        )

    def add(self, other: EnergyLedger) -> EnergyLedger:
        return EnergyLedger(
            electrostatic=self.electrostatic + other.electrostatic,
            twist=self.twist + other.twist,
            geometric=self.geometric + other.geometric,
        )


@dataclass
class ChannelState:
    """Flux-channel polarity and throttle.

    direction: +1 inbound (store into rear hemi), −1 reverted (forward dump),
               0 idle / coast.
    """

    direction: float = 0.0
    intensity: float = 0.0
    reversed: bool = False

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "direction": float(self.direction),
            "intensity": float(self.intensity),
            "reversed": bool(self.reversed),
        }


@dataclass
class PumpCommand:
    """Active motor drive during store / ready (pre-charge + pre-twist)."""

    precharge_power: float = 0.0
    """Instantaneous ES inject rate (energy / time)."""
    pretwist_power: float = 0.0
    """Instantaneous twist inject rate."""
    geometric_power: float = 0.0
    pump_active: bool = False
    ready: bool = False
    charge_boost: float = 0.0
    """Multiplicative dual-shell charge/|E| boost (0 = none)."""
    twist_drive: float = 0.0
    """Azimuthal pre-twist drive strength."""
    vibration_amp_boost: float = 0.0
    """Extra charged-vibration amplitude while pumping."""
    phase_assist: float = 0.0
    """Pull toward phase lock while pumping at resonance."""
    deficit_es: float = 0.0
    deficit_twist: float = 0.0

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "precharge_power": float(self.precharge_power),
            "pretwist_power": float(self.pretwist_power),
            "geometric_power": float(self.geometric_power),
            "pump_active": bool(self.pump_active),
            "ready": bool(self.ready),
            "charge_boost": float(self.charge_boost),
            "twist_drive": float(self.twist_drive),
            "vibration_amp_boost": float(self.vibration_amp_boost),
            "phase_assist": float(self.phase_assist),
            "deficit_es": float(self.deficit_es),
            "deficit_twist": float(self.deficit_twist),
        }


@dataclass
class TransducerReading:
    """Instantaneous sensed power split across the three channels."""

    power_electrostatic: float
    power_twist: float
    power_geometric: float
    power_total: float
    rear_mask_mean: float = 0.0

    def as_ledger_increment(self, dt: float, efficiency: float) -> EnergyLedger:
        s = efficiency * dt
        return EnergyLedger(
            electrostatic=self.power_electrostatic * s,
            twist=self.power_twist * s,
            geometric=self.power_geometric * s,
        )


@dataclass
class TransducerReport:
    """Full transducer output for one control step."""

    ledger: EnergyLedger
    channels: ChannelState
    reading: TransducerReading
    dump: EnergyLedger
    """Energy extracted this step (nonzero on release)."""
    directed_impulse: float
    """Scalar forward impulse after reversion × intensity × efficiency."""
    pump: PumpCommand | None = None
    """Active pre-charge / pre-twist command (store / ready)."""
    pumped: EnergyLedger | None = None
    """Energy actively injected this step (motor path)."""
    channel_field: NDArray[np.floating] | None = None
    """Spatial flux-channel polarity field (for Hopf / defect modulation)."""
    flow_intensity_field: NDArray[np.floating] | None = None
    """Spatial release intensity map (rear-weighted)."""
    charge_boost_field: NDArray[np.floating] | None = None
    """Spatial dual-shell pre-charge boost map."""
    pretwist_velocity: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None
    """Azimuthal (vx, vy) pre-twist velocity injected into the medium."""

    def as_dict(self) -> dict:
        out = {
            "ledger": self.ledger.as_dict(),
            "channels": self.channels.as_dict(),
            "reading": {
                "power_electrostatic": self.reading.power_electrostatic,
                "power_twist": self.reading.power_twist,
                "power_geometric": self.reading.power_geometric,
                "power_total": self.reading.power_total,
            },
            "dump": self.dump.as_dict(),
            "directed_impulse": float(self.directed_impulse),
        }
        if self.pump is not None:
            out["pump"] = self.pump.as_dict()
        if self.pumped is not None:
            out["pumped"] = self.pumped.as_dict()
        return out


def rear_hemi_weight(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    axis: str = "x",
    softness: float = 0.25,
) -> NDArray[np.floating]:
    """Smooth weight ≈1 on rear hemi (negative axis)."""
    coord = x if axis == "x" else y
    return 1.0 / (1.0 + np.exp(coord / max(softness, 1e-6)))


def channel_targets(cfg: TransducerConfig | None = None) -> tuple[float, float, float]:
    """Absolute (ES, twist, geometric) targets for ready state."""
    cfg = cfg or TransducerConfig()
    wsum = cfg.w_electrostatic + cfg.w_twist + cfg.w_geometric
    if wsum <= 1e-15:
        wsum = 1.0
    budget = cfg.target_energy * cfg.capacity
    t_es = (
        cfg.target_electrostatic
        if cfg.target_electrostatic is not None
        else budget * cfg.w_electrostatic / wsum
    )
    t_tw = (
        cfg.target_twist
        if cfg.target_twist is not None
        else budget * cfg.w_twist / wsum
    )
    t_geo = budget * cfg.w_geometric / wsum
    return float(t_es), float(t_tw), float(t_geo)


def is_ready(ledger: EnergyLedger, cfg: TransducerConfig | None = None) -> bool:
    """True when ledger meets total (and optional per-channel) targets."""
    cfg = cfg or TransducerConfig()
    if ledger.total + 1e-12 < cfg.target_energy * cfg.capacity:
        return False
    t_es, t_tw, _ = channel_targets(cfg)
    # Soft: allow 8% undershoot on individual channels if total is met
    if ledger.electrostatic + 1e-12 < 0.92 * t_es:
        return False
    if ledger.twist + 1e-12 < 0.92 * t_tw:
        return False
    return True


def sense_energy_channels(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    e_field: NDArray[np.floating],
    charge_density: NDArray[np.floating],
    velocity: NDArray[np.floating] | None,
    void_amplitude_field: NDArray[np.floating],
    psi: float,
    dx: float = 0.1,
    cfg: TransducerConfig | None = None,
) -> TransducerReading:
    """Quantify instantaneous power available to the three ledger channels.

    Electrostatic ~ ∫ ½ E² · rear  (capacitive energy density proxy)
    Twist         ~ ∫ |v_shell| · |σ| · rear · ψ  (flux-flywheel spin-up)
    Geometric     ~ ∫ η² · rear · elongation proxy (rubber-band tension)
    """
    cfg = cfg or TransducerConfig()
    rear = rear_hemi_weight(x, y, axis=cfg.axis)
    area = dx**2

    e2 = e_field**2
    p_es = float(np.sum(0.5 * e2 * rear) * area) * cfg.w_electrostatic * cfg.sense_gain

    if velocity is None:
        velocity = np.zeros_like(e_field)
    p_tw = (
        float(np.sum(np.abs(velocity) * np.abs(charge_density) * rear) * area)
        * cfg.w_twist
        * cfg.twist_coupling
        * cfg.sense_gain
        * max(psi, 0.0)
    )

    p_geo = (
        float(np.sum(void_amplitude_field**2 * rear) * area)
        * cfg.w_geometric
        * cfg.geometric_coupling
        * cfg.sense_gain
        * max(psi, 0.0)
    )

    scale = 1.0 / (1.0 + 0.15 * (p_es + p_tw + p_geo))
    p_es *= scale
    p_tw *= scale
    p_geo *= scale

    total = p_es + p_tw + p_geo
    wsum = cfg.w_electrostatic + cfg.w_twist + cfg.w_geometric
    if total > 1e-15 and wsum > 1e-15:
        blend = 0.55
        p_es = (1.0 - blend) * p_es + blend * total * (cfg.w_electrostatic / wsum)
        p_tw = (1.0 - blend) * p_tw + blend * total * (cfg.w_twist / wsum)
        p_geo = (1.0 - blend) * p_geo + blend * total * (cfg.w_geometric / wsum)

    return TransducerReading(
        power_electrostatic=p_es,
        power_twist=p_tw,
        power_geometric=p_geo,
        power_total=p_es + p_tw + p_geo,
        rear_mask_mean=float(np.mean(rear)),
    )


def compute_pump_command(
    ledger: EnergyLedger,
    psi: float,
    cfg: TransducerConfig | None = None,
    mode: str = "store",
    pump_intensity: float | None = None,
) -> PumpCommand:
    """Compute active pre-charge / pre-twist drive for store or ready hold.

    Deficit-seeking controller: pumps harder when below channel targets,
    tapers to zero at ready, and applies light top-up while holding READY.
    """
    cfg = cfg or TransducerConfig()
    thr = cfg.pump_intensity if pump_intensity is None else float(
        np.clip(pump_intensity, 0.0, cfg.max_intensity)
    )
    mode_l = mode.lower()
    t_es, t_tw, t_geo = channel_targets(cfg)
    ready = is_ready(ledger, cfg)

    lock_ok = (not cfg.pump_requires_lock) or (psi >= cfg.psi_pump_threshold)
    if mode_l not in ("store", "ready", "nucleate") or not lock_ok or thr <= 1e-12:
        return PumpCommand(ready=ready)

    # Deficits (how much more we want on each channel)
    def_es = max(0.0, t_es - ledger.electrostatic)
    def_tw = max(0.0, t_tw - ledger.twist)
    def_geo = max(0.0, t_geo - ledger.geometric)

    if mode_l == "ready" and ready:
        # Hold: light top-up proportional to small deficit / leak
        scale = cfg.topup_rate * thr
        p_es = scale * min(1.0, def_es / max(t_es, 1e-9)) if cfg.enable_precharge else 0.0
        p_tw = scale * min(1.0, def_tw / max(t_tw, 1e-9)) if cfg.enable_pretwist else 0.0
        p_geo = 0.5 * scale * min(1.0, def_geo / max(t_geo, 1e-9))
        active = (p_es + p_tw + p_geo) > 1e-9
    elif mode_l == "nucleate":
        # Soft prime only
        p_es = 0.25 * cfg.precharge_rate * thr * psi if cfg.enable_precharge else 0.0
        p_tw = 0.20 * cfg.pretwist_rate * thr * psi if cfg.enable_pretwist else 0.0
        p_geo = 0.15 * cfg.geometric_pump_rate * thr * psi
        active = True
    else:
        # Full store pump: rate × intensity × lock quality × deficit fraction
        lock = float(np.clip(psi, 0.0, 1.0))
        frac_es = min(1.0, def_es / max(0.15 * cfg.capacity, 1e-9))
        frac_tw = min(1.0, def_tw / max(0.15 * cfg.capacity, 1e-9))
        frac_geo = min(1.0, def_geo / max(0.15 * cfg.capacity, 1e-9))
        p_es = (
            cfg.precharge_rate * thr * lock * (0.25 + 0.75 * frac_es)
            if cfg.enable_precharge
            else 0.0
        )
        p_tw = (
            cfg.pretwist_rate * thr * lock * (0.25 + 0.75 * frac_tw)
            if cfg.enable_pretwist
            else 0.0
        )
        p_geo = cfg.geometric_pump_rate * thr * lock * (0.2 + 0.8 * frac_geo)
        # If already ready mid-store, taper heavily
        if ready:
            p_es *= 0.15
            p_tw *= 0.15
            p_geo *= 0.15
        active = (p_es + p_tw + p_geo) > 1e-9

    # Field-level drive amplitudes (for charge envelopes + pretwist velocity)
    charge_boost = cfg.charge_boost_gain * thr * min(1.0, p_es / max(cfg.precharge_rate, 1e-9))
    twist_drive = cfg.pretwist_gain * thr * min(1.0, p_tw / max(cfg.pretwist_rate, 1e-9))
    vib_boost = 0.55 * twist_drive + 0.25 * charge_boost
    phase_assist = cfg.phase_assist_gain * thr * float(np.clip(psi, 0.0, 1.0)) if active else 0.0

    return PumpCommand(
        precharge_power=float(p_es),
        pretwist_power=float(p_tw),
        geometric_power=float(p_geo),
        pump_active=active,
        ready=ready,
        charge_boost=float(charge_boost),
        twist_drive=float(twist_drive),
        vibration_amp_boost=float(vib_boost),
        phase_assist=float(phase_assist),
        deficit_es=float(def_es),
        deficit_twist=float(def_tw),
    )


def apply_active_pump(
    ledger: EnergyLedger,
    pump: PumpCommand,
    dt: float,
    cfg: TransducerConfig | None = None,
) -> tuple[EnergyLedger, EnergyLedger]:
    """Inject pre-charge / pre-twist energy into the ledger. Returns (new, pumped)."""
    cfg = cfg or TransducerConfig()
    if not pump.pump_active:
        return ledger, EnergyLedger()
    eff = cfg.pump_efficiency
    pumped = EnergyLedger(
        electrostatic=pump.precharge_power * dt * eff,
        twist=pump.pretwist_power * dt * eff,
        geometric=pump.geometric_power * dt * eff,
    )
    return ledger.add(pumped).clip_capacity(cfg.capacity), pumped


def precharge_boost_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    charge_boost: float,
    cfg: TransducerConfig | None = None,
) -> NDArray[np.floating]:
    """Spatial multiplicative boost for dual-shell |σ| / |E| during pre-charge.

    Peaked on rear hemi shell (craft pumps the rear reservoir harder).
    """
    cfg = cfg or TransducerConfig()
    if charge_boost <= 1e-12:
        return np.zeros_like(x)
    rear = rear_hemi_weight(x, y, axis=cfg.axis)
    r = np.sqrt(x**2 + y**2)
    wall = np.exp(-((r - 1.0) ** 2) / (2.0 * 0.18**2))
    return charge_boost * wall * (0.35 + 0.65 * rear)


def pretwist_velocity_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    twist_drive: float,
    cfg: TransducerConfig | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Azimuthal velocity inject that pre-twists the rear-hemi flux flywheel.

    v_θ ∝ twist_drive · rear · wall — parametric flywheel spin-up.
    """
    cfg = cfg or TransducerConfig()
    if twist_drive <= 1e-12:
        z = np.zeros_like(x)
        return z, z
    rear = rear_hemi_weight(x, y, axis=cfg.axis)
    r = np.sqrt(x**2 + y**2) + 1e-12
    wall = np.exp(-((r - 1.0) ** 2) / (2.0 * 0.2**2))
    amp = twist_drive * wall * (0.3 + 0.7 * rear)
    # Azimuthal unit vector (−y/r, x/r)
    vx = -amp * y / r
    vy = amp * x / r
    return vx, vy


def apply_precharge_to_fields(
    charge_density: NDArray[np.floating],
    e_field: NDArray[np.floating],
    boost_field: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply spatial pre-charge boost to dual-shell σ and capacitive |E|."""
    factor = 1.0 + boost_field
    return charge_density * factor, e_field * factor


def channel_direction_for_mode(
    mode: str,
    intensity: float = 1.0,
) -> ChannelState:
    """Map control mode → channel polarity.

    store    → +1 inbound (active pump + collect)
    ready    → +0.6 inbound hold / top-up
    release  → −1 reverted (forward dump)
    nucleate → small inbound priming
    coast/idle → 0
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    mode = mode.lower()
    if mode == "store":
        return ChannelState(direction=1.0, intensity=intensity, reversed=False)
    if mode == "ready":
        return ChannelState(direction=0.6, intensity=0.6 * intensity, reversed=False)
    if mode == "release":
        return ChannelState(direction=-1.0, intensity=intensity, reversed=True)
    if mode == "nucleate":
        return ChannelState(direction=0.35, intensity=0.35 * intensity, reversed=False)
    return ChannelState(direction=0.0, intensity=0.0, reversed=False)


def flux_channel_polarity_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    channels: ChannelState,
    cfg: TransducerConfig | None = None,
) -> NDArray[np.floating]:
    """Spatial field encoding flux-channel direction for Hopf / defect coupling."""
    cfg = cfg or TransducerConfig()
    rear = rear_hemi_weight(x, y, axis=cfg.axis)
    r = np.sqrt(x**2 + y**2)
    wall = np.exp(-((r - 1.0) ** 2) / (2.0 * 0.2**2))
    return (
        cfg.hopf_channel_gain
        * channels.direction
        * channels.intensity
        * wall
        * (0.4 + 0.6 * rear)
    )


def flow_intensity_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    channels: ChannelState,
    cfg: TransducerConfig | None = None,
) -> NDArray[np.floating]:
    """How hard the transducer is metering flow at each point (0–1-ish)."""
    cfg = cfg or TransducerConfig()
    rear = rear_hemi_weight(x, y, axis=cfg.axis)
    if channels.reversed:
        front_bias = 1.0 - 0.45 * rear
        return channels.intensity * front_bias
    return channels.intensity * rear


def accumulate_ledger(
    ledger: EnergyLedger,
    reading: TransducerReading,
    dt: float,
    cfg: TransducerConfig | None = None,
    open_channels: bool = True,
) -> EnergyLedger:
    """Bank passively sensed power into the ledger (generator path)."""
    cfg = cfg or TransducerConfig()
    if not open_channels:
        return ledger.clip_capacity(cfg.capacity)
    incr = reading.as_ledger_increment(dt, cfg.store_efficiency)
    return ledger.add(incr).clip_capacity(cfg.capacity)


def dump_ledger(
    ledger: EnergyLedger,
    dt: float,
    cfg: TransducerConfig | None = None,
    intensity: float | None = None,
) -> tuple[EnergyLedger, EnergyLedger, float]:
    """Revert channels and extract a metered dump.

    Returns (remaining_ledger, dump_ledger, directed_impulse).
    """
    cfg = cfg or TransducerConfig()
    intensity = cfg.intensity if intensity is None else float(
        np.clip(intensity, 0.0, cfg.max_intensity)
    )
    if ledger.total <= 1e-15 or intensity <= 1e-15:
        return ledger, EnergyLedger(), 0.0

    rate = 1.8 * intensity
    frac = min(1.0, rate * dt + 0.05 * intensity * dt)
    dump = ledger.scale(frac)
    remaining = ledger.scale(1.0 - frac).clip_capacity(cfg.capacity)

    directed = dump.total * cfg.reverse_gain * intensity * cfg.release_efficiency
    return remaining, dump, float(directed)


def coast_leak(
    ledger: EnergyLedger,
    dt: float,
    cfg: TransducerConfig | None = None,
    rate: float | None = None,
) -> EnergyLedger:
    """Slow passive leakage when channels are idle or holding ready."""
    cfg = cfg or TransducerConfig()
    leak = cfg.leak_rate if rate is None else rate
    factor = max(0.0, 1.0 - leak * dt)
    return ledger.scale(factor)


def transducer_shift_contribution(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    ledger: EnergyLedger,
    directed_impulse: float,
    channels: ChannelState,
    cfg: TransducerConfig | None = None,
    rear_bias: float = 0.85,
    pump: PumpCommand | None = None,
) -> NDArray[np.floating]:
    """Directional shift from ledger + pump tension + metered impulse.

    Store/ready: negative axial bias on rear proportional to ledger (+ extra
    pre-twist packing). Release: forward impulse, intensity-weighted.
    """
    cfg = cfg or TransducerConfig()
    axis = cfg.axis
    coord = x if axis == "x" else y
    rear = rear_hemi_weight(x, y, axis=axis)
    r = np.sqrt(x**2 + y**2)
    wall = np.exp(-((r - 1.0) ** 2) / (2.0 * 0.18**2))

    # Pre-twist / pre-charge add a bit more rear tension while pumping
    pump_extra = 0.0
    if pump is not None and pump.pump_active:
        pump_extra = 0.15 * (pump.charge_boost + pump.twist_drive)

    store_term = (
        -rear_bias
        * (ledger.total + pump_extra)
        * rear
        * wall
        * np.sign(coord + 1e-12)
        * max(channels.direction, 0.0)
    )
    intensity_map = flow_intensity_field(x, y, channels, cfg)
    release_term = (
        2.4
        * directed_impulse
        * wall
        * intensity_map
        * (1.0 if channels.reversed else 0.0)
    )
    return store_term + release_term


def modulate_defect_by_channels(
    defect: NDArray[np.floating],
    channel_field: NDArray[np.floating],
    strength: float = 0.25,
) -> NDArray[np.floating]:
    """Modulate topological defect density by flux-channel polarity."""
    c_norm = channel_field / (np.max(np.abs(channel_field)) + 1e-12)
    return defect * (1.0 + strength * c_norm)


@dataclass
class FluxTransducer:
    """Stateful craft-side flux transducer (motor + generator + gearbox).

    - **Motor (store/ready):** active pre-charge + pre-twist toward targets
    - **Generator:** passive collection from ambient fields
    - **Gearbox (release):** channel reversion + intensity throttle
    """

    cfg: TransducerConfig = field(default_factory=TransducerConfig)
    ledger: EnergyLedger = field(default_factory=EnergyLedger)
    total_pumped: EnergyLedger = field(default_factory=EnergyLedger)
    """Cumulative actively injected energy (fidelity / accounting)."""

    def reset(self) -> None:
        self.ledger = EnergyLedger()
        self.total_pumped = EnergyLedger()

    def set_intensity(self, intensity: float) -> None:
        self.cfg = replace(
            self.cfg,
            intensity=float(np.clip(intensity, 0.0, self.cfg.max_intensity)),
        )

    def set_pump_intensity(self, intensity: float) -> None:
        self.cfg = replace(
            self.cfg,
            pump_intensity=float(np.clip(intensity, 0.0, self.cfg.max_intensity)),
        )

    def set_targets(
        self,
        target_energy: float | None = None,
        target_electrostatic: float | None = None,
        target_twist: float | None = None,
    ) -> None:
        kw: dict = {}
        if target_energy is not None:
            kw["target_energy"] = float(np.clip(target_energy, 0.0, 1.0))
        if target_electrostatic is not None:
            kw["target_electrostatic"] = float(target_electrostatic)
        if target_twist is not None:
            kw["target_twist"] = float(target_twist)
        if kw:
            self.cfg = replace(self.cfg, **kw)

    def step(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        mode: str,
        e_field: NDArray[np.floating],
        charge_density: NDArray[np.floating],
        void_amplitude_field: NDArray[np.floating],
        psi: float,
        dt: float = 0.05,
        dx: float = 0.1,
        velocity: NDArray[np.floating] | None = None,
        intensity: float | None = None,
        pump_intensity: float | None = None,
    ) -> TransducerReport:
        """One transducer tick: sense → pump/collect or reverse+meter → report.

        Parameters
        ----------
        mode:
            ``nucleate`` | ``store`` | ``ready`` | ``release`` | ``coast`` | ``idle``
        intensity:
            Release throttle override.
        pump_intensity:
            Active pre-charge/pre-twist throttle override.
        """
        thr = self.cfg.intensity if intensity is None else float(
            np.clip(intensity, 0.0, self.cfg.max_intensity)
        )
        p_thr = self.cfg.pump_intensity if pump_intensity is None else float(
            np.clip(pump_intensity, 0.0, self.cfg.max_intensity)
        )
        channels = channel_direction_for_mode(mode, intensity=thr if mode.lower() == "release" else p_thr)
        reading = sense_energy_channels(
            x,
            y,
            e_field=e_field,
            charge_density=charge_density,
            velocity=velocity,
            void_amplitude_field=void_amplitude_field,
            psi=psi,
            dx=dx,
            cfg=self.cfg,
        )

        dump = EnergyLedger()
        directed = 0.0
        pumped = EnergyLedger()
        mode_l = mode.lower()
        pump = compute_pump_command(
            self.ledger, psi, self.cfg, mode=mode_l, pump_intensity=p_thr
        )

        if mode_l == "store":
            # Passive collection (generator) + active pre-charge / pre-twist (motor)
            self.ledger = accumulate_ledger(
                self.ledger, reading, dt, self.cfg, open_channels=True
            )
            self.ledger, pumped = apply_active_pump(self.ledger, pump, dt, self.cfg)
            self.total_pumped = self.total_pumped.add(pumped).clip_capacity(
                self.cfg.capacity * 4.0
            )
            # Recompute pump readiness after inject
            pump = replace(pump, ready=is_ready(self.ledger, self.cfg))
        elif mode_l == "ready":
            self.ledger = coast_leak(
                self.ledger, dt, self.cfg, rate=self.cfg.ready_hold_leak
            )
            # Top-up to hold target
            pump = compute_pump_command(
                self.ledger, psi, self.cfg, mode="ready", pump_intensity=p_thr
            )
            self.ledger, pumped = apply_active_pump(self.ledger, pump, dt, self.cfg)
            self.total_pumped = self.total_pumped.add(pumped)
            pump = replace(pump, ready=is_ready(self.ledger, self.cfg))
        elif mode_l == "nucleate":
            primed = reading.as_ledger_increment(dt, self.cfg.store_efficiency * 0.35)
            self.ledger = self.ledger.add(primed).clip_capacity(self.cfg.capacity)
            self.ledger, pumped = apply_active_pump(self.ledger, pump, dt, self.cfg)
            self.total_pumped = self.total_pumped.add(pumped)
        elif mode_l == "release":
            # Kill pump; dump via reverted channels
            pump = PumpCommand(ready=is_ready(self.ledger, self.cfg), pump_active=False)
            self.ledger, dump, directed = dump_ledger(
                self.ledger, dt, self.cfg, intensity=thr
            )
        else:
            pump = PumpCommand(ready=is_ready(self.ledger, self.cfg), pump_active=False)
            self.ledger = coast_leak(self.ledger, dt, self.cfg)

        ch_field = flux_channel_polarity_field(x, y, channels, self.cfg)
        flow_field = flow_intensity_field(x, y, channels, self.cfg)
        boost_field = precharge_boost_field(x, y, pump.charge_boost, self.cfg)
        pretwist = pretwist_velocity_field(x, y, pump.twist_drive, self.cfg)

        return TransducerReport(
            ledger=EnergyLedger(
                electrostatic=self.ledger.electrostatic,
                twist=self.ledger.twist,
                geometric=self.ledger.geometric,
            ),
            channels=channels,
            reading=reading,
            dump=dump,
            directed_impulse=directed,
            pump=pump,
            pumped=pumped,
            channel_field=ch_field,
            flow_intensity_field=flow_field,
            charge_boost_field=boost_field,
            pretwist_velocity=pretwist,
        )


def ledger_from_total(
    total: float,
    cfg: TransducerConfig | None = None,
) -> EnergyLedger:
    """Split a scalar stored_energy into weighted channels (bootstrap)."""
    cfg = cfg or TransducerConfig()
    wsum = cfg.w_electrostatic + cfg.w_twist + cfg.w_geometric
    if wsum <= 1e-15:
        wsum = 1.0
    return EnergyLedger(
        electrostatic=total * cfg.w_electrostatic / wsum,
        twist=total * cfg.w_twist / wsum,
        geometric=total * cfg.w_geometric / wsum,
    ).clip_capacity(cfg.capacity)

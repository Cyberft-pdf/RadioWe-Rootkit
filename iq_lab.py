"""
IQ capture and spectrum utilities for RTL-SDR (pyrtlsdr).

Intended for authorized lab use: your own transmitters, ISM demos,
or public downlinks where reception is legal in your jurisdiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

try:
    from rtlsdr import RtlSdr
except ImportError:  # pragma: no cover - runtime env without dongle/lib
    RtlSdr = None  # type: ignore[misc, assignment]


GainSetting = Union[float, Literal["auto"]]


@dataclass(frozen=True)
class CaptureConfig:
    center_hz: float
    sample_rate_hz: float
    gain_db: GainSetting = "auto"
    bias_tee: bool = False


def capture_samples(cfg: CaptureConfig, num_samples: int) -> np.ndarray:
    """Read complex64 IQ from the dongle."""
    if RtlSdr is None:
        raise RuntimeError(
            "pyrtlsdr / librtlsdr is not available. Install requirements and RTL-SDR drivers."
        )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    sdr = RtlSdr()
    try:
        sdr.sample_rate = cfg.sample_rate_hz
        sdr.center_freq = cfg.center_hz
        sdr.gain = cfg.gain_db if isinstance(cfg.gain_db, str) else float(cfg.gain_db)
        if cfg.bias_tee and hasattr(sdr, "set_bias_tee"):
            sdr.set_bias_tee(True)
        raw = sdr.read_samples(num_samples)
        return np.asarray(raw, dtype=np.complex64)
    finally:
        sdr.close()


class RtlSdrSession:
    """Keep the dongle open for repeated captures (e.g. live UI)."""

    def __init__(self, cfg: CaptureConfig) -> None:
        if RtlSdr is None:
            raise RuntimeError(
                "pyrtlsdr / librtlsdr is not available. Install requirements and RTL-SDR drivers."
            )
        self._cfg = cfg
        self._sdr = RtlSdr()
        self._sdr.sample_rate = cfg.sample_rate_hz
        self._sdr.center_freq = cfg.center_hz
        self._sdr.gain = cfg.gain_db if isinstance(cfg.gain_db, str) else float(cfg.gain_db)
        if cfg.bias_tee and hasattr(self._sdr, "set_bias_tee"):
            self._sdr.set_bias_tee(True)

    @property
    def sample_rate_hz(self) -> float:
        return float(self._cfg.sample_rate_hz)

    @property
    def center_hz(self) -> float:
        return float(self._sdr.center_freq)

    def set_center_hz(self, hz: float) -> None:
        self._sdr.center_freq = hz

    def read(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        raw = self._sdr.read_samples(num_samples)
        return np.asarray(raw, dtype=np.complex64)

    def close(self) -> None:
        self._sdr.close()


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def averaged_psd(
    samples: np.ndarray,
    sample_rate_hz: float,
    fft_size: int = 2048,
    averages: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch-style averaged one-sided power spectrum (linear power, not dB).

    Returns (freq_offsets_hz, psd_linear) where freq_offsets are relative to
    center (DC at 0), suitable for plotting shifted spectrum.
    """
    x = np.asarray(samples, dtype=np.complex64).ravel()
    if x.size < fft_size:
        raise ValueError("Not enough samples for requested fft_size")

    step = max(1, (x.size - fft_size) // max(averages, 1))
    acc = np.zeros(fft_size, dtype=np.float64)
    used = 0
    w = _hann(fft_size)
    win_power = float(np.sum(w**2))

    for start in range(0, x.size - fft_size + 1, step):
        if used >= averages:
            break
        seg = x[start : start + fft_size] * w
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc += (np.abs(spec) ** 2) / win_power
        used += 1

    if used == 0:
        raise RuntimeError("averages produced zero segments")

    psd = acc / used
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    return freqs.astype(np.float32), psd.astype(np.float32)


def psd_db(freq_offsets_hz: np.ndarray, psd_linear: np.ndarray) -> np.ndarray:
    """Convert linear PSD to dB; guards zeros."""
    p = np.maximum(psd_linear, 1e-20)
    return (10.0 * np.log10(p)).astype(np.float32)


def iq_metrics(samples: np.ndarray) -> dict[str, float]:
    """Simple health / anomaly-oriented metrics on a chunk of IQ."""
    x = np.asarray(samples, dtype=np.complex64).ravel()
    power = float(np.mean(np.abs(x) ** 2))
    amp = np.abs(x)
    crest = float(np.max(amp) / (np.mean(amp) + 1e-12))
    # Normalized fourth moment of magnitude (high for strong impulsive RFI)
    m2 = float(np.mean(amp**2))
    m4 = float(np.mean(amp**4))
    kappa = float(m4 / (m2**2 + 1e-20))
    return {
        "mean_power_linear": power,
        "crest_factor_amp": crest,
        "amp_kurtosis_ratio": kappa,
    }


def dominant_offset_hz(freq_offsets_hz: np.ndarray, psd_linear: np.ndarray) -> float:
    """Frequency offset (Hz) of the strongest bin."""
    idx = int(np.argmax(psd_linear))
    return float(freq_offsets_hz[idx])

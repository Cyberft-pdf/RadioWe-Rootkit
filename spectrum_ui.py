"""
Simple pygame spectrum viewer for RTL-SDR (lab / authorized use).

Controls:
  Left / Right  — tune center frequency ±100 kHz
  Up / Down     — tune ±1 MHz
  Esc / Q       — quit
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pygame

from iq_lab import (
    CaptureConfig,
    RtlSdrSession,
    averaged_psd,
    dominant_offset_hz,
    psd_db,
)


@dataclass
class UiConfig:
    center_mhz: float
    rate_msps: float
    gain: str | float
    fft_size: int
    averages: int
    bias_tee: bool
    demo: bool


def _mhz_to_hz(m: float) -> float:
    return m * 1e6


def _synthetic_iq(num_samples: int, sample_rate_hz: float, tone_offset_hz: float = 50e3) -> np.ndarray:
    """Noise + a narrow tone for UI testing without hardware."""
    rng = np.random.default_rng()
    t = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
    tone = np.exp(2j * np.pi * tone_offset_hz * t).astype(np.complex64)
    noise = (
        (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples))
        * 0.15
    ).astype(np.complex64)
    return (tone * 0.5 + noise).astype(np.complex64)


def run_spectrum_ui(cfg: UiConfig) -> int:
    pygame.init()
    pygame.display.set_caption("RTL-SDR spectrum (pygame)")
    width, height = 960, 540
    graph_top = 48
    graph_h = height - graph_top - 72
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()

    rate_hz = cfg.rate_msps * 1e6
    center_hz = _mhz_to_hz(cfg.center_mhz)
    gain_setting: str | float = cfg.gain if cfg.gain == "auto" else float(cfg.gain)

    session: RtlSdrSession | None = None
    if not cfg.demo:
        try:
            session = RtlSdrSession(
                CaptureConfig(
                    center_hz=center_hz,
                    sample_rate_hz=rate_hz,
                    gain_db=gain_setting,
                    bias_tee=cfg.bias_tee,
                )
            )
        except RuntimeError as e:
            print(e, file=sys.stderr)
            print('Start with --demo to try the UI without a dongle.', file=sys.stderr)
            pygame.quit()
            return 1

    iq_len = max(cfg.fft_size * (cfg.averages + 16), int(rate_hz * 0.05))
    iq_len = min(iq_len, int(rate_hz * 0.25))

    floor_db = -80.0
    ceil_db = -20.0
    smooth_peak = -40.0
    demo_center_hz = center_hz

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (
                    pygame.K_LEFT,
                    pygame.K_RIGHT,
                    pygame.K_UP,
                    pygame.K_DOWN,
                ):
                    step = 100e3
                    if event.key == pygame.K_LEFT:
                        delta = -step
                    elif event.key == pygame.K_RIGHT:
                        delta = step
                    elif event.key == pygame.K_UP:
                        delta = 1e6
                    else:
                        delta = -1e6
                    if session is not None:
                        session.set_center_hz(session.center_hz + delta)
                    else:
                        demo_center_hz += delta

        if session is not None:
            iq = session.read(iq_len)
            cur_center = session.center_hz
        else:
            iq = _synthetic_iq(iq_len, rate_hz)
            cur_center = demo_center_hz

        freqs, psd_lin = averaged_psd(
            iq,
            sample_rate_hz=rate_hz,
            fft_size=cfg.fft_size,
            averages=cfg.averages,
        )
        db = psd_db(freqs, psd_lin)
        peak_hz = dominant_offset_hz(freqs, psd_lin)
        peak_db = float(np.max(db))

        # Slow auto-range for readability
        target_floor = peak_db - 55.0
        target_ceil = peak_db + 8.0
        floor_db += 0.08 * (target_floor - floor_db)
        ceil_db += 0.08 * (target_ceil - ceil_db)
        smooth_peak += 0.15 * (peak_db - smooth_peak)

        screen.fill((12, 14, 22))
        # Grid
        for g in range(0, graph_h, 40):
            pygame.draw.line(
                screen,
                (35, 40, 55),
                (0, graph_top + g),
                (width, graph_top + g),
                1,
            )

        span = float(db.max() - db.min()) + 1e-6
        pts: list[tuple[int, int]] = []
        n_bins = db.shape[0]
        for x in range(width):
            bi = int(x * (n_bins - 1) / max(width - 1, 1))
            val = float(db[bi])
            norm = (val - floor_db) / max(ceil_db - floor_db, 1e-3)
            norm = max(0.0, min(1.0, norm))
            y = graph_top + graph_h - int(norm * graph_h)
            pts.append((x, y))

        if len(pts) > 1:
            pygame.draw.lines(screen, (80, 200, 255), False, pts, 2)

        center_mhz_disp = cur_center / 1e6
        lines = [
            f"Center: {center_mhz_disp:.6f} MHz   |   Peak offset: {peak_hz / 1e3:.2f} kHz",
            f"Peak ~ {smooth_peak:.1f} dB (rel)   |   {cfg.rate_msps} Msps   |   {'DEMO' if cfg.demo else 'RTL-SDR'}",
            "Left/Right: ±100 kHz   Up/Down: ±1 MHz   Esc: quit",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, (220, 224, 232))
            screen.blit(surf, (12, 8 + i * 22))

        fps_surf = font.render(f"{clock.get_fps():.0f} FPS", True, (140, 150, 170))
        screen.blit(fps_surf, (width - 88, height - 28))

        pygame.display.flip()
        clock.tick(30)

    if session is not None:
        session.close()
    pygame.quit()
    return 0

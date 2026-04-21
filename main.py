"""
CLI for RTL-SDR lab capture and spectrum (pyrtlsdr + numpy).

Examples:
  python main.py capture --center-mhz 137.9 --seconds 1 --out lab.npz
  python main.py spectrum --center-mhz 100 --rate-msps 2.048
  python main.py analyze lab.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from iq_lab import (
    CaptureConfig,
    averaged_psd,
    capture_samples,
    dominant_offset_hz,
    iq_metrics,
    psd_db,
)


def _mhz_to_hz(m: float) -> float:
    return m * 1e6


def _cmd_capture(args: argparse.Namespace) -> int:
    rate_hz = args.rate_msps * 1e6
    center_hz = _mhz_to_hz(args.center_mhz)
    n = int(rate_hz * args.seconds)
    cfg = CaptureConfig(
        center_hz=center_hz,
        sample_rate_hz=rate_hz,
        gain_db=args.gain if args.gain == "auto" else float(args.gain),
        bias_tee=args.bias_tee,
    )
    print(f"Capturing {n} samples @ {args.center_mhz} MHz, {args.rate_msps} Msps …")
    iq = capture_samples(cfg, n)
    out = Path(args.out)
    meta = {
        "center_hz": center_hz,
        "sample_rate_hz": rate_hz,
        "gain": args.gain,
        "bias_tee": args.bias_tee,
    }
    np.savez_compressed(out, iq=iq, meta=np.array([meta], dtype=object))
    m = iq_metrics(iq)
    print("IQ metrics:", m)
    print(f"Saved {out} (shape {iq.shape})")
    return 0


def _cmd_spectrum(args: argparse.Namespace) -> int:
    rate_hz = args.rate_msps * 1e6
    center_hz = _mhz_to_hz(args.center_mhz)
    n = int(rate_hz * args.seconds)
    cfg = CaptureConfig(
        center_hz=center_hz,
        sample_rate_hz=rate_hz,
        gain_db=args.gain if args.gain == "auto" else float(args.gain),
        bias_tee=args.bias_tee,
    )
    iq = capture_samples(cfg, n)
    freqs, psd_lin = averaged_psd(
        iq,
        sample_rate_hz=rate_hz,
        fft_size=args.fft_size,
        averages=args.averages,
    )
    peak_off = dominant_offset_hz(freqs, psd_lin)
    peak_db = float(np.max(psd_db(freqs, psd_lin)))
    print(f"Strongest offset from center: {peak_off / 1e3:.3f} kHz")
    print(f"Peak PSD (relative): {peak_db:.2f} dB")
    m = iq_metrics(iq)
    print("IQ metrics:", m)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.recording)
    data = np.load(path, allow_pickle=True)
    iq = np.asarray(data["iq"], dtype=np.complex64).ravel()
    meta = None
    if "meta" in data:
        meta = data["meta"].item() if data["meta"].size else None
    if meta:
        print("Recording meta:", meta)
    rate = float(meta["sample_rate_hz"]) if meta else float(args.rate_msps) * 1e6
    print("IQ metrics:", iq_metrics(iq))
    freqs, psd_lin = averaged_psd(
        iq,
        sample_rate_hz=rate,
        fft_size=args.fft_size,
        averages=args.averages,
    )
    db = psd_db(freqs, psd_lin)
    print(f"Peak offset: {dominant_offset_hz(freqs, psd_lin) / 1e3:.3f} kHz")
    print(f"Peak PSD (relative): {float(np.max(db)):.2f} dB")
    if args.export_csv:
        csv_path = Path(args.export_csv)
        stacked = np.column_stack([freqs, psd_lin, db])
        np.savetxt(
            csv_path,
            stacked,
            delimiter=",",
            header="freq_offset_hz,psd_linear,psd_db",
            comments="",
        )
        print(f"Wrote {csv_path}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from spectrum_ui import UiConfig, run_spectrum_ui

    return run_spectrum_ui(
        UiConfig(
            center_mhz=args.center_mhz,
            rate_msps=args.rate_msps,
            gain=args.gain,
            fft_size=args.fft_size,
            averages=args.averages,
            bias_tee=args.bias_tee,
            demo=args.demo,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RTL-SDR lab capture / spectrum")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--center-mhz", type=float, required=True)
        sp.add_argument("--rate-msps", type=float, default=2.048, help="Sample rate in Msps")
        sp.add_argument(
            "--gain",
            default="auto",
            help='Gain in dB or "auto"',
        )
        sp.add_argument("--bias-tee", action="store_true", help="Enable bias tee if supported")
        sp.add_argument("--seconds", type=float, default=0.25, help="Capture length")

    sc = sub.add_parser("capture", help="Save IQ to compressed .npz")
    add_common(sc)
    sc.add_argument("--out", type=str, default="capture.npz")
    sc.set_defaults(func=_cmd_capture)

    ss = sub.add_parser("spectrum", help="Print quick spectrum stats (live capture)")
    add_common(ss)
    ss.add_argument("--fft-size", type=int, default=2048)
    ss.add_argument("--averages", type=int, default=16)
    ss.set_defaults(func=_cmd_spectrum)

    sa = sub.add_parser("analyze", help="Analyze a .npz from capture")
    sa.add_argument("recording", type=str)
    sa.add_argument("--rate-msps", type=float, default=2.048, help="Fallback if .npz has no meta")
    sa.add_argument("--fft-size", type=int, default=2048)
    sa.add_argument("--averages", type=int, default=32)
    sa.add_argument("--export-csv", type=str, default="", help="Optional path for freq,psd CSV")
    sa.set_defaults(func=_cmd_analyze)

    su = sub.add_parser("ui", help="Live spectrum window (pygame)")
    add_common(su)
    su.add_argument("--fft-size", type=int, default=1024)
    su.add_argument("--averages", type=int, default=4)
    su.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic IQ (no dongle); spectrum is fake, arrows still move displayed center",
    )
    su.set_defaults(func=_cmd_ui)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

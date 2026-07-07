"""
TCSPC Reconvolution Fitting Tool
================================

Programm zur Analyse zeitaufgeloester Fluoreszenzabklingkurven (TCSPC) mittels
Reconvolution-Fit (Faltung eines multiexponentiellen Modells mit einer
gemessenen Instrument Response Function, IRF).

Modell:
    I(t) = IRF (gefaltet mit) [ Summe_i A_i * exp(-t/tau_i) ] + Offset

Enthaelt zwei Fit-Modi:
    1. Einzel-Fit  : ein Datensatz (eine Wellenlaenge)
    2. Globaler Fit: mehrere Wellenlaengen mit gemeinsamen tau_i
                     (target analysis / global analysis)

Benoetigte Pakete: numpy, scipy, pandas, matplotlib, tkinter (Standardbibliothek)

Start:
    python tcspc_reconvolution_fit.py

Changelog
---------
v2 (2026-07-07):
  - Amplituden-Skalierung korrigiert: Die Faltung hat das Ergebnis bisher
    zusaetzlich mit dt multipliziert, obwohl die IRF bereits auf Summe = 1
    normiert war. Dadurch trugen alle gefitteten Amplituden A_i einen
    versteckten Faktor 1/dt (bei 2 ps Zeitschritt: Faktor 500) und lagen
    in unueblich hohen Zahlenbereichen (mehrere hundert statt ~1-10).
    Die Multiplikation mit dt wurde entfernt; Zeitkonstanten, Chi-Quadrat
    und die Form der Fit-Kurve sind davon unveraendert, nur A_i liegt jetzt
    auf der gleichen Groessenordnung wie die Rohdaten.
  - Zusaetzliche Kennzahl tau_mean_intensity_ns ergaenzt (intensitaets-
    gewichtete mittlere Lebensdauer, tau_mean_intensity = Summe(A_i*tau_i^2)
    / Summe(A_i*tau_i)). Die bisherige amplitudengewichtete tau_mean reagiert
    sehr empfindlich auf numerisch "entartete" Komponenten mit extrem
    kurzem tau und entsprechend riesiger Amplitude (die kaum echte
    Intensitaet beitragen, aber die Mittelung dominieren). tau_mean_intensity
    ist gegenueber solchen Artefakten deutlich robuster und wird zusaetzlich
    zur bisherigen tau_mean ausgegeben.
v1 (2026-07-06): Erste Version (Einzel-/globaler Reconvolution-Fit, GUI).
"""

from __future__ import annotations

import os
import re
import sys
import time
import queue
import traceback
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Fitting laeuft in einem Hintergrund-Thread -> kein GUI-Backend noetig,
# Plots werden ausschliesslich als PNG-Dateien gespeichert.
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.signal import fftconvolve

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext


# =============================================================================
# Konstanten / Defaults
# =============================================================================

DEFAULT_N_COMPONENTS = 4
DEFAULT_TAU_LOWER_NS = 0.0
DEFAULT_TAU_UPPER_NS = 5.0

# tau darf physikalisch nicht exakt 0 sein (Division durch 0 im Modell).
TAU_MIN_EPS_NS = 1.0e-4  # 0.1 ps

# Maximale erlaubte IRF-Verschiebung (in beide Richtungen), in ns.
MAX_IRF_SHIFT_NS = 1.0

# Anzahl Multi-Start-Versuche fuer robustere Konvergenz.
N_MULTISTART = 6
RANDOM_SEED = 42

# Presets fuer initiale tau-Schaetzwerte (ns). Der Nutzer hat drei Presets
# vorgegeben; ein viertes wurde ergaenzt (breiterer Bereich), da vier Buttons
# gewuenscht waren, aber nur drei Beispiel-Listen mitgeliefert wurden.
PRESETS: Dict[int, List[float]] = {
    1: [0.05, 0.2, 0.8, 2.5],
    2: [0.03, 0.15, 0.6, 1.8],
    3: [0.1, 0.4, 1.2, 3.5],
    4: [0.02, 0.1, 0.5, 2.0],
}


# =============================================================================
# Datenklassen
# =============================================================================

@dataclass
class SingleFitResult:
    success: bool
    message: str
    n_components: int
    amplitudes: np.ndarray
    taus_ns: np.ndarray
    offset: float
    irf_shift_ns: float
    tau_mean_ns: float
    tau_mean_abs_ns: Optional[float]
    tau_mean_intensity_ns: float
    chi2: float
    red_chi2: float
    dof: int
    t: np.ndarray
    y_raw: np.ndarray
    fit_curve: np.ndarray
    residuals: np.ndarray
    irf_display_t: np.ndarray
    irf_display_y: np.ndarray


@dataclass
class GlobalFitResult:
    success: bool
    message: str
    n_components: int
    taus_ns: np.ndarray
    wavelengths: List[float]
    labels: List[str]
    amplitudes_per_ds: List[np.ndarray]
    offsets: List[float]
    irf_shifts_ns: List[float]
    chi2_per_ds: List[float]
    red_chi2_per_ds: List[float]
    tau_mean_per_ds: List[float]
    tau_mean_abs_per_ds: List[Optional[float]]
    tau_mean_intensity_per_ds: List[float]
    total_chi2: float
    total_red_chi2: float
    t_list: List[np.ndarray]
    y_raw_list: List[np.ndarray]
    fit_curves: List[np.ndarray]
    residuals_list: List[np.ndarray]
    irf_display_list: List[Tuple[np.ndarray, np.ndarray]]


# =============================================================================
# Datei-Import (robust)
# =============================================================================

def load_two_column_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Laedt eine TXT/CSV-Datei mit mindestens einer Zeit- und einer
    Intensitaetsspalte. Erkennt Trennzeichen und optionale Kopfzeile
    automatisch. Gibt (t, y) als float-Arrays zurueck, sortiert nach t,
    mit relativer Zeitachse (t[0] = 0).
    """
    if not os.path.isfile(path):
        raise ValueError(f"Datei nicht gefunden: {path}")

    df = None
    last_err: Optional[Exception] = None
    for sep in [None, "\t", ",", ";", r"\s+"]:
        try:
            candidate = pd.read_csv(path, sep=sep, engine="python", header=None, comment="#")
            if candidate.shape[1] < 2:
                continue
            first_row_numeric = pd.to_numeric(candidate.iloc[0], errors="coerce").notna().all()
            if not first_row_numeric:
                # Erste Zeile ist vermutlich eine Kopfzeile -> mit Header neu einlesen.
                candidate = pd.read_csv(path, sep=sep, engine="python", header=0, comment="#")
            df = candidate
            break
        except Exception as exc:  # Naechstes Trennzeichen probieren.
            last_err = exc
            df = None
            continue

    if df is None or df.shape[1] < 2:
        raise ValueError(
            f"Datei konnte nicht gelesen werden oder enthaelt weniger als 2 Spalten: "
            f"{os.path.basename(path)} ({last_err})"
        )

    col_names = [str(c).strip().lower() for c in df.columns]
    time_keywords = ["time", "zeit", "t_ns", "t (ns)", "ns", "t"]
    y_keywords = ["counts", "intensity", "intensitaet", "intensität", "cps", "signal", "y"]

    time_col = None
    y_col = None
    for i, name in enumerate(col_names):
        if time_col is None and any(name == k or name.startswith(k) for k in time_keywords):
            time_col = i
        if y_col is None and any(name == k or name.startswith(k) for k in y_keywords):
            y_col = i

    if time_col is None or y_col is None or time_col == y_col:
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = [i for i in range(numeric_df.shape[1]) if numeric_df.iloc[:, i].notna().sum() > 0]
        if len(numeric_cols) < 2:
            raise ValueError(f"Es konnten keine zwei numerischen Spalten gefunden werden in: {os.path.basename(path)}")
        time_col, y_col = numeric_cols[0], numeric_cols[1]

    t = pd.to_numeric(df.iloc[:, time_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, y_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 10:
        raise ValueError(f"Zu wenige gueltige (nicht-NaN) Datenpunkte in: {os.path.basename(path)}")
    t, y = t[mask], y[mask]

    order = np.argsort(t, kind="mergesort")
    t, y = t[order], y[order]

    # Doppelte Zeitpunkte entfernen (sonst Probleme bei der Interpolation).
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, y = t[keep], y[keep]

    # Relative Zeitachse: Falls die Zeitspalte nicht bei 0 beginnt.
    t = t - t[0]

    return t, y


def parse_wavelength_from_filename(path: str) -> Optional[float]:
    """Versucht, eine Wellenlaenge (in nm) aus dem Dateinamen zu extrahieren."""
    name = os.path.basename(path)
    m = re.search(r"(\d{3,4}(?:\.\d+)?)\s*nm", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d{3,4}(?:\.\d+)?)", name)
    if m2:
        return float(m2.group(1))
    return None


# =============================================================================
# IRF-Hilfsfunktionen (Peak-Erkennung, Resampling, Verschiebung)
# =============================================================================

def estimate_peak_time(t: np.ndarray, y: np.ndarray) -> float:
    """Peak-Zeitpunkt mit Sub-Bin-Genauigkeit ueber parabolische Interpolation."""
    i = int(np.argmax(y))
    if i <= 0 or i >= len(y) - 1:
        return float(t[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(t[i])
    delta = 0.5 * (y0 - y2) / denom
    dt_local = t[i + 1] - t[i]
    return float(t[i] + delta * dt_local)


def resample_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Lineare Interpolation von (t_src, y_src) auf t_target; ausserhalb -> 0."""
    return np.interp(t_target, t_src, y_src, left=0.0, right=0.0)


def shift_signal(t: np.ndarray, y: np.ndarray, shift_ns: float) -> np.ndarray:
    """
    Verschiebt ein auf Gitter `t` definiertes Signal um `shift_ns` mittels
    Interpolation (kein Integer-Bin-Shift). Positive shift_ns verzoegert
    das Signal (verschiebt es zu spaeteren Zeiten).
    """
    if shift_ns == 0.0:
        return y
    return np.interp(t - shift_ns, t, y, left=0.0, right=0.0)


def normalize_irf_area(irf_y: np.ndarray) -> np.ndarray:
    """Normalisiert die IRF so, dass ihre Flaeche (Summe) = 1 ist."""
    area = np.sum(irf_y)
    if area <= 0:
        raise ValueError("Summe der IRF-Intensitaeten ist <= 0 - ungueltige IRF-Datei.")
    return irf_y / area


# =============================================================================
# Modell & Faltung
# =============================================================================

def multiexp_model(t: np.ndarray, amps: np.ndarray, taus_ns: np.ndarray) -> np.ndarray:
    """I(t) = Summe_i A_i * exp(-t/tau_i), fuer t >= 0 (kausal)."""
    taus_safe = np.maximum(taus_ns, TAU_MIN_EPS_NS)
    return np.sum(amps[:, None] * np.exp(-t[None, :] / taus_safe[:, None]), axis=0)


def convolve_with_irf(model_y: np.ndarray, irf_y: np.ndarray) -> np.ndarray:
    """
    Faltet das Modell kausal mit der IRF (beide auf demselben Zeitgitter).
    fftconvolve(..., mode='full')[:N] entspricht der diskreten kausalen
    Faltungssumme. Die IRF ist auf Summe = 1 normiert, wirkt also wie ein
    gewichteter gleitender Durchschnitt: die gefaltete Kurve bleibt dadurch
    auf derselben Amplituden-Groessenordnung wie das unkonvolvierte Modell
    (keine zusaetzliche Multiplikation mit dt - siehe Changelog v2).
    """
    n = len(model_y)
    return fftconvolve(model_y, irf_y, mode="full")[:n]


def forward_model(
    t: np.ndarray,
    amps: np.ndarray,
    taus_ns: np.ndarray,
    offset: float,
    shift_ns: float,
    irf_t: np.ndarray,
    irf_y_area_norm: np.ndarray,
) -> np.ndarray:
    """Komplettes Vorwaertsmodell: verschobene IRF (*) multiexp. Zerfall + Offset."""
    shifted_irf = shift_signal(irf_t, irf_y_area_norm, shift_ns)
    decay = multiexp_model(t, amps, taus_ns)
    conv = convolve_with_irf(decay, shifted_irf)
    return conv + offset


def clip_strict(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Clip strikt innerhalb (lb, ub), damit least_squares nicht exakt auf der Grenze startet."""
    span = np.maximum(ub - lb, 1e-12)
    local_eps = np.minimum(eps, span * 1e-6)
    return np.clip(x, lb + local_eps, ub - local_eps)


# =============================================================================
# Vorbereitung eines Datensatzes fuer den Fit (Normalisierung, Gewichte, IRF)
# =============================================================================

@dataclass
class FitContext:
    label: str
    wavelength: Optional[float]
    t: np.ndarray
    y_raw: np.ndarray
    dt: float
    scale: float
    y_hat: np.ndarray
    weight_hat: np.ndarray
    irf_t: np.ndarray
    irf_y_norm: np.ndarray
    coarse_shift_ns: float


def prepare_fit_context(
    label: str,
    wavelength: Optional[float],
    t: np.ndarray,
    y_raw: np.ndarray,
    irf_t_raw: np.ndarray,
    irf_y_raw: np.ndarray,
    use_poisson_weights: bool,
) -> FitContext:
    dt_values = np.diff(t)
    dt = float(np.median(dt_values))
    if dt <= 0:
        raise ValueError(f"Ungueltiges Zeitgitter (dt <= 0) fuer Datensatz '{label}'.")

    # IRF auf das Zeitgitter der Messdaten interpolieren und flaechennormieren.
    irf_y_grid = resample_to_grid(irf_t_raw, irf_y_raw, t)
    irf_y_norm = normalize_irf_area(irf_y_grid)

    # Grobe Peak-Ausrichtung: Differenz der (sub-bin) Peak-Zeitpunkte.
    peak_data = estimate_peak_time(t, y_raw)
    peak_irf = estimate_peak_time(t, irf_y_norm)
    coarse_shift = float(np.clip(peak_data - peak_irf, -MAX_IRF_SHIFT_NS, MAX_IRF_SHIFT_NS))

    # Normierung der Daten fuer numerisch stabilen Fit (Amplituden/Offset ~ O(1)).
    scale = float(np.max(y_raw))
    if scale <= 0:
        scale = 1.0
    y_hat = y_raw / scale

    # Poisson-Gewichtung 1/sqrt(counts+1), auf hat-Raum umgerechnet (siehe Kommentar unten).
    y_raw_clipped = np.clip(y_raw, 0.0, None)
    if use_poisson_weights:
        weight_raw = 1.0 / np.sqrt(y_raw_clipped + 1.0)
    else:
        weight_raw = np.ones_like(y_raw)
    # weight_hat = weight_raw * scale, damit gilt:
    #   (y_hat - model_hat) * weight_hat == (y_raw - model_raw) * weight_raw
    # -> chi2 aus den hat-Residuen entspricht exakt dem gewichteten chi2 in Rohdaten-Einheiten.
    weight_hat = weight_raw * scale

    return FitContext(
        label=label,
        wavelength=wavelength,
        t=t,
        y_raw=y_raw,
        dt=dt,
        scale=scale,
        y_hat=y_hat,
        weight_hat=weight_hat,
        irf_t=t,
        irf_y_norm=irf_y_norm,
        coarse_shift_ns=coarse_shift,
    )


def initial_amp_offset_guess(y_hat: np.ndarray, n_components: int) -> Tuple[np.ndarray, float]:
    """Einfache, robuste Startwerte fuer Amplituden und Offset im hat-Raum."""
    tail_n = max(5, int(0.05 * len(y_hat)))
    offset0 = float(np.median(y_hat[-tail_n:]))
    peak0 = float(np.max(y_hat))
    amp_total = max(peak0 - offset0, 1e-3)
    amps0 = np.full(n_components, amp_total / n_components)
    return amps0, offset0


# =============================================================================
# Einzel-Fit
# =============================================================================

def _unpack_single(params: np.ndarray, n_comp: int, fit_shift: bool, fixed_shift: float):
    amps = params[0:n_comp]
    taus = params[n_comp:2 * n_comp]
    offset = params[2 * n_comp]
    shift = params[2 * n_comp + 1] if fit_shift else fixed_shift
    return amps, taus, offset, shift


def _single_residuals(params, ctx: FitContext, n_comp: int, fit_shift: bool, fixed_shift: float):
    amps, taus, offset, shift = _unpack_single(params, n_comp, fit_shift, fixed_shift)
    model_hat = forward_model(ctx.t, amps, taus, offset, shift, ctx.irf_t, ctx.irf_y_norm)
    return (ctx.y_hat - model_hat) * ctx.weight_hat


def fit_single_dataset(
    label: str,
    wavelength: Optional[float],
    t: np.ndarray,
    y_raw: np.ndarray,
    irf_t_raw: np.ndarray,
    irf_y_raw: np.ndarray,
    n_components: int,
    tau_lower: float,
    tau_upper: float,
    tau_guess: List[float],
    allow_negative_amp: bool,
    use_poisson_weights: bool,
    fit_irf_shift: bool,
    n_multistart: int = N_MULTISTART,
) -> SingleFitResult:
    if tau_upper <= tau_lower:
        raise ValueError("Obere tau-Grenze muss groesser als die untere Grenze sein.")
    tau_lower_eff = max(tau_lower, TAU_MIN_EPS_NS)

    ctx = prepare_fit_context(label, wavelength, t, y_raw, irf_t_raw, irf_y_raw, use_poisson_weights)

    amps0_base, offset0 = initial_amp_offset_guess(ctx.y_hat, n_components)
    taus0_base = np.clip(np.array(tau_guess, dtype=float), tau_lower_eff, tau_upper)
    shift0 = ctx.coarse_shift_ns if fit_irf_shift else ctx.coarse_shift_ns

    # Grenzen zusammenbauen: [Amplituden (n), taus (n), offset (1), shift (0 oder 1)]
    amp_lb = -np.inf if allow_negative_amp else 0.0
    amp_ub = np.inf
    lb = [amp_lb] * n_components + [tau_lower_eff] * n_components + [-np.inf]
    ub = [amp_ub] * n_components + [tau_upper] * n_components + [np.inf]
    if fit_irf_shift:
        lb.append(-MAX_IRF_SHIFT_NS)
        ub.append(MAX_IRF_SHIFT_NS)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    rng = np.random.default_rng(RANDOM_SEED)
    best = None
    best_cost = np.inf
    messages = []

    for attempt in range(n_multistart):
        if attempt == 0:
            taus_try = taus0_base.copy()
        else:
            factors = rng.uniform(0.6, 1.6, size=n_components)
            taus_try = np.clip(taus0_base * factors, tau_lower_eff, tau_upper)

        amps_try = amps0_base.copy()
        if allow_negative_amp and attempt % 2 == 1 and n_components > 0:
            fastest_idx = int(np.argmin(taus_try))
            amps_try[fastest_idx] *= -1.0

        x0 = np.concatenate([amps_try, taus_try, [offset0]])
        if fit_irf_shift:
            x0 = np.concatenate([x0, [shift0]])
        x0 = clip_strict(x0, lb, ub)

        try:
            result = least_squares(
                _single_residuals,
                x0,
                bounds=(lb, ub),
                args=(ctx, n_components, fit_irf_shift, shift0),
                method="trf",
                x_scale="jac",
                max_nfev=1000,
            )
        except Exception as exc:
            messages.append(f"Versuch {attempt + 1} fehlgeschlagen: {exc}")
            continue

        cost = float(np.sum(result.fun ** 2))
        if cost < best_cost:
            best_cost = cost
            best = result

    if best is None:
        return SingleFitResult(
            success=False,
            message="Alle Fit-Versuche sind fehlgeschlagen:\n" + "\n".join(messages),
            n_components=n_components,
            amplitudes=np.full(n_components, np.nan),
            taus_ns=np.full(n_components, np.nan),
            offset=np.nan,
            irf_shift_ns=np.nan,
            tau_mean_ns=np.nan,
            tau_mean_abs_ns=None,
            tau_mean_intensity_ns=np.nan,
            chi2=np.nan,
            red_chi2=np.nan,
            dof=0,
            t=t,
            y_raw=y_raw,
            fit_curve=np.full_like(y_raw, np.nan),
            residuals=np.full_like(y_raw, np.nan),
            irf_display_t=t,
            irf_display_y=np.full_like(y_raw, np.nan),
        )

    amps_hat, taus, offset_hat, shift = _unpack_single(best.x, n_components, fit_irf_shift, shift0)

    # Nach aufsteigendem tau sortieren.
    order = np.argsort(taus)
    taus = taus[order]
    amps_hat = amps_hat[order]

    amps = amps_hat * ctx.scale
    offset = offset_hat * ctx.scale

    fit_curve = forward_model(t, amps, taus, offset, shift, ctx.irf_t, ctx.irf_y_norm)
    residuals = y_raw - fit_curve

    n_free = 2 * n_components + 1 + (1 if fit_irf_shift else 0)
    dof = max(len(t) - n_free, 1)
    chi2 = best_cost
    red_chi2 = chi2 / dof

    sum_amp = np.sum(amps)
    tau_mean = float(np.sum(amps * taus) / sum_amp) if sum_amp != 0 else float("nan")
    has_negative = bool(np.any(amps < 0))
    tau_mean_abs = None
    if has_negative:
        sum_abs = np.sum(np.abs(amps))
        tau_mean_abs = float(np.sum(np.abs(amps) * taus) / sum_abs) if sum_abs != 0 else float("nan")

    # Intensitaets-/zweitmomentgewichtete mittlere Lebensdauer: robuster als tau_mean
    # gegenueber numerisch entarteten Komponenten mit sehr kleinem tau und
    # entsprechend riesiger Amplitude (die kaum zu A_i*tau_i beitragen, aber
    # die amplitudengewichtete Mittelung dominieren wuerden).
    sum_a_tau = np.sum(amps * taus)
    tau_mean_intensity = float(np.sum(amps * taus ** 2) / sum_a_tau) if sum_a_tau != 0 else float("nan")

    # Verschobene IRF fuer Anzeige/CSV: auf Peak der Rohdaten skaliert (nur zur Visualisierung).
    shifted_irf_norm = shift_signal(ctx.irf_t, ctx.irf_y_norm, shift)
    irf_display_y = shifted_irf_norm
    if np.max(shifted_irf_norm) > 0:
        irf_display_y = shifted_irf_norm / np.max(shifted_irf_norm) * np.max(y_raw)

    success = bool(best.success) if hasattr(best, "success") else True
    message = "Fit erfolgreich." if success else f"least_squares meldet Konvergenzproblem: {best.message}"

    return SingleFitResult(
        success=success,
        message=message,
        n_components=n_components,
        amplitudes=amps,
        taus_ns=taus,
        offset=offset,
        irf_shift_ns=shift,
        tau_mean_ns=tau_mean,
        tau_mean_abs_ns=tau_mean_abs,
        tau_mean_intensity_ns=tau_mean_intensity,
        chi2=chi2,
        red_chi2=red_chi2,
        dof=dof,
        t=t,
        y_raw=y_raw,
        fit_curve=fit_curve,
        residuals=residuals,
        irf_display_t=t,
        irf_display_y=irf_display_y,
    )


# =============================================================================
# Globaler Fit (mehrere Wellenlaengen, gemeinsame taus)
# =============================================================================

def _unpack_global(params: np.ndarray, n_comp: int, K: int, fit_shift: bool, fixed_shifts: List[float]):
    taus = params[0:n_comp]
    idx = n_comp
    amps_list = []
    offsets = []
    for _ in range(K):
        amps_list.append(params[idx:idx + n_comp])
        idx += n_comp
        offsets.append(params[idx])
        idx += 1
    if fit_shift:
        shifts = list(params[idx:idx + K])
    else:
        shifts = fixed_shifts
    return taus, amps_list, offsets, shifts


def _global_residuals(params, contexts: List[FitContext], n_comp: int, K: int, fit_shift: bool, fixed_shifts: List[float]):
    taus, amps_list, offsets, shifts = _unpack_global(params, n_comp, K, fit_shift, fixed_shifts)
    all_res = []
    for k in range(K):
        ctx = contexts[k]
        model_hat = forward_model(ctx.t, amps_list[k], taus, offsets[k], shifts[k], ctx.irf_t, ctx.irf_y_norm)
        all_res.append((ctx.y_hat - model_hat) * ctx.weight_hat)
    return np.concatenate(all_res)


def fit_global_datasets(
    datasets: List[Tuple[str, Optional[float], np.ndarray, np.ndarray]],  # (label, wavelength, t, y_raw)
    irf_t_raw: np.ndarray,
    irf_y_raw: np.ndarray,
    n_components: int,
    tau_lower: float,
    tau_upper: float,
    tau_guess: List[float],
    allow_negative_amp: bool,
    use_poisson_weights: bool,
    fit_irf_shift: bool,
    n_multistart: int = N_MULTISTART,
) -> GlobalFitResult:
    if tau_upper <= tau_lower:
        raise ValueError("Obere tau-Grenze muss groesser als die untere Grenze sein.")
    tau_lower_eff = max(tau_lower, TAU_MIN_EPS_NS)

    K = len(datasets)
    if K == 0:
        raise ValueError("Keine Datensaetze fuer den globalen Fit ausgewaehlt.")

    contexts: List[FitContext] = []
    for label, wavelength, t, y_raw in datasets:
        ctx = prepare_fit_context(label, wavelength, t, y_raw, irf_t_raw, irf_y_raw, use_poisson_weights)
        contexts.append(ctx)

    taus0_base = np.clip(np.array(tau_guess, dtype=float), tau_lower_eff, tau_upper)
    fixed_shifts0 = [ctx.coarse_shift_ns for ctx in contexts]

    amp_lb = -np.inf if allow_negative_amp else 0.0
    amp_ub = np.inf

    lb = [tau_lower_eff] * n_components
    ub = [tau_upper] * n_components
    for _ in range(K):
        lb += [amp_lb] * n_components + [-np.inf]
        ub += [amp_ub] * n_components + [np.inf]
    if fit_irf_shift:
        lb += [-MAX_IRF_SHIFT_NS] * K
        ub += [MAX_IRF_SHIFT_NS] * K
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    rng = np.random.default_rng(RANDOM_SEED)
    best = None
    best_cost = np.inf
    messages = []

    amp_offset_guesses = [initial_amp_offset_guess(ctx.y_hat, n_components) for ctx in contexts]

    for attempt in range(n_multistart):
        if attempt == 0:
            taus_try = taus0_base.copy()
        else:
            factors = rng.uniform(0.6, 1.6, size=n_components)
            taus_try = np.clip(taus0_base * factors, tau_lower_eff, tau_upper)

        x0_parts = [taus_try]
        for k in range(K):
            amps0_k, offset0_k = amp_offset_guesses[k]
            amps_try = amps0_k.copy()
            if allow_negative_amp and attempt % 2 == 1 and n_components > 0:
                fastest_idx = int(np.argmin(taus_try))
                amps_try[fastest_idx] *= -1.0
            x0_parts.append(amps_try)
            x0_parts.append(np.array([offset0_k]))
        if fit_irf_shift:
            x0_parts.append(np.array(fixed_shifts0))
        x0 = np.concatenate(x0_parts)
        x0 = clip_strict(x0, lb, ub)

        try:
            result = least_squares(
                _global_residuals,
                x0,
                bounds=(lb, ub),
                args=(contexts, n_components, K, fit_irf_shift, fixed_shifts0),
                method="trf",
                x_scale="jac",
                max_nfev=1500,
            )
        except Exception as exc:
            messages.append(f"Versuch {attempt + 1} fehlgeschlagen: {exc}")
            continue

        cost = float(np.sum(result.fun ** 2))
        if cost < best_cost:
            best_cost = cost
            best = result

    if best is None:
        raise ValueError("Alle Fit-Versuche des globalen Fits sind fehlgeschlagen:\n" + "\n".join(messages))

    taus, amps_list_hat, offsets_hat, shifts = _unpack_global(best.x, n_components, K, fit_irf_shift, fixed_shifts0)

    order = np.argsort(taus)
    taus = taus[order]
    amps_list_hat = [a[order] for a in amps_list_hat]

    amps_list = [amps_list_hat[k] * contexts[k].scale for k in range(K)]
    offsets = [offsets_hat[k] * contexts[k].scale for k in range(K)]

    fit_curves = []
    residuals_list = []
    chi2_per_ds = []
    red_chi2_per_ds = []
    tau_mean_per_ds = []
    tau_mean_abs_per_ds: List[Optional[float]] = []
    tau_mean_intensity_per_ds = []
    irf_display_list = []

    n_local_free = n_components + 1 + (1 if fit_irf_shift else 0)

    for k in range(K):
        ctx = contexts[k]
        fit_curve = forward_model(ctx.t, amps_list[k], taus, offsets[k], shifts[k], ctx.irf_t, ctx.irf_y_norm)
        residual = ctx.y_raw - fit_curve
        fit_curves.append(fit_curve)
        residuals_list.append(residual)

        model_hat_k = forward_model(ctx.t, amps_list_hat[k], taus, offsets_hat[k], shifts[k], ctx.irf_t, ctx.irf_y_norm)
        res_hat_k = (ctx.y_hat - model_hat_k) * ctx.weight_hat
        chi2_k = float(np.sum(res_hat_k ** 2))
        dof_k = max(len(ctx.t) - n_local_free, 1)
        chi2_per_ds.append(chi2_k)
        red_chi2_per_ds.append(chi2_k / dof_k)

        sum_amp = np.sum(amps_list[k])
        tau_mean_k = float(np.sum(amps_list[k] * taus) / sum_amp) if sum_amp != 0 else float("nan")
        tau_mean_per_ds.append(tau_mean_k)
        if np.any(amps_list[k] < 0):
            sum_abs = np.sum(np.abs(amps_list[k]))
            tau_mean_abs_per_ds.append(float(np.sum(np.abs(amps_list[k]) * taus) / sum_abs) if sum_abs != 0 else float("nan"))
        else:
            tau_mean_abs_per_ds.append(None)

        sum_a_tau = np.sum(amps_list[k] * taus)
        tau_mean_intensity_k = float(np.sum(amps_list[k] * taus ** 2) / sum_a_tau) if sum_a_tau != 0 else float("nan")
        tau_mean_intensity_per_ds.append(tau_mean_intensity_k)

        shifted_irf_norm = shift_signal(ctx.irf_t, ctx.irf_y_norm, shifts[k])
        irf_display_y = shifted_irf_norm
        if np.max(shifted_irf_norm) > 0:
            irf_display_y = shifted_irf_norm / np.max(shifted_irf_norm) * np.max(ctx.y_raw)
        irf_display_list.append((ctx.t, irf_display_y))

    total_free = n_components + K * (n_components + 1) + (K if fit_irf_shift else 0)
    total_points = sum(len(ctx.t) for ctx in contexts)
    total_dof = max(total_points - total_free, 1)
    total_chi2 = best_cost
    total_red_chi2 = total_chi2 / total_dof

    success = bool(best.success) if hasattr(best, "success") else True
    message = "Globaler Fit erfolgreich." if success else f"least_squares meldet Konvergenzproblem: {best.message}"

    labels = [d[0] for d in datasets]
    wavelengths = [d[1] if d[1] is not None else float("nan") for d in datasets]

    return GlobalFitResult(
        success=success,
        message=message,
        n_components=n_components,
        taus_ns=taus,
        wavelengths=wavelengths,
        labels=labels,
        amplitudes_per_ds=amps_list,
        offsets=offsets,
        irf_shifts_ns=list(shifts),
        chi2_per_ds=chi2_per_ds,
        red_chi2_per_ds=red_chi2_per_ds,
        tau_mean_per_ds=tau_mean_per_ds,
        tau_mean_abs_per_ds=tau_mean_abs_per_ds,
        tau_mean_intensity_per_ds=tau_mean_intensity_per_ds,
        total_chi2=total_chi2,
        total_red_chi2=total_red_chi2,
        t_list=[ctx.t for ctx in contexts],
        y_raw_list=[ctx.y_raw for ctx in contexts],
        fit_curves=fit_curves,
        residuals_list=residuals_list,
        irf_display_list=irf_display_list,
    )


# =============================================================================
# Output: CSV / PNG / Parameter-Dateien
# =============================================================================

def _safe_name(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")


def save_single_fit_outputs(output_dir: str, result: SingleFitResult, label: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"SingleFit_{_safe_name(label)}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    pd.DataFrame({"time_ns": result.t, "intensity_raw": result.y_raw}).to_csv(
        os.path.join(run_dir, "raw_data.csv"), index=False
    )
    pd.DataFrame({"time_ns": result.irf_display_t, "irf_shifted": result.irf_display_y}).to_csv(
        os.path.join(run_dir, "irf_used.csv"), index=False
    )
    pd.DataFrame({"time_ns": result.t, "fit": result.fit_curve}).to_csv(
        os.path.join(run_dir, "fit_curve.csv"), index=False
    )
    pd.DataFrame({"time_ns": result.t, "residual": result.residuals}).to_csv(
        os.path.join(run_dir, "residuals.csv"), index=False
    )

    # Fit-Plot (log-Y, wie in der TCSPC-Praxis ueblich).
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(result.t, np.clip(result.y_raw, 1e-12, None), ".", color="0.4", markersize=2, label="Rohdaten")
    ax.semilogy(result.t, np.clip(result.fit_curve, 1e-12, None), "-", color="crimson", linewidth=1.5, label="Fit")
    ax.semilogy(result.t, np.clip(result.irf_display_y, 1e-12, None), "--", color="steelblue", linewidth=1.0, label="IRF (verschoben)")
    ax.set_xlabel("Zeit (ns)")
    ax.set_ylabel("Intensitaet (log)")
    ax.set_title(f"Einzel-Fit: {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fit_plot.png"), dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(result.t, result.residuals, ".", color="0.3", markersize=2)
    ax2.axhline(0.0, color="crimson", linewidth=1.0)
    ax2.set_xlabel("Zeit (ns)")
    ax2.set_ylabel("Residuum")
    ax2.set_title(f"Residuen: {label}")
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "residuals_plot.png"), dpi=150)
    plt.close(fig2)

    lines = []
    lines.append(f"Einzel-Fit Ergebnisse: {label}")
    lines.append(f"Status: {'OK' if result.success else 'FEHLER/Warnung'} - {result.message}")
    lines.append(f"Anzahl Komponenten: {result.n_components}")
    lines.append("")
    lines.append("Komponente\tA_i\ttau_i (ns)")
    for i in range(result.n_components):
        lines.append(f"{i + 1}\t{result.amplitudes[i]:.6g}\t{result.taus_ns[i]:.6g}")
    lines.append("")
    lines.append(f"Offset: {result.offset:.6g}")
    lines.append(f"IRF-Shift (ns): {result.irf_shift_ns:.6g}")
    lines.append(f"Mittlere Lebensdauer, amplitudengewichtet, tau_mean (ns): {result.tau_mean_ns:.6g}")
    if result.tau_mean_abs_ns is not None:
        lines.append(f"Mittlere Lebensdauer (Betragsamplituden) tau_mean_abs (ns): {result.tau_mean_abs_ns:.6g}")
    lines.append(f"Mittlere Lebensdauer, intensitaetsgewichtet, tau_mean_intensity (ns): {result.tau_mean_intensity_ns:.6g}")
    lines.append(f"Chi-Quadrat: {result.chi2:.6g}")
    lines.append(f"Reduziertes Chi-Quadrat: {result.red_chi2:.6g} (dof={result.dof})")

    with open(os.path.join(run_dir, "fit_parameters.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return run_dir


def save_global_fit_outputs(output_dir: str, result: GlobalFitResult) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"GlobalFit_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    raw_rows = []
    fit_rows = []
    res_rows = []
    irf_rows = []
    for k, label in enumerate(result.labels):
        wl = result.wavelengths[k]
        t = result.t_list[k]
        raw_rows.append(pd.DataFrame({"label": label, "wavelength_nm": wl, "time_ns": t, "intensity_raw": result.y_raw_list[k]}))
        fit_rows.append(pd.DataFrame({"label": label, "wavelength_nm": wl, "time_ns": t, "fit": result.fit_curves[k]}))
        res_rows.append(pd.DataFrame({"label": label, "wavelength_nm": wl, "time_ns": t, "residual": result.residuals_list[k]}))
        irf_t, irf_y = result.irf_display_list[k]
        irf_rows.append(pd.DataFrame({"label": label, "wavelength_nm": wl, "time_ns": irf_t, "irf_shifted": irf_y}))

    pd.concat(raw_rows, ignore_index=True).to_csv(os.path.join(run_dir, "raw_data_all.csv"), index=False)
    pd.concat(fit_rows, ignore_index=True).to_csv(os.path.join(run_dir, "fit_curves_all.csv"), index=False)
    pd.concat(res_rows, ignore_index=True).to_csv(os.path.join(run_dir, "residuals_all.csv"), index=False)
    pd.concat(irf_rows, ignore_index=True).to_csv(os.path.join(run_dir, "irf_used_all.csv"), index=False)

    # DAS: Amplitude jeder Komponente vs. Wellenlaenge.
    das_dict = {"wavelength_nm": result.wavelengths}
    for i in range(result.n_components):
        das_dict[f"A{i + 1} (tau={result.taus_ns[i]:.4g}ns)"] = [amps[i] for amps in result.amplitudes_per_ds]
    das_df = pd.DataFrame(das_dict).sort_values("wavelength_nm")
    das_df.to_csv(os.path.join(run_dir, "das.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    wl_arr = np.array(result.wavelengths)
    order = np.argsort(wl_arr)
    for i in range(result.n_components):
        amp_i = np.array([amps[i] for amps in result.amplitudes_per_ds])
        ax.plot(wl_arr[order], amp_i[order], "o-", label=f"tau={result.taus_ns[i]:.3g} ns")
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("Wellenlaenge (nm)")
    ax.set_ylabel("Amplitude A_i")
    ax.set_title("Decay-Associated Spectrum (DAS)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "das_plot.png"), dpi=150)
    plt.close(fig)

    # Uebersichts-Plot aller Fits.
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for k, label in enumerate(result.labels):
        ax2.semilogy(result.t_list[k], np.clip(result.y_raw_list[k], 1e-12, None), ".", markersize=2, alpha=0.5)
        ax2.semilogy(result.t_list[k], np.clip(result.fit_curves[k], 1e-12, None), "-", linewidth=1.2, label=label)
    ax2.set_xlabel("Zeit (ns)")
    ax2.set_ylabel("Intensitaet (log)")
    ax2.set_title("Globaler Fit: alle Datensaetze")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(run_dir, "global_fit_overview.png"), dpi=150)
    plt.close(fig2)

    lines = []
    lines.append("Globaler Fit - Ergebnisse")
    lines.append(f"Status: {'OK' if result.success else 'FEHLER/Warnung'} - {result.message}")
    lines.append(f"Anzahl Komponenten: {result.n_components}")
    lines.append("")
    lines.append("Gemeinsame Zeitkonstanten (ns):")
    for i, tau in enumerate(result.taus_ns):
        lines.append(f"  tau_{i + 1} = {tau:.6g} ns")
    lines.append("")
    for k, label in enumerate(result.labels):
        lines.append(f"--- {label} (wavelength = {result.wavelengths[k]} nm) ---")
        for i in range(result.n_components):
            lines.append(f"  A_{i + 1} = {result.amplitudes_per_ds[k][i]:.6g}")
        lines.append(f"  Offset = {result.offsets[k]:.6g}")
        lines.append(f"  IRF-Shift (ns) = {result.irf_shifts_ns[k]:.6g}")
        lines.append(f"  tau_mean, amplitudengewichtet (ns) = {result.tau_mean_per_ds[k]:.6g}")
        if result.tau_mean_abs_per_ds[k] is not None:
            lines.append(f"  tau_mean_abs (ns) = {result.tau_mean_abs_per_ds[k]:.6g}")
        lines.append(f"  tau_mean_intensity, intensitaetsgewichtet (ns) = {result.tau_mean_intensity_per_ds[k]:.6g}")
        lines.append(f"  Chi-Quadrat = {result.chi2_per_ds[k]:.6g}")
        lines.append(f"  Reduziertes Chi-Quadrat = {result.red_chi2_per_ds[k]:.6g}")
        lines.append("")
    lines.append(f"Gesamt Chi-Quadrat: {result.total_chi2:.6g}")
    lines.append(f"Gesamt reduziertes Chi-Quadrat: {result.total_red_chi2:.6g}")

    with open(os.path.join(run_dir, "global_fit_parameters.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return run_dir


# =============================================================================
# GUI
# =============================================================================

def adapt_preset_to_n(preset: List[float], n: int) -> List[float]:
    """Kuerzt oder erweitert eine Preset-Liste auf n Eintraege."""
    values = list(preset)
    if n <= len(values):
        return values[:n]
    while len(values) < n:
        values.append(values[-1] * 2.5)
    return values[:n]


class TCSPCFitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TCSPC Reconvolution Fitting Tool")
        self.geometry("980x760")

        self.irf_path = tk.StringVar()
        self.single_file_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.n_components = tk.IntVar(value=DEFAULT_N_COMPONENTS)
        self.tau_lower = tk.DoubleVar(value=DEFAULT_TAU_LOWER_NS)
        self.tau_upper = tk.DoubleVar(value=DEFAULT_TAU_UPPER_NS)

        self.allow_negative_amp = tk.BooleanVar(value=True)
        self.use_poisson_weight = tk.BooleanVar(value=True)
        self.fit_irf_shift = tk.BooleanVar(value=True)

        self.tau_guess_vars: List[tk.DoubleVar] = [tk.DoubleVar(value=v) for v in adapt_preset_to_n(PRESETS[1], 10)]

        self.global_files: List[Dict[str, Any]] = []  # {"path": str, "wavelength": float or None}

        self.result_queue: "queue.Queue" = queue.Queue()
        self._running = False

        self._build_widgets()
        self._rebuild_tau_guess_entries()

    # ------------------------------------------------------------------ UI

    def _build_widgets(self):
        pad = {"padx": 6, "pady": 4}

        files_frame = ttk.LabelFrame(self, text="Dateien")
        files_frame.pack(fill="x", **pad)

        ttk.Label(files_frame, text="IRF-Datei:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files_frame, textvariable=self.irf_path, width=70).grid(row=0, column=1, **pad)
        ttk.Button(files_frame, text="Durchsuchen...", command=self._browse_irf).grid(row=0, column=2, **pad)

        ttk.Label(files_frame, text="Einzel-Fit Datei:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files_frame, textvariable=self.single_file_path, width=70).grid(row=1, column=1, **pad)
        ttk.Button(files_frame, text="Durchsuchen...", command=self._browse_single_file).grid(row=1, column=2, **pad)

        ttk.Label(files_frame, text="Ausgabeordner:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files_frame, textvariable=self.output_dir, width=70).grid(row=2, column=1, **pad)
        ttk.Button(files_frame, text="Durchsuchen...", command=self._browse_output_dir).grid(row=2, column=2, **pad)

        # Globaler Fit: Dateiliste mit Wellenlaengen
        global_frame = ttk.LabelFrame(self, text="Globaler Fit - Dateien pro Wellenlaenge")
        global_frame.pack(fill="x", **pad)

        self.global_tree = ttk.Treeview(global_frame, columns=("path", "wavelength"), show="headings", height=5)
        self.global_tree.heading("path", text="Datei")
        self.global_tree.heading("wavelength", text="Wellenlaenge (nm)")
        self.global_tree.column("path", width=650)
        self.global_tree.column("wavelength", width=140, anchor="center")
        self.global_tree.grid(row=0, column=0, columnspan=4, sticky="nsew", **pad)
        self.global_tree.bind("<Double-1>", self._edit_global_wavelength)

        ttk.Button(global_frame, text="Dateien hinzufuegen...", command=self._add_global_files).grid(row=1, column=0, **pad)
        ttk.Button(global_frame, text="Auswahl entfernen", command=self._remove_global_selected).grid(row=1, column=1, **pad)
        ttk.Button(global_frame, text="Liste leeren", command=self._clear_global_files).grid(row=1, column=2, **pad)
        ttk.Label(global_frame, text="(Doppelklick auf Zeile aendert die Wellenlaenge)").grid(row=1, column=3, sticky="w", **pad)

        # Fit-Parameter
        params_frame = ttk.LabelFrame(self, text="Fit-Parameter")
        params_frame.pack(fill="x", **pad)

        ttk.Label(params_frame, text="Anzahl Komponenten (1-10):").grid(row=0, column=0, sticky="w", **pad)
        n_spin = ttk.Spinbox(params_frame, from_=1, to=10, textvariable=self.n_components, width=5,
                              command=self._rebuild_tau_guess_entries)
        n_spin.grid(row=0, column=1, sticky="w", **pad)
        n_spin.bind("<Return>", lambda e: self._rebuild_tau_guess_entries())
        n_spin.bind("<FocusOut>", lambda e: self._rebuild_tau_guess_entries())

        ttk.Label(params_frame, text="tau untere Grenze (ns):").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(params_frame, textvariable=self.tau_lower, width=8).grid(row=0, column=3, **pad)
        ttk.Label(params_frame, text="tau obere Grenze (ns):").grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(params_frame, textvariable=self.tau_upper, width=8).grid(row=0, column=5, **pad)

        ttk.Checkbutton(params_frame, text="Negative Amplituden erlauben", variable=self.allow_negative_amp).grid(
            row=1, column=0, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(params_frame, text="Poisson-Gewichtung verwenden", variable=self.use_poisson_weight).grid(
            row=1, column=2, columnspan=2, sticky="w", **pad)
        ttk.Checkbutton(params_frame, text="IRF-Shift fitten", variable=self.fit_irf_shift).grid(
            row=1, column=4, columnspan=2, sticky="w", **pad)

        # Presets
        preset_frame = ttk.Frame(params_frame)
        preset_frame.grid(row=2, column=0, columnspan=6, sticky="w", **pad)
        ttk.Label(preset_frame, text="Presets fuer tau-Startwerte:").pack(side="left", padx=4)
        for i in range(1, 5):
            ttk.Button(preset_frame, text=f"Preset {i}", command=lambda i=i: self._apply_preset(i)).pack(side="left", padx=4)

        # Dynamische tau-Startwert-Eingaben
        self.tau_guess_frame = ttk.LabelFrame(self, text="Initial Guesses tau_i (ns)")
        self.tau_guess_frame.pack(fill="x", **pad)

        # Aktionen
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", **pad)
        self.single_fit_btn = ttk.Button(action_frame, text="Einzel-Fit starten", command=self._run_single_fit_clicked)
        self.single_fit_btn.pack(side="left", padx=8)
        self.global_fit_btn = ttk.Button(action_frame, text="Globalen Fit starten", command=self._run_global_fit_clicked)
        self.global_fit_btn.pack(side="left", padx=8)

        # Log
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=14, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _rebuild_tau_guess_entries(self):
        for widget in self.tau_guess_frame.winfo_children():
            widget.destroy()
        n = self._get_n_components_safe()
        for i in range(n):
            ttk.Label(self.tau_guess_frame, text=f"tau_{i + 1}:").grid(row=i // 5, column=(i % 5) * 2, sticky="w", padx=4, pady=2)
            ttk.Entry(self.tau_guess_frame, textvariable=self.tau_guess_vars[i], width=8).grid(
                row=i // 5, column=(i % 5) * 2 + 1, sticky="w", padx=4, pady=2)

    def _get_n_components_safe(self) -> int:
        try:
            n = int(self.n_components.get())
        except (tk.TclError, ValueError):
            n = DEFAULT_N_COMPONENTS
        n = max(1, min(10, n))
        self.n_components.set(n)
        return n

    def _apply_preset(self, preset_id: int):
        n = self._get_n_components_safe()
        values = adapt_preset_to_n(PRESETS[preset_id], n)
        lower = self.tau_lower.get()
        upper = self.tau_upper.get()
        for i in range(n):
            v = min(max(values[i], max(lower, TAU_MIN_EPS_NS)), upper)
            self.tau_guess_vars[i].set(round(v, 6))
        self.log(f"Preset {preset_id} angewendet (n={n}).")

    # ------------------------------------------------------------- File I/O

    def _browse_irf(self):
        path = filedialog.askopenfilename(title="IRF-Datei waehlen",
                                           filetypes=[("Text/CSV", "*.txt *.csv *.dat"), ("Alle Dateien", "*.*")])
        if path:
            self.irf_path.set(path)

    def _browse_single_file(self):
        path = filedialog.askopenfilename(title="Rohdaten-Datei (Einzel-Fit) waehlen",
                                           filetypes=[("Text/CSV", "*.txt *.csv *.dat"), ("Alle Dateien", "*.*")])
        if path:
            self.single_file_path.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Ausgabeordner waehlen")
        if path:
            self.output_dir.set(path)

    def _add_global_files(self):
        paths = filedialog.askopenfilenames(title="Rohdaten-Dateien (globaler Fit) waehlen",
                                             filetypes=[("Text/CSV", "*.txt *.csv *.dat"), ("Alle Dateien", "*.*")])
        for path in paths:
            wl = parse_wavelength_from_filename(path)
            self.global_files.append({"path": path, "wavelength": wl})
        self._refresh_global_tree()

    def _remove_global_selected(self):
        selected = self.global_tree.selection()
        indices = sorted((self.global_tree.index(item) for item in selected), reverse=True)
        for idx in indices:
            del self.global_files[idx]
        self._refresh_global_tree()

    def _clear_global_files(self):
        self.global_files = []
        self._refresh_global_tree()

    def _refresh_global_tree(self):
        for item in self.global_tree.get_children():
            self.global_tree.delete(item)
        for entry in self.global_files:
            wl_display = entry["wavelength"] if entry["wavelength"] is not None else "?"
            self.global_tree.insert("", "end", values=(entry["path"], wl_display))

    def _edit_global_wavelength(self, event):
        item = self.global_tree.identify_row(event.y)
        if not item:
            return
        idx = self.global_tree.index(item)
        current = self.global_files[idx]["wavelength"]
        new_val = simpledialog.askfloat("Wellenlaenge bearbeiten", "Wellenlaenge (nm):",
                                         initialvalue=current if current is not None else 0.0, parent=self)
        if new_val is not None:
            self.global_files[idx]["wavelength"] = new_val
            self._refresh_global_tree()

    # ------------------------------------------------------------------ Log

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_running(self, running: bool):
        self._running = running
        state = "disabled" if running else "normal"
        self.single_fit_btn.configure(state=state)
        self.global_fit_btn.configure(state=state)

    # -------------------------------------------------------- Validation

    def _get_common_fit_params(self):
        n = self._get_n_components_safe()
        tau_lower = float(self.tau_lower.get())
        tau_upper = float(self.tau_upper.get())
        if tau_upper <= tau_lower:
            raise ValueError("Die obere tau-Grenze muss groesser sein als die untere Grenze.")
        tau_guess = [float(self.tau_guess_vars[i].get()) for i in range(n)]
        return {
            "n_components": n,
            "tau_lower": tau_lower,
            "tau_upper": tau_upper,
            "tau_guess": tau_guess,
            "allow_negative_amp": bool(self.allow_negative_amp.get()),
            "use_poisson_weights": bool(self.use_poisson_weight.get()),
            "fit_irf_shift": bool(self.fit_irf_shift.get()),
        }

    # -------------------------------------------------------- Single fit

    def _run_single_fit_clicked(self):
        if self._running:
            return
        try:
            if not self.irf_path.get():
                raise ValueError("Bitte eine IRF-Datei auswaehlen.")
            if not self.single_file_path.get():
                raise ValueError("Bitte eine Rohdaten-Datei fuer den Einzel-Fit auswaehlen.")
            if not self.output_dir.get():
                raise ValueError("Bitte einen Ausgabeordner auswaehlen.")
            params = self._get_common_fit_params()
        except Exception as exc:
            messagebox.showerror("Ungueltige Eingabe", str(exc))
            return

        self._set_running(True)
        self.log("Starte Einzel-Fit...")
        thread = threading.Thread(target=self._worker_wrapper, args=(self._do_single_fit, params), daemon=True)
        thread.start()
        self.after(200, self._poll_queue)

    def _do_single_fit(self, params: Dict[str, Any]) -> str:
        irf_t, irf_y = load_two_column_file(self.irf_path.get())
        t, y = load_two_column_file(self.single_file_path.get())
        label = os.path.splitext(os.path.basename(self.single_file_path.get()))[0]
        wavelength = parse_wavelength_from_filename(self.single_file_path.get())

        result = fit_single_dataset(
            label=label,
            wavelength=wavelength,
            t=t, y_raw=y,
            irf_t_raw=irf_t, irf_y_raw=irf_y,
            **params,
        )
        run_dir = save_single_fit_outputs(self.output_dir.get(), result, label)
        return run_dir

    # -------------------------------------------------------- Global fit

    def _run_global_fit_clicked(self):
        if self._running:
            return
        try:
            if not self.irf_path.get():
                raise ValueError("Bitte eine IRF-Datei auswaehlen.")
            if len(self.global_files) < 2:
                raise ValueError("Bitte mindestens zwei Dateien fuer den globalen Fit hinzufuegen.")
            for entry in self.global_files:
                if entry["wavelength"] is None:
                    raise ValueError(f"Fuer Datei '{entry['path']}' fehlt die Wellenlaenge (per Doppelklick editierbar).")
            if not self.output_dir.get():
                raise ValueError("Bitte einen Ausgabeordner auswaehlen.")
            params = self._get_common_fit_params()
        except Exception as exc:
            messagebox.showerror("Ungueltige Eingabe", str(exc))
            return

        self._set_running(True)
        self.log("Starte globalen Fit...")
        thread = threading.Thread(target=self._worker_wrapper, args=(self._do_global_fit, params), daemon=True)
        thread.start()
        self.after(200, self._poll_queue)

    def _do_global_fit(self, params: Dict[str, Any]) -> str:
        irf_t, irf_y = load_two_column_file(self.irf_path.get())
        datasets = []
        for entry in self.global_files:
            t, y = load_two_column_file(entry["path"])
            label = f"{entry['wavelength']:g}nm"
            datasets.append((label, entry["wavelength"], t, y))

        result = fit_global_datasets(
            datasets=datasets,
            irf_t_raw=irf_t, irf_y_raw=irf_y,
            **params,
        )
        run_dir = save_global_fit_outputs(self.output_dir.get(), result)
        return run_dir

    # -------------------------------------------------------- Threading

    def _worker_wrapper(self, fn, *args):
        try:
            value = fn(*args)
            self.result_queue.put(("ok", value))
        except Exception:
            self.result_queue.put(("error", traceback.format_exc()))

    def _poll_queue(self):
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_queue)
            return

        self._set_running(False)
        if status == "ok":
            self.log(f"Fertig. Ergebnisse gespeichert in:\n{payload}")
            messagebox.showinfo("Fertig", f"Fit abgeschlossen.\nErgebnisse gespeichert in:\n{payload}")
        else:
            self.log("FEHLER:\n" + payload)
            messagebox.showerror("Fehler", "Der Fit ist fehlgeschlagen. Details siehe Log-Fenster.")


# =============================================================================
# Main
# =============================================================================

def main():
    app = TCSPCFitApp()
    app.mainloop()


if __name__ == "__main__":
    main()

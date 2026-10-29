#!/usr/bin/env python3
"""Schrittzahlen aus unabhängigen Trainings zusammenfassen.

Aufruf im Ordner mit den Ergebnisdateien:
    python auswertung_seeds.py
Oder mit einem ausdrücklich angegebenen Datenordner:
    python auswertung_seeds.py /pfad/zum/datenordner

Benötigt nur numpy und matplotlib. Es werden keine Simulationen ausgeführt.
Alle Eingabedateien müssen zu denselben übrigen Versuchsparametern gehören.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re

import numpy as np


ALPHAS = (0.01, 0.999999)
DATEINAME = re.compile(
    r"^Aufgabe_4_Schrittzahlen_a"
    r"(?P<alpha>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"_seed(?P<seed>\d+)_.+\.txt$"
)


def einlesen(ordner):
    """Liefert pro alpha und Seed genau eine Datei samt ihren drei Spalten."""
    laeufe = {alpha: {} for alpha in ALPHAS}
    for pfad in sorted(ordner.glob("Aufgabe_4_Schrittzahlen_a*_seed*_*.txt")):
        treffer = DATEINAME.fullmatch(pfad.name)
        if treffer is None:
            raise ValueError(f"Dateiname nicht erkannt: {pfad.name}")
        alpha = float(treffer["alpha"])
        if alpha not in laeufe:
            continue
        seed = int(treffer["seed"])
        if seed in laeufe[alpha]:
            bisher = laeufe[alpha][seed][0]
            raise ValueError(
                f"Doppelter Seed {seed} für alpha={alpha}:\n"
                f"  {bisher.name}\n  {pfad.name}\n"
                "Bitte nur eine Datei pro alpha und Seed im Datenordner behalten."
            )

        daten = np.loadtxt(pfad, ndmin=2)
        if daten.size == 0 or daten.shape[1] != 3:
            raise ValueError(f"{pfad.name}: Erwartet werden P, mean_steps, std_steps.")
        if not np.isfinite(daten).all():
            raise ValueError(f"{pfad.name}: Enthält fehlende oder unendliche Werte.")
        daten = daten[np.argsort(daten[:, 0])]
        if np.any((daten[:, 0] < 0) | (daten[:, 0] > 1)):
            raise ValueError(f"{pfad.name}: P muss zwischen 0 und 1 liegen.")
        if np.any(np.diff(daten[:, 0]) <= 0):
            raise ValueError(f"{pfad.name}: Ein P-Wert kommt mehrfach vor.")
        if np.any(daten[:, 1] <= 0) or np.any(daten[:, 2] < 0):
            raise ValueError(f"{pfad.name}: Ungültige Schrittzahlen oder Streuungen.")
        laeufe[alpha][seed] = (pfad, daten)
    return laeufe


def zusammenfassen(laeufe):
    """Mittelt die Trainingsmittelwerte; Episodenstreuungen werden nicht gemittelt."""
    for alpha in ALPHAS:
        if not laeufe[alpha]:
            raise ValueError(f"Keine passenden Seed-Dateien für alpha={alpha} gefunden.")

    seeds = sorted(laeufe[ALPHAS[0]])
    if set(seeds) != set(laeufe[ALPHAS[1]]):
        raise ValueError(
            "Für beide Lernraten müssen dieselben Seeds vorliegen. Gefunden:\n"
            + "\n".join(f"alpha={a}: {sorted(laeufe[a])}" for a in ALPHAS)
        )

    p_werte = laeufe[ALPHAS[0]][seeds[0]][1][:, 0]
    ergebnisse = {}
    for alpha in ALPHAS:
        trainingsmittel = []
        for seed in seeds:
            pfad, daten = laeufe[alpha][seed]
            if len(daten) != len(p_werte) or not np.allclose(
                daten[:, 0], p_werte, rtol=0, atol=1e-10
            ):
                raise ValueError(f"Abweichendes P-Raster in {pfad.name}.")
            # Spalte 1 enthält den Mittelwert der Auswertungsepisoden eines Trainings.
            # Spalte 2 beschreibt Episodenstreuung, nicht Streuung zwischen Seeds.
            trainingsmittel.append(daten[:, 1])

        trainingsmittel = np.asarray(trainingsmittel)
        mittelwert = np.mean(trainingsmittel, axis=0)
        # Stichprobenstandardabweichung zwischen unabhängigen Trainings.
        # Bei nur einem Seed ist diese nicht bestimmbar, daher NaN statt null.
        seed_std = (
            np.std(trainingsmittel, axis=0, ddof=1)
            if len(seeds) > 1
            else np.full(len(p_werte), np.nan)
        )
        ergebnisse[alpha] = (mittelwert, seed_std)
    return p_werte, seeds, ergebnisse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datenordner", nargs="?", type=Path, default=Path("."))
    parser.add_argument("--no-show", action="store_true", help="Grafik nur speichern.")
    args = parser.parse_args()
    ordner = args.datenordner.expanduser().resolve()
    if not ordner.is_dir():
        parser.error(f"Kein Datenordner: {ordner}")

    try:
        laeufe = einlesen(ordner)
        p_werte, seeds, ergebnisse = zusammenfassen(laeufe)
    except (ValueError, OSError) as fehler:
        parser.exit(1, f"Auswertung abgebrochen: {fehler}\n")

    if len(seeds) == 1:
        print("Hinweis: Nur ein Seed je alpha; noch keine Seed-Streuung bestimmbar.")

    ausgabe = ordner / "auswertung_seedvergleich"
    ausgabe.mkdir(exist_ok=True)
    zeit = datetime.now().strftime("%Y-%m-%dT%H_%M_%S_%f")
    txt_pfad = ausgabe / f"Schrittzahlen_Seedvergleich_{zeit}.txt"
    png_pfad = ausgabe / f"Schrittzahlen_Seedvergleich_{zeit}.png"

    zeilen = []
    for alpha in ALPHAS:
        mittelwert, seed_std = ergebnisse[alpha]
        for p, mittel, streuung in zip(p_werte, mittelwert, seed_std):
            zeilen.append((alpha, p, len(seeds), mittel, streuung))

    quellen = [
        f"alpha={alpha}, seed={seed}: {laeufe[alpha][seed][0].name}"
        for alpha in ALPHAS for seed in seeds
    ]
    header = (
        "Mittelwert der Trainingsmittelwerte; gleiche Gewichtung aller Seeds.\n"
        "std_between_seeds: Stichprobenstandardabweichung (ddof=1) der Trainingsmittelwerte.\n"
        "Keine Konfidenzintervalle. NaN bedeutet: Mit einem Seed nicht bestimmbar.\n"
        "Die Episodenstandardabweichungen der Eingabedateien werden nicht gemittelt.\n"
        "Quellen:\n" + "\n".join(quellen) + "\n"
        "alpha P_obstacle n_seeds mean_steps std_between_seeds"
    )
    np.savetxt(
        txt_pfad, zeilen,
        fmt=["%.8g", "%.8g", "%d", "%.10g", "%.10g"], header=header
    )

    # Import erst nach der Datenprüfung. --no-show funktioniert auch ohne GUI.
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for alpha in ALPHAS:
        mittelwert, seed_std = ergebnisse[alpha]
        ax.errorbar(
            p_werte, mittelwert,
            yerr=seed_std if len(seeds) > 1 else None,
            fmt="o-", capsize=4,
            label=rf"$\alpha={alpha:g}$ ({len(seeds)} Seeds)"
        )
    ax.set_xlabel(r"$P_{\mathrm{obstacle}}$")
    ax.set_ylabel("Mittlere Schritte pro Episode")
    ax.set_title("Schrittzahlen ohne Exploration: Vergleich der Lernraten")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.text(
        0.5, 0.012,
        "Fehlerbalken: ±1 Standardabweichung zwischen den Trainingsmittelwerten"
        if len(seeds) > 1 else "Nur ein Seed: keine Fehlerbalken bestimmbar",
        ha="center", va="bottom", fontsize=9
    )
    # Platz für die Erläuterung unterhalb der Achsenbeschriftung lassen.
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    fig.savefig(png_pfad, dpi=150)

    print(f"Verwendete Seeds je alpha: {seeds}")
    print(f"Ergebnistabelle: {txt_pfad}")
    print(f"Vergleichsplot:  {png_pfad}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

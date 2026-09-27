"""Font inventory shared by the atlas builder and the synthetic OCR test generator."""
from __future__ import annotations

import os
from pathlib import Path

WIN = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
try:
    import matplotlib

    MPL = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
except Exception:  # noqa: BLE001
    MPL = Path("/nonexistent")

# Fonts the atlas is built from: the common Windows UI faces + DejaVu (the Linux/Chrome fallback).
ATLAS_FONTS = [
    WIN / "segoeui.ttf", WIN / "segoeuib.ttf", WIN / "seguisb.ttf", WIN / "segoeuil.ttf", WIN / "segoeuii.ttf",
    WIN / "arial.ttf", WIN / "arialbd.ttf", WIN / "ariali.ttf",
    WIN / "tahoma.ttf", WIN / "tahomabd.ttf",
    WIN / "verdana.ttf", WIN / "verdanab.ttf",
    WIN / "calibri.ttf", WIN / "calibrib.ttf",
    WIN / "times.ttf", WIN / "timesbd.ttf",
    WIN / "georgia.ttf", WIN / "georgiab.ttf",
    WIN / "consola.ttf", WIN / "cour.ttf",
    WIN / "trebuc.ttf", WIN / "trebucbd.ttf",
    WIN / "impact.ttf", WIN / "bahnschrift.ttf", WIN / "micross.ttf", WIN / "framd.ttf",
    MPL / "DejaVuSans.ttf", MPL / "DejaVuSans-Bold.ttf", MPL / "DejaVuSerif.ttf",
    # Heavier / display / script coverage (web banners, ads)
    WIN / "ariblk.ttf", WIN / "seguibl.ttf", WIN / "calibril.ttf", WIN / "segoeuisl.ttf", WIN / "arialbi.ttf",
    WIN / "verdanai.ttf", WIN / "timesi.ttf", WIN / "georgiai.ttf", WIN / "consolab.ttf", WIN / "courbd.ttf",
    WIN / "CascadiaCode.ttf", WIN / "LeelawUI.ttf", WIN / "LeelaUIb.ttf", WIN / "sylfaen.ttf",
    WIN / "Inkfree.ttf", WIN / "segoepr.ttf",
]

# Never seen by the atlas: used to measure generalisation to unknown typefaces.
HELDOUT_FONTS = [
    WIN / "candara.ttf", WIN / "constan.ttf", WIN / "corbel.ttf", WIN / "comic.ttf",
    WIN / "gadugi.ttf", WIN / "ebrima.ttf", WIN / "lucon.ttf", WIN / "pala.ttf", WIN / "cambriab.ttf",
    MPL / "DejaVuSansMono.ttf",
]

# Characters the recogniser knows. ASCII printable + the extra symbols common in UIs / web banners.
CHARSET = [chr(c) for c in range(33, 127)] + list("€£¥©®°•–—’“”«»áéíóúñÁÉÍÓÚÑüÜäöÄÖß×ı")


def available(fonts):
    return [f for f in fonts if f.exists()]

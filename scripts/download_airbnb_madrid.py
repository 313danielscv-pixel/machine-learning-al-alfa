from __future__ import annotations

import re
import urllib.request
from pathlib import Path

INDEX_URL = "https://insideairbnb.com/get-the-data/"
OFFICIAL_FILE_PREFIX = "https://data.insideairbnb.com/"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "listings.csv.gz"


def find_madrid_listings_url() -> str:
    request = urllib.request.Request(
        INDEX_URL,
        headers={"User-Agent": "MachineLearningAlAlfa/1.0 (educational project)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    madrid_section = re.search(
        r"<h3>\s*Madrid,[\s\S]*?</table>",
        html,
        flags=re.IGNORECASE,
    )
    if madrid_section is None:
        raise RuntimeError(
            "No se encontro la tabla de Madrid en la pagina oficial de Inside Airbnb."
        )
    match = re.search(
        r'href="(https://data\.insideairbnb\.com/[^"]+/madrid/'
        r'\d{4}-\d{2}-\d{2}/data/listings\.csv\.gz)"',
        madrid_section.group(0),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise RuntimeError(
            "La tabla oficial de Madrid no contiene un enlace listings.csv.gz conocido."
        )
    url = match.group(1)
    if not url.startswith(OFFICIAL_FILE_PREFIX):
        raise RuntimeError("Se rechazo un enlace de datos que no pertenece al dominio oficial.")
    return url


def download() -> Path:
    url = find_madrid_listings_url()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MachineLearningAlAlfa/1.0 (educational project)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    if not content.startswith(b"\x1f\x8b"):
        raise RuntimeError("La descarga no parece ser un archivo gzip valido.")
    DESTINATION.write_bytes(content)
    print(f"Descargado: {DESTINATION}")
    print(f"Fuente oficial: {url}")
    print(f"Tamano: {len(content):,} bytes")
    print("Licencia declarada por Inside Airbnb: Creative Commons Attribution 4.0.")
    return DESTINATION


if __name__ == "__main__":
    download()

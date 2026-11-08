"""Ingesta de microdatos EXPO (DANE/ANDA) para la Actividad 2 de MLOps.

Lee los CSV mensuales dentro de ZIP anidados sin extraerlos a disco.
No utiliza Prefect, MLflow ni modifica las configuraciones del modelo.

Uso (desde la raiz del repositorio):
    poetry run python src/ingesta_anda.py

Entradas locales (ignoradas por Git):
    data/raw/Expo_2025.zip
    data/raw/Expo_2026.zip

Salidas locales (ignoradas por Git):
    data/processed/train_202508_202605.csv.gz
    data/processed/valid_202606_202608.csv.gz
    data/processed/sept_202609.csv.gz
    data/processed/reporte_ingesta.json
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
CHUNK_SIZE = 20_000

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
PERIODOS = [f"{y}-{m:02d}" for y, a, b in [(2025, 8, 12), (2026, 1, 9)]
            for m in range(a, b + 1)]
ARCHIVOS = {
    "train": "train_202508_202605.csv.gz",
    "valid": "valid_202606_202608.csv.gz",
    "sept": "sept_202609.csv.gz",
}
COLUMNAS_SALIDA = [
    "periodo", "capitulo", "modad", "dpto_origen", "pnk", "fobdol", "destino_usa"
]
COLUMNAS_ORIGEN = ["FECH", "PAIS", "COD_PAI4", "POSAR", "MODAD", "DPTO1", "PNK", "FOBDOL"]
NUMERO_ES = re.compile(r"^-?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?$")
ZIP_MES = re.compile(
    r"(?:^|[^a-z])(" + "|".join(MESES) + r")\s+(20\d{2})\.zip$", re.IGNORECASE
)


def conjunto(periodo: str) -> str:
    if "2025-08" <= periodo <= "2026-05":
        return "train"
    if "2026-06" <= periodo <= "2026-08":
        return "valid"
    if periodo == "2026-09":
        return "sept"
    raise ValueError(f"Periodo fuera del contrato: {periodo}")


def localizar_meses() -> dict[str, tuple[Path, str]]:
    """Ubica los 14 ZIP mensuales sin depender del orden de los archivos."""
    ubicaciones: dict[str, tuple[Path, str]] = {}
    for anio in (2025, 2026):
        principal = RAW_DIR / f"Expo_{anio}.zip"
        if not principal.is_file():
            raise FileNotFoundError(f"Falta el archivo original: {principal}")
        with ZipFile(principal) as z:
            for nombre in z.namelist():
                coincidencia = ZIP_MES.search(Path(nombre).name.lower())
                if coincidencia is None:
                    continue
                mes, anio_nombre = coincidencia.groups()
                if int(anio_nombre) != anio:
                    raise ValueError(f"Año inconsistente en {principal.name}: {nombre}")
                periodo = f"{anio}-{MESES[mes.lower()]:02d}"
                if periodo not in PERIODOS:
                    continue
                if periodo in ubicaciones:
                    raise ValueError(f"ZIP mensual duplicado: {periodo}")
                ubicaciones[periodo] = (principal, nombre)
    faltantes = sorted(set(PERIODOS) - ubicaciones.keys())
    if faltantes:
        raise ValueError(f"Faltan meses necesarios: {', '.join(faltantes)}")
    return ubicaciones


def numero_colombiano(columna: pd.Series) -> pd.Series:
    """Convierte '33.716,76' a 33716.76 y marca entradas inválidas como NaN."""
    valores = columna.astype("string").str.strip()
    formato_correcto = valores.str.fullmatch(NUMERO_ES.pattern).fillna(False)
    normalizados = valores.where(formato_correcto, pd.NA)
    normalizados = normalizados.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(normalizados, errors="coerce")


def procesar_bloque(bloque: pd.DataFrame, periodo: str) -> tuple[pd.DataFrame, Counter]:
    """Normaliza un bloque, registra exclusiones y evita fuga directa del destino."""
    bloque.columns = [nombre.strip().upper() for nombre in bloque.columns]
    ausentes = sorted(set(COLUMNAS_ORIGEN) - set(bloque.columns))
    if ausentes:
        raise ValueError(f"Faltan columnas en {periodo}: {', '.join(ausentes)}")

    d = bloque.loc[:, COLUMNAS_ORIGEN].copy()
    for columna in COLUMNAS_ORIGEN:
        d[columna] = d[columna].astype("string").str.strip()

    cod_pais = d["PAIS"]
    iso_pais = d["COD_PAI4"].str.upper()
    posicion = d["POSAR"]
    peso = numero_colombiano(d["PNK"])
    fob = numero_colombiano(d["FOBDOL"])
    periodo_fuente = f"{int(periodo[:4]) % 100:02d}{periodo[5:7]}"

    pais_valido = cod_pais.str.fullmatch(r"\d{3}").fillna(False)
    iso_valido = iso_pais.str.fullmatch(r"[A-Z]{3}").fillna(False)
    coherente_usa = (cod_pais.eq("840") == iso_pais.eq("USA")).fillna(False)

    motivos = pd.Series("", index=d.index, dtype="string")

    def excluir(condicion: pd.Series, etiqueta: str) -> None:
        nonlocal motivos
        motivos = motivos.mask(motivos.eq("") & condicion.fillna(True), etiqueta)

    excluir(d["FECH"].ne(periodo_fuente), "periodo_incorrecto")
    excluir(~pais_valido | ~iso_valido | ~coherente_usa, "pais_invalido")
    excluir(~posicion.str.fullmatch(r"\d{10}").fillna(False), "posar_invalida")
    excluir(d["MODAD"].eq("") | d["DPTO1"].eq(""), "categoria_faltante")
    excluir(peso.isna() | peso.le(0), "peso_invalido")
    excluir(fob.isna() | fob.lt(0), "fob_invalido")

    aptos = motivos.eq("")
    salida = pd.DataFrame({
        "periodo": periodo,
        "capitulo": posicion.str[:2],
        "modad": d["MODAD"],
        # DPTO1 viene como '5' o '08'; 2 caracteres permiten códigos uniformes.
        "dpto_origen": d["DPTO1"].str.zfill(2),
        "pnk": peso,
        "fobdol": fob,
        "destino_usa": cod_pais.eq("840").astype("int64"),
    }, index=d.index).loc[aptos, COLUMNAS_SALIDA]

    if not salida["destino_usa"].isin([0, 1]).all():
        raise AssertionError("La variable objetivo no es binaria")
    return salida, Counter(motivos.loc[~aptos].tolist())


def ejecutar() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meses = localizar_meses()
    resumen = defaultdict(lambda: {"originales": 0, "validos": 0, "usa": 0, "otros": 0, "excluidos": Counter()})
    temporales = {clave: OUT_DIR / (nombre + ".tmp") for clave, nombre in ARCHIVOS.items()}

    try:
        with ExitStack() as salidas:
            handles = {clave: salidas.enter_context(gzip.open(
                ruta, "wt", encoding="utf-8", newline=""
            )) for clave, ruta in temporales.items()}
            encabezado_escrito = {clave: False for clave in ARCHIVOS}

            for periodo in PERIODOS:
                principal, nombre_mes = meses[periodo]
                grupo = conjunto(periodo)
                with ZipFile(principal) as externo:
                    with ZipFile(BytesIO(externo.read(nombre_mes))) as interno:
                        csvs = [nombre for nombre in interno.namelist() if nombre.lower().endswith(".csv")]
                        if len(csvs) != 1:
                            raise ValueError(f"Se esperaba un solo CSV en {nombre_mes}: {csvs}")
                        with interno.open(csvs[0]) as datos:
                            bloques = pd.read_csv(
                                datos, sep=";", encoding="utf-8-sig", dtype="string",
                                keep_default_na=False, chunksize=CHUNK_SIZE
                            )
                            for bloque in bloques:
                                procesados, excluidos = procesar_bloque(bloque, periodo)
                                r = resumen[grupo]
                                r["originales"] += len(bloque)
                                r["validos"] += len(procesados)
                                r["usa"] += int(procesados["destino_usa"].sum())
                                r["otros"] += int((procesados["destino_usa"] == 0).sum())
                                r["excluidos"].update(excluidos)
                                procesados.to_csv(handles[grupo], index=False,
                                                   header=not encabezado_escrito[grupo])
                                encabezado_escrito[grupo] = True
                print(f"{periodo}: procesado")

        reporte = {}
        for grupo, valores in resumen.items():
            reporte[grupo] = {
                **{k: int(valores[k]) for k in ("originales", "validos", "usa", "otros")},
                "porcentaje_usa": round(100 * valores["usa"] / valores["validos"], 4) if valores["validos"] else 0,
                "excluidos_por_motivo": dict(valores["excluidos"]),
                "periodos": {"train": ["2025-08", "2026-05"],
                             "valid": ["2026-06", "2026-08"],
                             "sept": ["2026-09", "2026-09"]}[grupo]
            }
        for grupo, temporal in temporales.items():
            if not temporal.is_file():
                raise RuntimeError(f"No se generó la salida de {grupo}")
            temporal.replace(OUT_DIR / ARCHIVOS[grupo])
        (OUT_DIR / "reporte_ingesta.json").write_text(
            json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\nRESULTADOS (después de limpieza):")
        for grupo in ("train", "valid", "sept"):
            r = reporte[grupo]
            print(f"  {ARCHIVOS[grupo]}: {r['validos']:,} registros; "
                  f"USA: {r['usa']:,} ({r['porcentaje_usa']} %); "
                  f"excluidos: {sum(r['excluidos_por_motivo'].values()):,}")
        print(f"Auditoría: {OUT_DIR / 'reporte_ingesta.json'}")
    finally:
        for temporal in temporales.values():
            temporal.unlink(missing_ok=True)


if __name__ == "__main__":
    ejecutar()

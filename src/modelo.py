
"""Modelo supervisado: destino USA frente a otros paises.

Logica de Machine Learning independiente de Prefect y MLflow.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OBJETIVO = "destino_usa"

NUMERICAS = ["pnk", "fobdol"]
CATEGORICAS = ["capitulo", "modad", "dpto_origen"]
VARIABLES = NUMERICAS + CATEGORICAS


def resolver_variables(config_features):
    """Selecciona variables usando ALL, NUMERIC o CUSTOM."""

    seleccion = str(config_features["seleccion"]).upper()

    if seleccion == "ALL":
        elegidas = VARIABLES.copy()

    elif seleccion == "NUMERIC":
        elegidas = NUMERICAS.copy()

    elif seleccion == "CUSTOM":
        elegidas = config_features.get("custom", [])

    else:
        raise ValueError(
            f"Seleccion no valida: {seleccion}. "
            "Use ALL, NUMERIC o CUSTOM."
        )

    if not elegidas:
        raise ValueError("No se seleccionaron variables.")

    desconocidas = [v for v in elegidas if v not in VARIABLES]
    if desconocidas:
        raise ValueError(
            f"Variables no permitidas: {desconocidas}"
        )

    if len(elegidas) != len(set(elegidas)):
        raise ValueError("Hay variables duplicadas.")

    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]

    return numericas, categoricas


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Construye preprocesamiento y clasificador."""

    if tipo != "logistic_regression":
        raise ValueError(f"Modelo no soportado: {tipo}")

    transformadores = []

    if numericas:
        procesamiento_num = Pipeline([
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ])
        transformadores.append(
            ("numericas", procesamiento_num, numericas)
        )

    if categoricas:
        procesamiento_cat = Pipeline([
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformadores.append(
            ("categoricas", procesamiento_cat, categoricas)
        )

    if not transformadores:
        raise ValueError("El modelo no tiene variables predictoras.")

    preprocesador = ColumnTransformer(
        transformers=transformadores
    )

    clasificador = LogisticRegression(**parametros)

    return Pipeline([
        ("preprocesamiento", preprocesador),
        ("clasificador", clasificador),
    ])


def calcular_metricas(y_real, probabilidades, umbral=0.5):
    """Calcula las metricas para clasificacion binaria."""

    y_real = np.asarray(y_real)
    probabilidades = np.asarray(probabilidades)

    if len(np.unique(y_real)) != 2:
        raise ValueError(
            "El AUC requiere las dos clases en validacion."
        )

    if not 0 <= umbral <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1.")

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    return {
        "auc": float(
            roc_auc_score(y_real, probabilidades)
        ),
        "accuracy": float(
            accuracy_score(y_real, predicciones)
        ),
        "precision": float(
            precision_score(
                y_real, predicciones, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_real, predicciones, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_real, predicciones, zero_division=0
            )
        ),
    }

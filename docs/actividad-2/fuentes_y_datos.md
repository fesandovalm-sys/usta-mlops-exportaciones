# Fuente y preparación de datos — Actividad 2 de MLOps

## 1. Fuente y alcance

Se utilizaron los microdatos públicos de la operación estadística de exportaciones (EXPO) del Departamento Administrativo Nacional de Estadística (DANE), disponibles en el Archivo Nacional de Datos (ANDA).

- **Fuente:** [Microdatos EXPO 2025–2026 — DANE](https://microdatos.dane.gov.co/index.php/catalog/859/get-microdata).
- **Diccionario:** [Descripción de variables — DANE](https://microdatos.dane.gov.co/index.php/catalog/859/data-dictionary/F1).
- **Período:** agosto de 2025 a septiembre de 2026.
- **Unidad de análisis:** registro del archivo de microdatos de exportaciones.

El ejercicio consiste en clasificar si un registro de exportación tiene como país de destino Estados Unidos (`destino_usa = 1`) u otro país válido (`destino_usa = 0`).

Se trata de un prototipo académico de clasificación supervisada, no de un modelo oficial del DANE.

## 2. Organización e ingesta

Los datos originales se encuentran en `Expo_2025.zip` y `Expo_2026.zip`, que contienen archivos ZIP mensuales con versiones CSV, Stata y SPSS.

Se seleccionó el formato CSV. Los archivos presentan 27 columnas, separador de punto y coma (`;`), codificación UTF-8 con BOM y números con formato decimal colombiano.

El programa `src/ingesta_anda.py` procesa los 14 meses mediante lectura por bloques de 20.000 registros, sin extraer permanentemente los archivos originales.

Los ZIP y los conjuntos procesados se conservan localmente en `data/`, excluida del control de versiones.

## 3. Transformación y contrato de datos

Se definió un esquema único de siete columnas:

| Columna | Tipo o representación | Transformación |
|---|---|---|
| `periodo` | Texto `YYYY-MM` | Mes y año del archivo, contrastados con `FECH` |
| `capitulo` | Texto de dos dígitos | Primeros dos dígitos de `POSAR` |
| `modad` | Texto categórico | Variable `MODAD` |
| `dpto_origen` | Texto categórico | Variable `DPTO1`, normalizada a dos caracteres |
| `pnk` | Numérica | Peso neto `PNK`, convertido a formato decimal |
| `fobdol` | Numérica | Valor FOB en dólares `FOBDOL`, convertido a formato decimal |
| `destino_usa` | Entero binario | Clasificación derivada de `PAIS` |

Se conserva la representación textual de los códigos categóricos para evitar la pérdida de ceros iniciales. Los valores numéricos se transforman del formato colombiano al formato requerido para su procesamiento en Python.

### Variable objetivo y prevención de fuga de información

Se identificó a Estados Unidos mediante `PAIS = 840`, contrastado con `COD_PAI4 = USA`.

Los registros con identificadores faltantes, inválidos o inconsistentes se excluyen del procesamiento.

Los campos `PAIS`, `COD_PAI4` y `POSAR` no se incorporan directamente a los conjuntos procesados; de `POSAR` únicamente se conserva el capítulo arancelario.

Las variables predictoras acordadas son `capitulo`, `modad`, `dpto_origen`, `pnk` y `fobdol`. Las columnas `periodo` y `destino_usa` se mantienen para identificación temporal y evaluación, pero se excluyen de las entradas predictoras.

## 4. Separación temporal

Se establecieron tres conjuntos sin superposición de períodos:

| Conjunto | Período | Archivo generado |
|---|---|---|
| Entrenamiento | Agosto 2025 – mayo 2026 | `train_202508_202605.csv.gz` |
| Validación | Junio – agosto 2026 | `valid_202606_202608.csv.gz` |
| Aplicación temporal | Septiembre 2026 | `sept_202609.csv.gz` |

La etiqueta real de septiembre se conserva para una evaluación retrospectiva posterior, pero no debe suministrarse al modelo durante la inferencia.

Todos los archivos se generan como CSV comprimidos mediante gzip, con encabezado y codificación UTF-8.

## 5. Validaciones y resultados

El procesamiento original comprendió 559.450 registros. Después de aplicar las reglas de validación se conservaron 559.397 registros y se excluyeron 53.

| Conjunto | Registros válidos | Destino USA | Otros destinos | % USA | Excluidos |
|---|---:|---:|---:|---:|---:|
| Entrenamiento | 404.214 | 51.751 | 352.463 | 12,80 % | 49 |
| Validación | 114.830 | 14.333 | 100.497 | 12,48 % | 3 |
| Septiembre | 40.353 | 4.558 | 35.795 | 11,30 % | 1 |
| **Total** | **559.397** | **70.642** | **488.755** | **12,63 %** | **53** |

Se verificaron las siguientes condiciones:

- Coincidencia de los nombres y el orden de las siete columnas.
- Correspondencia de los registros con sus períodos.
- Existencia exclusiva de etiquetas binarias 0 y 1.
- Tipos numéricos de `pnk` y `fobdol`.
- Formato de dos dígitos para `capitulo`.
- Ausencia de valores nulos en los tres archivos finales.

Los criterios de exclusión comprenden inconsistencias en países de destino, períodos, posiciones arancelarias, categorías y valores numéricos inválidos, según las validaciones implementadas. El detalle por motivo está disponible en el reporte local `data/processed/reporte_ingesta.json`.

## 6. Pruebas y reproducibilidad

Se desarrollaron 21 pruebas automatizadas en `tests/test_ingesta_anda.py`, utilizando datos sintéticos independientes de los microdatos originales.

Las pruebas cubren conversión numérica, construcción de etiquetas, preservación de ceros iniciales, separación temporal, columnas obligatorias, exclusiones documentadas y reproducibilidad.

**Resultado:** 21 pruebas aprobadas con Pytest 9.2.0 y Python 3.12.10.

Para reproducir el procesamiento desde la raíz del repositorio:

```powershell
poetry run python src/ingesta_anda.py
```

Para ejecutar las pruebas, con Pytest instalado:

```powershell
poetry run python -m pytest tests/test_ingesta_anda.py -v
```

El código y las pruebas se publicaron en la rama `lili/actividad-2`, mediante el commit `c513ac9`.

## 7. Limitaciones

El conjunto presenta una distribución de clases desbalanceada: aproximadamente el 12,63 % de los registros válidos corresponde a Estados Unidos. Por ello, la evaluación del modelo debe considerar AUC-ROC y métricas complementarias.

La depuración identifica inconsistencias según las reglas implementadas, pero no sustituye los controles oficiales de calidad de la operación estadística.

La exclusión de identificadores directos del país reduce el riesgo de fuga del objetivo, aunque no descarta asociaciones indirectas que deban revisarse durante el modelado.

Los resultados del modelo que posteriormente desarrolle el equipo deberán interpretarse como evidencia de un ejercicio académico y no como pronósticos oficiales de exportaciones.

## 8. Responsabilidad y trazabilidad

La preparación de datos, la construcción de `destino_usa`, la separación temporal y las pruebas de ingesta corresponden a la contribución de Liliana Herrera.

El desarrollo del clasificador, sus configuraciones, los flujos de Prefect, el registro en MLflow y la API de FastAPI corresponden a la contribución de Fredy Sandoval.

La implementación de ingesta y sus pruebas contaron con asistencia de ChatGPT para estructuración de código y verificaciones; las ejecuciones y resultados consignados se comprobaron en el entorno local del proyecto. La documentación diferencia esta asistencia del trabajo realizado y validado por los integrantes.
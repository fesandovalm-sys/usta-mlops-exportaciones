# Actividad 3 — Monitoreo, drift y CI

**Asignatura:** MLOps · Maestría en Ciencia de Datos, Universidad Santo Tomás  
**Integrantes:** Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez  
**Repositorio:** [usta-mlops-exportaciones](https://github.com/fesandovalm-sys/usta-mlops-exportaciones)  
**Guía:** [Actividad 3 — Henry Cala](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md)  
**Estado:** documentación preliminar; resultados de integración y monitoreo pendientes de incorporar.

## Objetivo y continuidad del proyecto

Extender el sistema de clasificación desarrollado en la Actividad 2 con simulación controlada de cambios en los datos, detección de *feature drift* y *target drift*, alertas y pruebas automatizadas. Se mantiene el caso de estudio basado en microdatos de exportaciones colombianas EXPO del DANE/ANDA: estimar si una operación tiene como destino Estados Unidos (`destino_usa = 1`) u otro destino válido (`destino_usa = 0`). El modelo registrado en MLflow utiliza el alias `champion`.

## Componentes del proyecto

| Componente | Archivo o ubicación | Estado documentado |
|---|---|---|
| Simulación, validación y métricas de drift | [`src/monitoreo.py`](../../src/monitoreo.py) | Implementado y probado |
| Pruebas de monitoreo | [`tests/test_monitoreo.py`](../../tests/test_monitoreo.py) | Implementadas y ejecutadas |
| Pruebas de contrato de datos | [`tests/test_datos.py`](../../tests/test_datos.py) | Implementadas y ejecutadas |
| Modelo y variables | [`src/modelo.py`](../../src/modelo.py) | Proveniente de la Actividad 2 |
| Entrenamiento y registro | [`flows/entrenamiento_flow.py`](../../flows/entrenamiento_flow.py) | Proveniente de la Actividad 2 |
| Inferencia con drift emulado y conservación del lote | [`flows/inferencia_flow.py`](../../flows/inferencia_flow.py) | Integración y evidencia final pendientes |
| Flujos de detección con Prefect/Evidently | `flows/` | Pendiente de verificar archivos definitivos y ejecuciones |
| API FastAPI | [`api/main.py`](../../api/main.py) | Proveniente de la Actividad 2 |
| Parámetros de monitoreo | [`configs/config.yaml`](../../configs/config.yaml) | Configuración existente; validación integral pendiente |
| Memoria de actividad | [`Memoria3.md`](Memoria3.md) | Borrador en elaboración |

## Simulación controlada de drift

La función `aplicar_drift(lote, nivel, semilla=42)` admite niveles entre 0 y 1; para la actividad se consideran `0.0`, `0.3` y `0.6`. Desplaza las variables numéricas `pnk` y `fobdol` mediante `2 × nivel × escala`, donde la escala se obtiene de la desviación estándar del lote (o de una escala alternativa si la variable es constante). Impide valores negativos y conserva el lote de entrada sin modificar.

La variable objetivo se modifica **solo con fines de simulación**: se convierten aleatoriamente algunas etiquetas `0` en `1` hasta aumentar su tasa en un máximo de `0.30 × nivel` puntos de proporción, sujeto al límite de 1. La columna `destino_usa_original` preserva las etiquetas observadas anteriores al cambio. La semilla permite reproducir la selección. **La tasa simulada no representa un cambio real en las exportaciones.**

`validar_lote` rechaza lotes vacíos, sin objetivo, con nulos o etiquetas distintas de 0 y 1. `drift_de_target` contrasta las proporciones binarias mediante chi-cuadrado y devuelve tasas, valor p y decisión de drift. `fraccion_desde_evidently` extrae la proporción de variables con drift a partir de la métrica `DriftedColumnsCount`.

## Monitoreo y alertas — pendiente de validación integral

La solución prevista compara cada lote de inferencia con una referencia de entrenamiento, con dos flujos de Prefect: uno para distribuciones de variables mediante Evidently y otro para la proporción de `destino_usa`, que requiere etiquetas. La configuración contempla `umbral_drift: 0.30` y `alfa_evento: 0.05`. **Pendiente:** verificar los flujos definitivos, el origen del conjunto de referencia, la persistencia de reportes por lote y las alertas efectivamente emitidas.

**Pendiente de evidencia:** ejecuciones y archivos diferenciados de niveles `0.0`, `0.3` y `0.6`; reportes comparativos de `0.0` y `0.6`; comprobación de alerta solo por encima del umbral; registro de ejecuciones en Prefect/MLflow. No se atribuyen resultados a esas ejecuciones hasta contar con sus evidencias.

## Pruebas verificadas

Desde la raíz del repositorio, con el entorno Poetry instalado:

    poetry install --with dev
    poetry run python -m pytest -q

**Resultado local confirmado:** `44 passed in 2.67s`, después de sincronizar la rama con `main` e instalar dependencias. Las pruebas de monitoreo y contrato de datos utilizan datos sintéticos y no requieren los servicios MLflow/Prefect ni los microdatos originales. Este resultado **no demuestra por sí solo** la ejecución de los flujos integrados ni el cumplimiento de las pruebas de API, modelo e inferencia exigidas por la guía; su cobertura y evidencias quedan por verificar.

**Trazabilidad:** [PR #16](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/16) integrado en `main` (merge `53da2e2`), correspondiente a `src/monitoreo.py` y las dos suites de pruebas.

## Pendientes para cerrar la actividad

- Confirmar los nombres y el comportamiento final de los dos flujos de drift, incluidos reportes y alertas.
- Incorporar resultados reales de los tres niveles y evidencias que permitan contrastar `0.0` frente a `0.6`.
- Comprobar que los lotes usados cumplen el criterio de procedencia de la partición de prueba establecido en la guía.
- Verificar cobertura de pruebas de modelo, inferencia y API, y CI únicamente si se implementa.
- Completar la [Memoria 3](Memoria3.md) y generar el PDF solo tras la validación final.

## Referencias

- [Guía oficial de Actividad 3](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md)
- [Plantilla de memoria](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md)
- [Documentación de la Actividad 2](../actividad-2/README.md)

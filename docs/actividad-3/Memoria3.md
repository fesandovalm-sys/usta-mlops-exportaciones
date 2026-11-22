# Memoria 3 — Monitoreo, drift y CI

**Asignatura:** MLOps · Maestría en Ciencia de Datos, Universidad Santo Tomás

**Integrantes:** Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez

**Caso de estudio:** clasificación del destino de exportaciones colombianas (EXPO, DANE/ANDA)

## 1. Objetivo

Extender el clasificador binario de la Actividad 2 para monitorear cambios en las variables predictoras (*feature drift*) y en el objetivo (*target drift*) mediante perturbaciones reproducibles, flujos de Prefect, alertas configurables y pruebas automatizadas. Se conserva la variable `destino_usa` (1: Estados Unidos; 0: otros destinos válidos) y el modelo registrado con el alias `champion` en MLflow.

## 2. Descripción del trabajo

Se desarrolló `src/monitoreo.py` con funciones de validación de lotes, simulación, contraste del objetivo mediante chi-cuadrado y extracción de métricas de Evidently, separadas de la orquestación. Se incorporaron `tests/test_monitoreo.py` y `tests/test_datos.py`, que emplean datos sintéticos. El flujo `flows/inferencia_flow.py` utiliza Random Forest `champion` v2, selecciona lotes de 500 registros de septiembre de 2026, aplica los niveles `0.0`, `0.3` y `0.6`, conserva `destino_usa_original` y registra predicciones y artefactos en MLflow.

Se implementaron dos flujos adicionales de Prefect: `flows/drift_features_flow.py`, que compara cinco características con una referencia reproducible de 10.000 registros de entrenamiento mediante Evidently, y `flows/drift_evento_flow.py`, que contrasta las proporciones binarias del objetivo con chi-cuadrado. Ambos guardan resúmenes y emiten mensajes de alerta según sus criterios configurados. Las nueve ejecuciones definitivas de inferencia y monitoreo finalizaron en estado `Completed`.

## 3. Decisiones técnicas

Se separó la lógica estadística de Prefect y MLflow para facilitar su prueba aislada. La simulación desplaza `pnk` y `fobdol`, modifica de manera controlada la proporción del objetivo y utiliza la semilla `42` para reproducibilidad. La etiqueta original se preserva para distinguir observaciones reales y valores artificiales. Las perturbaciones no describen cambios reales en las exportaciones.

Evidently se adoptó de acuerdo con la guía de la actividad, sin evaluación comparativa de herramientas. Para `destino_usa`, por su naturaleza binaria, se eligió chi-cuadrado para contrastar proporciones entre referencia y lote, con significación `α = 0,05`; no se realizó comparación formal con otros contrastes.

El umbral de características se fijó en **30 %** como regla operativa configurable, no como límite estadístico universal: entre cinco variables, una con drift equivale al 20 % y dos al 40 %, de modo que la alerta se activa al detectarse al menos dos. El valor configurado se aplica también al reporte con `DataDriftPreset(drift_share=umbral)`; previamente se identificó una discrepancia con el valor predeterminado de Evidently. No se optimizó estadísticamente este umbral.

## 4. Evidencias

El módulo de monitoreo y sus pruebas se integraron en el [PR #16](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/16); los flujos y nueve capturas, en el [PR #17](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/17); y la corrección del umbral del reporte, en el [PR #18](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/18). La suite registró **44 pruebas automatizadas satisfactorias**.

| Nivel | Inferencia (Prefect) | Características (Evidently) | Objetivo (Prefect) |
|---|---|---|---|
| `0.0` | [Evidencia 01](evidencias/01_prefect_inferencia_drift_0.png) | [Evidencia 02](evidencias/02_evidently_features_drift_0.png) | [Evidencia 03](evidencias/03_prefect_target_drift_0.png) |
| `0.3` | [Evidencia 04](evidencias/04_prefect_inferencia_drift_03.png) | [Evidencia 05](evidencias/05_evidently_features_drift_03.png) | [Evidencia 06](evidencias/06_prefect_target_drift_03.png) |
| `0.6` | [Evidencia 07](evidencias/07_prefect_inferencia_drift_06.png) | [Evidencia 08](evidencias/08_evidently_features_drift_06.png) | [Evidencia 09](evidencias/09_prefect_target_drift_06.png) |

## 5. Resultados

Se evaluaron tres niveles de perturbación sobre lotes de **500 registros**, utilizando **10.000 registros de entrenamiento** como referencia. La tasa de destino USA en la referencia fue **12,90 %**.

| Nivel | AUC temporal | Características con drift | Tasa USA del lote | Valor p del objetivo | Alerta características | Alerta objetivo |
|---|---:|---:|---:|---:|---|---|
| `0.0` | 0,7464 | 20 % (1/5) | 13,40 % | 0,797220 | No | No |
| `0.3` | 0,7029 | 40 % (2/5) | 22,40 % | < 0,000001 | Sí | Sí |
| `0.6` | 0,6752 | 60 % (3/5) | 31,40 % | < 0,000001 | Sí | Sí |

En `0.0` se conservaron las **67 etiquetas positivas originales**, sin modificaciones; aun así, Evidently identificó drift en una variable. La fracción del 20 % no superó el umbral global del 30 % y el contraste del objetivo tampoco resultó significativo (`p = 0,797220`). En `0.3` y `0.6` aumentaron las características con drift y la tasa simulada de destino USA; ambos sistemas generaron alertas. Los valores p de estos dos escenarios fueron inferiores a `0,000001`; el valor mostrado por Prefect como `0.000000` corresponde a redondeo, no a un valor p exactamente nulo.

El AUC temporal descendió de **0,7464** a **0,6752** entre los escenarios extremos. Estos resultados corresponden a drift **emulado** y no son mediciones de desempeño en producción ni permiten inferir, por sí solos, una relación causal generalizable.

## 6. Problemas encontrados

Durante la integración de la inferencia se corrigió un error de indentación y se validó la sintaxis de Python. La integración mediante ramas y solicitudes de incorporación se comprobó con las 44 pruebas automatizadas. Adicionalmente, se detectó que el reporte de Evidently utilizaba un umbral predeterminado de **50 %**, diferente del **30 %** establecido en YAML. Se corrigió mediante `DataDriftPreset(drift_share=umbral)`, se regeneraron los tres reportes y se verificó visualmente la aplicación del umbral; el ajuste quedó integrado mediante el PR #18. Las nueve ejecuciones definitivas terminaron sin fallos.

## 7. Conclusiones

La integración de inferencia, monitoreo y alertas permitió verificar el comportamiento del sistema bajo tres niveles de drift artificial. El escenario `0.0` evidenció que la ausencia de perturbación inducida no implica distribuciones idénticas a las de entrenamiento, ni exige una alerta cuando el cambio detectado queda por debajo del umbral operativo. Los escenarios `0.3` y `0.6` activaron las alertas de características y objetivo, acompañados de una reducción observada del AUC.

Las 44 pruebas respaldan las funciones automatizadas de ingesta, validación y monitoreo, pero no establecen cobertura unitaria específica del modelo, la inferencia o FastAPI. Se recomienda ampliar esa cobertura y contrastar los umbrales y el desempeño en otros períodos y lotes antes de considerar un monitoreo operativo. No se implementó CI con GitHub Actions, opción no obligatoria de la guía.

## 8. Uso de herramientas de IA

Se utilizó **ChatGPT** como apoyo para la elaboración y revisión del módulo de monitoreo y sus pruebas, la implementación y depuración de flujos de Prefect y Evidently, la preparación de comandos de PowerShell, la gestión de Git y la redacción documental. Sus propuestas se sometieron a comprobación mediante ejecución de pruebas, validación de sintaxis, revisión de reportes y contrastación con los resultados obtenidos; se corrigieron las soluciones que lo requirieron, incluida la discrepancia del umbral de Evidently.

**Liliana Carolina Herrera Prieto** desarrolló y probó, con asistencia de IA, `src/monitoreo.py`, `tests/test_monitoreo.py` y `tests/test_datos.py`, integrados mediante el PR #16; ejecutó las pruebas, gestionó su rama y preparó la documentación académica. **Fredy Ernesto Sandoval Méndez** desarrolló y verificó, también con asistencia de IA, la integración de inferencia y los dos flujos de drift; ejecutó los escenarios, revisó Prefect, MLflow y Evidently, recopiló las evidencias y gestionó los PR #17 y #18. La definición del caso, la aceptación de cambios, la interpretación de los resultados y la validación de las ejecuciones fueron responsabilidad del equipo, no decisiones delegadas a la IA.

**Referencias:** [Guía oficial de la Actividad 3](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md) · [Plantilla de memoria](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md) · [Repositorio del proyecto](https://github.com/fesandovalm-sys/usta-mlops-exportaciones).

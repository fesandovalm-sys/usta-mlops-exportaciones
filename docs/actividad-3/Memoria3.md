# Memoria 3 — Monitoreo, drift y CI

**MLOps · Maestría en Ciencia de Datos · Universidad Santo Tomás**  
**Integrantes:** Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez  
**Estado:** borrador. Los campos **[PENDIENTE]** requieren resultados y evidencias verificables.

## 1. Objetivo

Extender el clasificador de destino de exportaciones colombianas de las actividades previas con simulación reproducible de drift, monitoreo de variables y del objetivo, alertas y pruebas automatizadas. El caso de estudio conserva los microdatos EXPO del DANE/ANDA y la clasificación `destino_usa`: Estados Unidos (`1`) frente a otros destinos válidos (`0`).

## 2. Descripción del trabajo

Se desarrolló `src/monitoreo.py` con cuatro funciones independientes de Prefect y MLflow: `aplicar_drift`, `validar_lote`, `drift_de_target` y `fraccion_desde_evidently`. La simulación contempla niveles `0.0`, `0.3` y `0.6`; altera `pnk`, `fobdol` y la proporción simulada del objetivo, conservando las etiquetas originales en `destino_usa_original`.

Se añadieron `tests/test_monitoreo.py` y `tests/test_datos.py` para validar la lógica mediante datos sintéticos. Estos archivos se integraron mediante el [PR #16](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/pull/16). El entrenamiento y la API provienen de la Actividad 2.

Como primera comprobación de integración, se ejecutó la inferencia con nivel de drift `0.0`, utilizando el modelo Random Forest `champion` v2 y un lote de 500 registros de septiembre de 2026. Prefect registró la ejecución como **Completed**, con cuatro tareas finalizadas. La simulación conservó las 67 etiquetas positivas originales, sin modificaciones.

**[PENDIENTE — integración de Fredy]** Documentar los resultados de los niveles `0.3` y `0.6` y describir los flujos definitivos de detección de *feature drift* (Evidently) y *target drift* con Prefect, sus reportes y alertas.

## 3. Decisiones técnicas

La lógica de monitoreo se separó de la orquestación para permitir pruebas unitarias sin servicios levantados. La emulación aplica a `pnk` y `fobdol` un desplazamiento de `2 × nivel × escala`, con límite inferior de cero, y aumenta artificialmente la proporción de `destino_usa = 1` hasta `0.30 × nivel` puntos de proporción.

Una semilla fija permite reproducibilidad y `destino_usa_original` distingue las etiquetas observadas de las simuladas. **Estos cambios no representan una variación real de las exportaciones.**

Para el objetivo binario se utiliza chi-cuadrado (nivel de significación configurado: `0.05`); la fracción de variables con drift se extrae del resultado de Evidently.

**[PENDIENTE]** Justificar el umbral final y documentar las alternativas efectivamente consideradas o descartadas durante la integración.

## 4. Evidencias

- **Pruebas locales:** `poetry run python -m pytest -q` → **`44 passed in 2.67s`**, después de instalar las dependencias y sincronizar la rama.
- **Versionamiento:** PR #16 integrado, commit `9323146`, merge `53da2e2`.
- **Primera inferencia, nivel 0.0:** Prefect `Completed`, cuatro tareas finalizadas, modelo Random Forest `champion` v2, lote de 500 registros y ninguna etiqueta modificada.
- **[PENDIENTE DE INCORPORAR]** Captura `evidencias/01_prefect_inferencia_drift_0.png`, anunciada por Fredy como evidencia de la primera inferencia.
- **[PENDIENTE]** Evidencias de los niveles `0.3` y `0.6`, reportes Evidently, resultados de los dos flujos de detección y logs de alertas.
- **[PENDIENTE]** Evidencias adicionales de pruebas de modelo, API e inferencia, y de CI únicamente si se implementó.

## 5. Resultados

Se comprobaron **44 pruebas locales satisfactorias**. Las pruebas del módulo de monitoreo emplearon datos sintéticos, sin depender de MLflow, Prefect ni microdatos originales.

En la primera ejecución de inferencia con nivel `0.0`, Fredy verificó que el lote conservó sus 67 etiquetas originales de destino USA, sin modificaciones simuladas. El AUC temporal registrado fue **0,7464**. Este resultado corresponde al desempeño del modelo en el lote evaluado y no constituye una medición de drift estadístico.

| Criterio | Resultado |
|---|---|
| Inferencia nivel `0.0` | Completada; 500 registros, 0 etiquetas modificadas |
| AUC temporal, nivel `0.0` | 0,7464 |
| Inferencias niveles `0.3` y `0.6` | **[PENDIENTE]** |
| Reporte de *feature drift* por lote | **[PENDIENTE]** |
| Comparación estadística del target | **[PENDIENTE]** |
| Sin alerta en `0.0`; alerta por encima del umbral en `0.6` | **[PENDIENTE]** |
| Reportes estadísticos distintos entre `0.0` y `0.6` | **[PENDIENTE]** |

La ejecución con nivel `0.0` confirma que no se introdujeron modificaciones mediante la simulación. Sin embargo, **todavía no se ha calculado el drift estadístico con Evidently**, por lo que no puede afirmarse que el lote carezca de drift frente al entrenamiento.

## 6. Problemas encontrados

Unas modificaciones locales previas de la Memoria 2 impedían la actualización mediante `git pull`. Se preservaron con `git stash`, se sincronizó el repositorio y, tras comprobar que los archivos coincidían con las versiones integradas, se descartó el respaldo temporal.

El PR #16 se fusionó sin conflictos; las pruebas volvieron a pasar después del merge.

**[PENDIENTE — Fredy]** Documentar únicamente incidentes reales en la integración de inferencia, Prefect y Evidently, con sus soluciones.

## 7. Conclusiones

Se obtuvo una base de monitoreo comprobable y reproducible, separada de los servicios de orquestación. Preservar la etiqueta original evita confundir una alteración simulada con un hecho observado.

La primera inferencia con drift emulado de nivel `0.0` se completó satisfactoriamente, sin modificar las etiquetas del lote. Este resultado constituye una validación inicial de la integración, pero no permite establecer todavía el comportamiento del sistema ante drift ni la efectividad de las alertas.

**[PENDIENTE]** Completar las conclusiones con los resultados contrastados de los niveles `0.0` y `0.6`, la detección estadística y las limitaciones observadas.

## 8. Uso de herramientas de IA

Se utilizó asistencia de IA para elaborar el módulo de drift y sus pruebas, orientar la solución de incidencias de Git y preparar esta documentación.

Liliana ejecutó y verificó las pruebas, gestionó su rama y el PR, y comprobó los resultados locales. Fredy coordina la integración de inferencia y los flujos de detección con apoyo de su IA.

**[PENDIENTE — declaración individual]** Precisar qué propuestas de IA revisó, corrigió o descartó cada integrante y qué partes realizó sin IA. Validar esta atribución con Fredy antes de entregar.

**Fuentes:** [guía oficial](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-3.md) · [plantilla de memoria](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md) · [repositorio del equipo](https://github.com/fesandovalm-sys/usta-mlops-exportaciones).

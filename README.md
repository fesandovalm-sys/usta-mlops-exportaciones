# MLOps aplicado a exportaciones colombianas

Proyecto académico de la **Maestría en Ciencia de Datos de la Universidad Santo Tomás**, desarrollado por **Liliana Carolina Herrera Prieto y Fredy Ernesto Sandoval Méndez** para la asignatura **MLOps**.

Este repositorio reúne las actividades del curso y sus implementaciones progresivas. Cada actividad conserva su propia documentación; este archivo es el punto de entrada general.

## Actividades

| Actividad | Alcance | Documentación | Memoria |
|---|---|---|---|
| **1. Fundamentos, versionamiento y tracking** | Configuración de Poetry, Git/GitHub, Prefect, MLflow y Docker Compose; validación de la integración de herramientas. | [README de la Actividad 1](docs/actividad-1/README.md) | [Memoria 1](docs/actividad-1/Memoria1.md) · [PDF](docs/actividad-1/Memoria1.pdf) |
| **2. Pipeline y empaquetado** | Clasificación del destino de exportaciones colombianas (Estados Unidos frente a otros países), entrenamiento y evaluación, registro y promoción del modelo `champion`, inferencia diaria y API FastAPI. | [README de la Actividad 2](docs/actividad-2/README.md) | [Memoria 2](docs/actividad-2/Memoria2.md) · [PDF](docs/actividad-2/Memoria2.pdf) |

## Organización del repositorio

- `src/`: preparación de microdatos y construcción de modelos.
- `flows/`: entrenamiento, inferencia y despliegues de Prefect.
- `api/`: servicio de predicción con FastAPI.
- `configs/`: parámetros y selección de variables en YAML.
- `tests/`: pruebas automatizadas.
- `services/`: configuración de Prefect y MLflow con Docker Compose.
- `docs/actividad-1/` y `docs/actividad-2/`: instrucciones, documentación y evidencias de cada actividad.

## Entorno y reproducibilidad

El proyecto utiliza **Python, Poetry, Prefect, MLflow y Scikit-learn**. Las dependencias se especifican en [`pyproject.toml`](pyproject.toml) y [`poetry.lock`](poetry.lock). Para configurar el entorno y consultar las instrucciones de ejecución, utilizar los README de cada actividad.

Los microdatos originales y procesados, los archivos de predicciones y los registros locales de MLflow y Prefect **no forman parte del repositorio**. La Actividad 2 documenta cómo preparar los datos y reproducir los flujos; las [evidencias de sus ejecuciones](docs/actividad-2/evidencias/) sí están versionadas.

## Referencias del curso

- [Repositorio de materiales del profesor](https://github.com/henrycala/santoto-mlops-content).
- [Guía oficial de la Actividad 1](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-1.md).
- [Guía oficial de la Actividad 2](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-2.md).

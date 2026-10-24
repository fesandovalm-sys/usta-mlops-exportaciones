# MLOps aplicado al análisis de exportaciones colombianas

Proyecto académico de la **Maestría en Ciencia de Datos de la Universidad Santo Tomás**, desarrollado en el curso de **MLOps**.

**Integrantes:** Fredy Sandoval y Liliana Herrera.  
**Repositorio público:** https://github.com/fesandovalm-sys/usta-mlops-exportaciones

## 1. Propósito y relación con la tesis

Este repositorio organiza los ejercicios y desarrollos prácticos del curso de MLOps. Como caso de uso para las siguientes actividades, proponemos abordar la **detección de valores unitarios FOB potencialmente atípicos en las exportaciones colombianas**, considerando variables como la **subpartida arancelaria**, el **país de destino** y la **modalidad de exportación**.

Este caso de uso **forma parte de los temas analíticos que se explorarán en el trabajo de tesis** sobre **observabilidad del procesamiento de los datos de la operación estadística de exportaciones del DANE**. La identificación de valores atípicos se plantea como un **componente complementario** de ese trabajo de investigación; no constituye por sí sola la totalidad del sistema de observabilidad.

**Alcance de la Actividad 1:** configurar el repositorio colaborativo, administrar las dependencias con Poetry, poner en funcionamiento Prefect y MLflow y comprobar su integración con un flujo de prueba. **En esta actividad no se utiliza todavía un conjunto de datos de exportaciones ni se entrena un modelo de detección de atípicos.** Esas tareas se abordarán en etapas posteriores del curso y de la investigación, según su alcance y disponibilidad de datos.

## 2. Herramientas y versiones

| Herramienta | Versión o configuración | Uso |
| --- | --- | --- |
| Python | 3.12 en los equipos; el proyecto admite `>=3.12,<3.15` | Ejecución de código |
| Poetry | 2.x | Instalación y gestión de dependencias |
| Prefect | 3.8.8 | Orquestación y seguimiento del flujo |
| MLflow | 3.17.0 | Registro de experimentos, parámetros y métricas |
| Scikit-learn | 1.9.1 | Dependencia preparada para desarrollos posteriores |
| Pandas | 3.0.6 | Dependencia preparada para tratamiento de datos |
| NumPy | 2.5.4 | Dependencia preparada para cálculo numérico |
| PyYAML | 6.0.3 | Dependencia preparada para configuraciones |
| Docker Desktop / Docker Compose | Instalación compatible con Compose V2 | Despliegue de los servidores Prefect y MLflow |
| Git / GitHub | — | Control de versiones y colaboración |

Las dependencias se declaran en [`pyproject.toml`](pyproject.toml), y sus versiones resueltas se conservan en [`poetry.lock`](poetry.lock). Los dos archivos están versionados.

## 3. Estructura del repositorio

```text
usta-mlops-exportaciones/
├── README.md
├── .gitignore
├── pyproject.toml
├── poetry.lock
├── Memoria1.md
├── Memoria1.pdf
├── flows/
│   └── prueba_flow.py
├── services/
│   └── docker-compose.yml
├── configs/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   └── .gitkeep
├── tests/
│   └── .gitkeep
└── docs/
    └── evidencias/
        ├── 01_RepositorioGitHub.png
        ├── 02_colaboradores_git.png
        ├── 03_fredy_poetry.png
        ├── 04_lili_poetry.png
        ├── 05_fredy_prefect.png
        ├── 06_fredy_mlflow.png
        ├── 07_lili_prefect.png
        ├── 08_lili_mlflow.png
        └── 09_docker_compose.png
```

Las carpetas `configs/`, `notebooks/`, `src/` y `tests/` están reservadas para actividades posteriores; los archivos `.gitkeep` permiten mantenerlas en Git sin presentar desarrollos que aún no se han realizado.

La **[Memoria 1 en Markdown](Memoria1.md)** y su **[versión PDF](Memoria1.pdf)** documentan las decisiones técnicas, dificultades, evidencias, resultados, conclusiones y uso de IA. Este README se enfoca en **cómo instalar y reproducir la prueba**.

## 4. Requisitos e instalación

Las instrucciones siguientes están orientadas a **Windows y PowerShell**, los entornos empleados por el equipo.

**Requisitos:** Git, Python 3.12, Poetry 2.x y, para la modalidad con contenedores, Docker Desktop con motor Linux en funcionamiento.

Para verificar las herramientas:

```powershell
python --version
poetry --version
git --version
docker compose version
```

Para clonar el repositorio e instalar las dependencias, desde la carpeta en la que se desee trabajar:

```powershell
git clone https://github.com/fesandovalm-sys/usta-mlops-exportaciones.git
cd usta-mlops-exportaciones
poetry install
```

Validar el intérprete y las versiones **efectivamente instaladas** en el entorno de Poetry:

```powershell
poetry run python -c "import sys; from importlib.metadata import version; print('Python:', sys.version.split()[0]); print('Prefect:', version('prefect')); print('MLflow:', version('mlflow')); print('Scikit-learn:', version('scikit-learn')); print('Pandas:', version('pandas')); print('NumPy:', version('numpy')); print('PyYAML:', version('PyYAML'))"
```

Si Poetry informa `No dependencies to install or update`, significa que no necesita instalar o actualizar paquetes con respecto a su archivo de dependencias. Si no encuentra un intérprete compatible, puede seleccionarse Python 3.12 mediante `poetry env use 3.12` o su ruta ejecutable.

En uno de los equipos, una ruta de Windows demasiado extensa causó problemas al instalar dependencias. Se resolvió configurando el directorio de entornos virtuales en `C:\venvs`; se trata de una **preferencia local**, no de un requisito para todos los usuarios ni de un archivo que deba subirse a Git.

## 5. Iniciar Prefect y MLflow

La actividad se validó inicialmente con servidores lanzados directamente desde Poetry y, posteriormente, con **Docker Compose**. Ambas modalidades utilizan los puertos 4200 y 5000: **debe emplearse solo una a la vez**.

### Opción A. Docker Compose

Con Docker Desktop abierto, ejecutar desde la **raíz del repositorio**:

```powershell
docker compose -f .\services\docker-compose.yml up -d
docker compose -f .\services\docker-compose.yml ps
```

Los servicios deben figurar en estado `Up`. Verificar sus API:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:4200/api/health" -UseBasicParsing
Invoke-WebRequest -Uri "http://127.0.0.1:5000/health" -UseBasicParsing
```

Ambos endpoints deben responder con estado **HTTP 200**. Si un contenedor tarda en iniciar o presenta errores, consultar:

```powershell
docker compose -f .\services\docker-compose.yml logs --tail=50
```

### Opción B. Servidores locales con Poetry

Si se utiliza la modalidad sin Docker, abrir **dos terminales** en la raíz del repositorio y mantenerlas abiertas:

**Terminal 1 — Prefect:**

```powershell
poetry run prefect server start --host 127.0.0.1 --port 4200
```

**Terminal 2 — MLflow:**

```powershell
poetry run mlflow server --host 127.0.0.1 --port 5000
```

Esta modalidad fue utilizada durante las primeras validaciones. Sus registros locales pueden almacenarse en ubicaciones diferentes de las empleadas por los contenedores; no debe asumirse que ambas modalidades muestran el mismo historial de ejecuciones.

### Interfaces

| Servicio | Interfaz web | API de salud |
| --- | --- | --- |
| Prefect | http://127.0.0.1:4200 | http://127.0.0.1:4200/api/health |
| MLflow | http://127.0.0.1:5000 | http://127.0.0.1:5000/health |

## 6. Ejecutar el flujo de prueba

Con **una** de las modalidades anteriores en funcionamiento, abrir **otra terminal PowerShell** en la raíz del repositorio y ejecutar:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
poetry run python flows/prueba_flow.py
```

El archivo [`flows/prueba_flow.py`](flows/prueba_flow.py), adaptado del material de referencia del docente, define el flujo `prueba`. La tarea `sumar` calcula la suma de los números del 1 al 5 y registra en MLflow el parámetro de entrada y el resultado.

**Resultado esperado:**

| Sistema | Comprobación | Resultado |
| --- | --- | --- |
| Prefect | Flujo `prueba` | `Completed` |
| Prefect | Mensaje en logs | `Suma de 1 a 5 = 15` |
| MLflow | Experimento | `prueba-validacion` |
| MLflow | Nombre del run | `prueba-mlflow` |
| MLflow | Parámetro | `n = 5` |
| MLflow | Métrica | `suma = 15` |

Los nombres e identificadores automáticos de los *runs* pueden variar entre ejecuciones. Como comprobación realizada por el equipo, el **24 de octubre de 2026** el flujo `taupe-basilisk` finalizó en `Completed` desde el equipo de Fredy, conectado a Prefect y MLflow desplegados mediante Docker Compose; ambos endpoints de salud devolvieron HTTP 200. Estos resultados **validan la integración de herramientas**, no el desempeño de un modelo de aprendizaje automático.

## 7. Evidencias

Las **nueve capturas finales** se encuentran en [`docs/evidencias/`](docs/evidencias/) y forman parte de la rama `main`. Los nombres se corresponden con las evidencias relacionadas en la **[sección 4 de la Memoria 1](Memoria1.md#4-evidencias)**.

| Evidencia | Archivo |
| --- | --- |
| Repositorio del proyecto en GitHub | [`01_RepositorioGitHub.png`](docs/evidencias/01_RepositorioGitHub.png) |
| Colaboración en GitHub | [`02_colaboradores_git.png`](docs/evidencias/02_colaboradores_git.png) |
| Entorno Poetry de Fredy | [`03_fredy_poetry.png`](docs/evidencias/03_fredy_poetry.png) |
| Entorno Poetry de Liliana | [`04_lili_poetry.png`](docs/evidencias/04_lili_poetry.png) |
| Flujo Prefect ejecutado por Fredy | [`05_fredy_prefect.png`](docs/evidencias/05_fredy_prefect.png) |
| Experimento MLflow de Fredy | [`06_fredy_mlflow.png`](docs/evidencias/06_fredy_mlflow.png) |
| Flujo Prefect ejecutado por Liliana | [`07_lili_prefect.png`](docs/evidencias/07_lili_prefect.png) |
| Experimento MLflow de Liliana | [`08_lili_mlflow.png`](docs/evidencias/08_lili_mlflow.png) |
| Prefect y MLflow mediante Docker Compose | [`09_docker_compose.png`](docs/evidencias/09_docker_compose.png) |

Las capturas de Prefect y MLflow muestran los registros obtenidos en entornos locales. Los enlaces `127.0.0.1` solo funcionan en el computador que aloja cada servidor y **no son accesibles remotamente** para el docente; las imágenes versionadas permiten consultar esas evidencias desde GitHub.

Para consultar la trazabilidad del trabajo colaborativo, pueden revisarse los historiales de las ramas [`main`](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/commits/main), [`fredy/actividad-1`](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/commits/fredy/actividad-1) y [`lili/actividad-1`](https://github.com/fesandovalm-sys/usta-mlops-exportaciones/commits/lili/actividad-1).

## 8. Detener servicios y persistencia

Para detener los servicios de Docker Compose desde la raíz del proyecto:

```powershell
docker compose -f .\services\docker-compose.yml down
```

Este comando conserva los datos persistentes. MLflow utiliza `services/mlflow_store/` y Prefect el volumen de Docker `services_prefect_data`. **No ejecutar `docker compose down -v` si se desea preservar el historial de Prefect.** Ambos almacenamientos se excluyen del control de versiones. En la modalidad local con Poetry se detienen los servidores con `Ctrl + C` en las terminales correspondientes.

## 9. Alcance y siguientes actividades

La Actividad 1 se concentra en establecer una base reproducible y comprobar la integración de Git/GitHub, Poetry, Prefect y MLflow. No incluye carga de datos reales de exportaciones, definición de criterios de anomalía, entrenamiento, evaluación ni despliegue de un clasificador.

En las siguientes actividades se desarrollará progresivamente la detección de valores FOB unitarios potencialmente atípicos como **componente analítico complementario de la tesis**, junto con la definición de fuentes, criterios, variables y métricas pertinentes. El desarrollo del entrenamiento propio fue identificado en la guía del docente como **opcional para la Actividad 1**.

## 10. Referencias

- [Guía oficial de la Actividad 1](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/actividad-1.md).
- [Plantilla de Memoria 1](https://github.com/henrycala/santoto-mlops-content/blob/main/docs/plantilla-memoria.md).
- [Documentación de Poetry](https://python-poetry.org/docs/).
- [Documentación de Prefect](https://docs.prefect.io/).
- [Documentación de MLflow](https://mlflow.org/docs/latest/).
- [Documentación de Docker Compose](https://docs.docker.com/compose/).

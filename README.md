# Actividad 5: progresión de diabetes mediante regresión

## Alcance
Se emplea el dataset público `load_diabetes` de scikit-learn (442 registros, diez variables predictoras, objetivo continuo). **No es clasificación de diagnóstico**: la salida representa una medida cuantitativa de progresión al año.

## Fuente y diccionario
Documentación: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
Referencia original: Efron, Hastie, Johnstone y Tibshirani (2004), Least Angle Regression, Annals of Statistics.
Columnas: `age` edad, `sex` sexo codificado, `bmi` índice de masa corporal, `bp` presión arterial media, `s1` colesterol total, `s2` LDL, `s3` HDL, `s4` razón colesterol total/HDL, `s5` logaritmo de triglicéridos, `s6` glucemia; `target` progresión cuantitativa un año después. Los predictores de la versión scikit-learn ya están escalados en origen. No se interpretan como unidades clínicas originales.

## Reproducción
Desde la raíz `Actividad5`:
```bash
python -m venv .venv
# Active el entorno según su sistema operativo
pip install -r requirements.txt
python fuentes/train.py
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Abrir http://127.0.0.1:5000 para capturar el panel. Se generan `mlruns/`, métricas, modelos y artefactos al ejecutar el script con MLflow instalado. El ZIP incluye resultados comprobados sin MLflow, pero **no contiene capturas auténticas de su panel ni registros MLflow**, pues MLflow no estaba disponible en el entorno de elaboración.

## Decisiones metodológicas
División 80/20 con semilla 42. Validación KFold=5 en entrenamiento; selección por RMSE con GridSearchCV. Se comparan Ridge (regresión lineal regularizada) y Random Forest Regressor. Imputación mediana dentro del pipeline para evitar fuga de información; escalado adicional únicamente en Ridge. Se eliminan duplicados y objetivos ausentes. No se aplican reglas clínicas adicionales. Se reportan MAE, RMSE y R² en prueba y CV. Prueba t pareada solo exploratoria porque los folds no son independientes.

## Versionamiento y GitHub
El dataset original y el limpio están versionados como CSV. Ver CHANGELOG.md y hashes SHA-256 en `resultados/hashes_sha256.txt`. Para publicar:
```bash
git init
git add .
git commit -m "Actividad 5: dataset, modelos y documentación"
git branch -M main
git remote add origin https://github.com/USUARIO/Actividad5.git
git push -u origin main
```
Se debe crear primero un repositorio vacío en la cuenta GitHub. **No se ha publicado un repositorio remoto en nombre del usuario.**

## Ética y limitaciones
Datos históricos pequeños; no aptos para decisiones clínicas. La validación cruzada no sustituye validación externa ni vigilancia de deriva.

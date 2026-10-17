"""Entrenamiento, validación cruzada y registro MLflow."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import ttest_rel
from datos_prep import guardar_limpio

def ejecutar(raiz=None,registrar=True):
    raiz=Path(raiz) if raiz else Path(__file__).resolve().parents[1]
    datos=guardar_limpio(raiz)
    X=datos.drop(columns='target'); y=datos['target']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    modelos={
       'Ridge':(Pipeline([('imputar',SimpleImputer(strategy='median')),('escalar',StandardScaler()),('modelo',Ridge())]),{'modelo__alpha':[.1,1.,10.,100.]}),
       'RandomForest':(Pipeline([('imputar',SimpleImputer(strategy='median')),('modelo',RandomForestRegressor(random_state=42,n_jobs=1))]),{'modelo__n_estimators':[100,200],'modelo__max_depth':[3,6,None],'modelo__min_samples_leaf':[1,3]})}
    if registrar:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri((raiz/'mlruns').resolve().as_uri())
        mlflow.set_experiment('Actividad5_Diabetes_Regresion')
    filas=[]; predicciones={}; scores={}
    for nombre,(pipe,grid) in modelos.items():
        gs=GridSearchCV(pipe,grid,scoring='neg_root_mean_squared_error',cv=cv,n_jobs=1,refit=True)
        gs.fit(Xtr,ytr)
        pred=gs.predict(Xte)
        sc=cross_validate(gs.best_estimator_,Xtr,ytr,cv=cv,scoring={'mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','r2':'r2'})
        scores[nombre]=-sc['test_rmse']
        fila={'modelo':nombre,'MAE_test':mean_absolute_error(yte,pred),'RMSE_test':np.sqrt(mean_squared_error(yte,pred)),'R2_test':r2_score(yte,pred),'MAE_cv':-sc['test_mae'].mean(),'RMSE_cv':-sc['test_rmse'].mean(),'R2_cv':sc['test_r2'].mean(),'RMSE_cv_std':(-sc['test_rmse']).std(ddof=1),'mejores_parametros':json.dumps(gs.best_params_)}
        filas.append(fila);predicciones[nombre]=pred
        if registrar:
            with mlflow.start_run(run_name=nombre):
                mlflow.log_params(gs.best_params_)
                mlflow.log_metrics({k:float(v) for k,v in fila.items() if isinstance(v,(int,float,np.integer,np.floating))})
                mlflow.sklearn.log_model(
    gs.best_estimator_,
    name="modelo",
    skops_trusted_types=[
        "numpy.dtype",
        "sklearn.tree._tree.Tree"
    ]
)
                mlflow.log_artifact(str(raiz/'datos/datos_limp/diabetes_limpio.csv'),artifact_path='datos')
    resultado=raiz/'resultados';resultado.mkdir(exist_ok=True)
    pd.DataFrame(filas).to_csv(resultado/'metricas_modelos.csv',index=False)
    pd.DataFrame({'real':yte.to_numpy(),**predicciones}).to_csv(resultado/'predicciones_test.csv',index=False)
    prueba=ttest_rel(scores['Ridge'],scores['RandomForest'])
    (resultado/'prueba_estadistica.json').write_text(json.dumps({'prueba':'t pareada exploratoria sobre RMSE de 5 folds','estadistico_t':float(prueba.statistic),'valor_p':float(prueba.pvalue),'advertencia':'Los folds no son independientes; interpretar solo exploratoriamente.'},indent=2),encoding='utf-8')
    return pd.DataFrame(filas)

if __name__=='__main__': print(ejecutar().to_string(index=False))

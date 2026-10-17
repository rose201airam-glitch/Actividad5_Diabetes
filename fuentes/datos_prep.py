"""Preparación reproducible del conjunto Diabetes de scikit-learn."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_diabetes

def obtener_datos(raiz):
    raiz=Path(raiz); origen=raiz/'datos/datos_ini/diabetes_original.csv'
    origen.parent.mkdir(parents=True,exist_ok=True)
    if not origen.exists():
        load_diabetes(as_frame=True).frame.to_csv(origen,index=False)
    return pd.read_csv(origen)

def limpiar(df):
    df=df.copy().drop_duplicates().reset_index(drop=True)
    columnas=['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6','target']
    if set(df.columns)!=set(columnas):
        raise ValueError('El esquema de columnas no corresponde al dataset esperado.')
    for c in columnas: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['target'])
    return df

def guardar_limpio(raiz):
    raiz=Path(raiz); df=limpiar(obtener_datos(raiz))
    destino=raiz/'datos/datos_limp/diabetes_limpio.csv'
    destino.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(destino,index=False)
    return df

from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Public benchmark loading and recorded normalization rules.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, time, json, itertools, csv, traceback, functools
import numpy as np
import pandas as pd
import psutil
B = Path(__file__).resolve().parent
ROOT = B.parent
from public_problems import dataset as previous_dataset

@functools.lru_cache(maxsize=32)
def dataset(name, pool_seed=84001):
    if name in ('parkinsons','wine_red','wine_white'):
        new_data=Path(os.environ.get('REGRET_NEW_DATA',MODULE_DIR/'data'))
        if name=='parkinsons':
            df=pd.read_csv(new_data/'parkinsons/parkinsons_updrs.data')
            cols=[c for c in df.columns if c not in ('subject#','motor_UPDRS','total_UPDRS')]
            xx=df[cols].to_numpy(float); yy=df['total_UPDRS'].to_numpy(float); names=cols
        else:
            colour=name.split('_')[1]
            df=pd.read_csv(new_data/f'wine/winequality-{colour}.csv',sep=';')
            xx=df.iloc[:,:-1].to_numpy(float); yy=df.iloc[:,-1].to_numpy(float); names=list(df.columns[:-1])
        if len(xx)>=4096:
            dev=np.sort(np.random.default_rng(88101).choice(len(xx),2048,replace=False))
            ids=dev if pool_seed<88200 else np.sort(np.random.default_rng(pool_seed).choice(np.setdiff1d(np.arange(len(xx)),dev),2048,replace=False))
        else:
            dev=ids=np.arange(len(xx))
        span=np.ptp(xx[dev],axis=0);span=np.where(span>0,span,1.)
        x=np.round((xx[ids]-xx[dev].mean(0))/span*1024)/1024
        y=np.round((yy[ids]-yy[dev].min())/np.ptp(yy[dev])*1024)/1024
        return np.column_stack([np.ones(len(x)),x]),y,['intercept']+list(names)
    if name in ('superconductivity', 'powerplant') and pool_seed >= 84100:
        if name == 'superconductivity':
            df = pd.read_csv(DATA_CORE / 'superconductivity/train.csv')
            xx = df.iloc[:, :-1].to_numpy(float)
            yy = df.iloc[:, -1].to_numpy(float)
            names = list(df.columns[:-1])
            size = 1024
            old = np.sort(np.random.default_rng(74001).choice(len(df), 1024, replace=False))
            dev = np.sort(np.random.default_rng(84001).choice(np.setdiff1d(np.arange(len(df)), old), 1024, replace=False))
            excluded = np.union1d(old, dev)
        else:
            df = pd.read_excel(DATA_EXTRA / 'powerplant/CCPP/Folds5x2_pp.xlsx', sheet_name=0)
            xx = df.iloc[:, :-1].to_numpy(float)
            yy = df.iloc[:, -1].to_numpy(float)
            names = list(df.columns[:-1])
            size = 2048
            dev = np.sort(np.random.default_rng(84001).choice(len(df), size, replace=False))
            excluded = dev
        ids = np.sort(np.random.default_rng(pool_seed).choice(np.setdiff1d(np.arange(len(df)), excluded), size, replace=False))
        assert not np.intersect1d(ids, excluded).size
        mean = xx[dev].mean(0)
        span = np.ptp(xx[dev], axis=0)
        span = np.where(span > 0, span, 1.0)
        ymin = yy[dev].min()
        yscale = np.ptp(yy[dev])
        x = np.round((xx[ids] - mean) / span * 1024) / 1024
        y = np.round((yy[ids] - ymin) / yscale * 1024) / 1024
        return (np.column_stack([np.ones(len(x)), x]), y, ['intercept'] + list(names))
    if name in ('diabetes', 'concrete', 'superconductivity'):
        x, y, pool, names = previous_dataset(name, pool_seed)
        return (x, y, names)
    if name == 'energy':
        df = pd.read_excel(DATA_EXTRA / 'energy/ENB2012_data.xlsx').iloc[:, :10].dropna()
        xx = df.iloc[:, :8].to_numpy(float)
        yy = df.iloc[:, 8].to_numpy(float)
        names = list(map(str, df.columns[:8]))
    elif name == 'airfoil':
        df = pd.read_csv(DATA_EXTRA / 'airfoil/airfoil_self_noise.dat', sep='\\s+', header=None)
        xx = df.iloc[:, :-1].to_numpy(float)
        yy = df.iloc[:, -1].to_numpy(float)
        names = ['frequency', 'angle_of_attack', 'chord_length', 'free_stream_velocity', 'suction_thickness']
    elif name == 'powerplant':
        file = DATA_EXTRA / 'powerplant/CCPP/Folds5x2_pp.xlsx'
        if not file.exists():
            import zipfile
            zfile = DATA_EXTRA / 'powerplant/CCPP.zip'
            with zipfile.ZipFile(zfile) as z:
                payload = z.read('CCPP/Folds5x2_pp.xlsx')
                file.parent.mkdir(exist_ok=True, parents=True)
                file.write_bytes(payload)
                assert file.read_bytes() == payload
        df = pd.read_excel(file, sheet_name=0)
        xx = df.iloc[:, :-1].to_numpy(float)
        yy = df.iloc[:, -1].to_numpy(float)
        names = list(map(str, df.columns[:-1]))
    elif name == 'yacht':
        df = pd.read_csv(DATA_EXTRA / 'yacht/yacht_hydrodynamics.data', sep='\\s+', header=None)
        xx = df.iloc[:, :-1].to_numpy(float)
        yy = df.iloc[:, -1].to_numpy(float)
        names = ['position_buoyancy', 'prismatic_coefficient', 'length_displacement', 'beam_draught', 'length_beam', 'froude_number']
    else:
        raise ValueError(name)
    if name == 'powerplant':
        dev = np.random.default_rng(84001).choice(len(xx), 2048, replace=False)
        if pool_seed == 84001:
            ids = np.sort(dev)
        else:
            ids = np.sort(np.random.default_rng(pool_seed).choice(np.setdiff1d(np.arange(len(xx)), dev), 2048, replace=False))
        xx = xx[ids]
        yy = yy[ids]
    span = np.ptp(xx, axis=0)
    span = np.where(span > 0, span, 1.0)
    x = np.round((xx - xx.mean(0)) / span * 1024) / 1024
    y = np.round((yy - yy.min()) / np.ptp(yy) * 1024) / 1024
    return (np.column_stack([np.ones(len(x)), x]), y, ['intercept'] + names)

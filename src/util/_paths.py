import os

__all__ = ['BASE_PATH', 'PROJ_DIR', 'PKG_NM', 'DSET_DIR', 'MODEL_DIR']

paths = __file__.split(os.sep)
paths = paths[:paths.index('util')]

BASE_PATH = os.sep.join(paths[:-2])
PROJ_DIR = paths[-2]
PKG_NM = paths[-1]  # `src`

MODEL_DIR = 'models'
DSET_DIR = 'data'


if __name__ == '__main__':
    from stefutil.prettier import sic
    sic(BASE_PATH, PROJ_DIR, PKG_NM)

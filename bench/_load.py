"""Import a calibrated model.py (CALIB_DIR/<benchmark>/model.py, default ./calib) by path, without writing bytecode."""
import sys, os, importlib.util
sys.dont_write_bytecode = True
CALIB = os.environ.get('CALIB_DIR') or next((p for p in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'calib')) if os.path.isdir(p)), 'calib')   # remote runs set CALIB_DIR
def load(cid, name='model'):
    d = f'{CALIB}/{cid}'
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(f'calib_{cid}', f'{d}/model.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

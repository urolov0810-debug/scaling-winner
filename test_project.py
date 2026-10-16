import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import app_config


class ConfigTests(unittest.TestCase):
    def test_default_valid(self):
        cfg = app_config.initial_config()
        self.assertEqual(len(app_config.validate(cfg)['layers']), 3)

    def test_round_trip(self):
        cfg = app_config.initial_config()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'params.json'
            app_config.save_json(cfg,path)
            result = app_config.load_json(path)
            self.assertEqual(result['layers'][0]['theta'],30)
            self.assertEqual(result['clpt']['nx'],4)

    def test_negative_foundation(self):
        cfg = app_config.initial_config()
        cfg['kw']=-1
        with self.assertRaises(ValueError):
            app_config.validate(cfg)

    def test_bad_patch(self):
        cfg = app_config.initial_config()
        cfg['load']['patch_width']=2
        with self.assertRaises(ValueError):
            app_config.validate(cfg)

    def test_invalid_layer(self):
        cfg = app_config.initial_config()
        cfg['layers'][0]['E1']=-3
        with self.assertRaises(ValueError):
            app_config.validate(cfg)


class WorkerIntegrationTests(unittest.TestCase):
    def test_both_models(self):
        # one elastic isotropic ply to keep CI runs short and deterministic
        lay={'name':'Aluminum','h':0.008,'theta':0,'E1':70e9,'E2':70e9,
             'G12':70e9/2.6,'G13':70e9/2.6,'G23':70e9/2.6,
             'nu12':0.3,'rho':2700,'is_visco':False}
        cfg={'a':0.6,'b':0.5,'nx':4,'ny':4,
             'clpt':{'nx':2,'ny':2},
             'kw':1e7,'kp':1e5,
             'edges':{key:'simply_supported' for key in ('left','right','bottom','top')},
             'layers':[lay],
             'load':{'kind':'pulse','spatial':'patch','amplitude':10,
                     'duration':0.001,'x':0.3,'y':0.25,'patch_width':0.06,
                     'patch_height':0.05},
             'simulation':{'total_time':0.001,'dt':0.0001,'calculate_stress':False}}
        with tempfile.TemporaryDirectory() as folder:
            config_path=Path(folder)/'config.json'
            out=Path(folder)/'results'
            app_config.save_json(cfg,config_path)
            env=os.environ.copy();env['MPLBACKEND']='Agg'
            command=[sys.executable, str(PROJECT/'main.py'),
                     '--worker','--config',str(config_path), '--output',str(out),
                     '--model','both']
            r=subprocess.run(command,cwd=PROJECT,env=env,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,text=True,timeout=90)
            self.assertEqual(r.returncode,0,msg=r.stdout)
            self.assertIn('YAKUNIY HOLAT: OK',r.stdout)
            for path in ('fsdt/summary.json','fsdt/response.png',
                         'clpt/summary.json','clpt/response.png',
                         'comparison.json','comparison.png','comparison.csv'):
                self.assertTrue((out/path).exists(),path)
            data=json.loads((out/'comparison.json').read_text())
            self.assertGreater(data['fsdt_frequencies_hz'][0],0)

if __name__ == '__main__':
    unittest.main()

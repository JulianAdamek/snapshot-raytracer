#!/usr/bin/env python3
"""Controls for frozen high-resolution fixtures and production orchestration."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ode_comparison as exercise

TOOL = exercise.ROOT/'validation/validation-tools'
CONFIG = exercise.cosmology({'h':'0.67','Omega_b':'0.049','Omega_cdm':'0.268580851704',
    'm_ncdm':'0.00868907,0.05','deg_ncdm':'1,1','boxsize':'1440','Ngrid':'2400'})


class ODEComparisonTests(unittest.TestCase):
    def test_background_knots_and_wide_shells(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'info.bin'
            support = exercise.static_info(path,CONFIG,0.5)
            rows = list(struct.iter_unpack('=i6d',path.read_bytes()))
            self.assertEqual(len(rows),support['records'])
            self.assertTrue(all(b[0] == a[0]+1 and b[1] > a[1] and b[2] > a[2] for a,b in zip(rows,rows[1:])))
            self.assertEqual([r[2] for r in rows if r[1] == support['observer_tau']],[1.0])
            self.assertGreaterEqual(support['extra_lookback_cells'],64-1e-8)
            self.assertLess(support['extra_lookback_cells'],66+1e-8)
            self.assertGreater(1/rows[0][2]-1,0.5)
            self.assertTrue(all(r[3:5] == r[5:7] == tuple(support['shell_boundaries']) for r in rows))
            # Independent quadrature maps each tabulated scale factor back to
            # conformal time. Test off-observer knots as well as the far endpoint.
            for row in rows[::127]:
                z = 1/row[2]-1
                self.assertAlmostEqual((support['observer_tau']-row[1])*CONFIG['ngrid'],
                                       exercise.distance_cells(z,CONFIG),delta=2e-8)

    def test_shared_settings_and_variant_mapping(self):
        support={'observer_tau':6.7,'lookback_cells':2250,'vertex':[0.5]*3}
        manifest={'synthetic':support,'target_z':0.5,'slab_cells':16,'padding_cells':16,
                  'redshift_tolerance':1e-8}
        work=Path('/fixture')
        for face in range(12):
            a=exercise.command(work,manifest,'corrected',face,16384)
            b=exercise.command(work,manifest,'legacy',face,16384)
            self.assertEqual(a[1:],b[1:])
            self.assertEqual(a[a.index('--batch')+1],face)
            self.assertEqual(a[a.index('--time-derivatives')+1],'zero')
            self.assertEqual(a[a.index('--thickness')+1],1)
        assignments=[(exercise.VARIANTS[g//3],(g%3)*4+r) for g in range(6) for r in range(4)]
        self.assertEqual(set(assignments),{(v,f) for v in exercise.VARIANTS for f in range(12)})
        self.assertEqual(len(assignments),24)

    def test_metric_failure_controls(self):
        manifest={'synthetic':{'lookback_cells':2250},
                  'variants':{'corrected':dict.fromkeys(exercise.PHYSICS_KEYS,1)}}
        good={'physics':manifest['variants']['corrected'],'time_derivatives':'zero',
              'storage':'ziggurat','ordering':'NESTED','scalar_source':'phi',
              'redshift_tolerance':1e-8,'effective_lookback_cells':2250,'residency_misses':0,
              'ray_status_counts':{'success':256**2,'missing time bracket':0}}
        exercise.check_metrics(good,manifest,'corrected',256)
        for field,value in [('physics',dict.fromkeys(exercise.PHYSICS_KEYS,0)),
                            ('residency_misses',1),('effective_lookback_cells',2000),
                            ('time_derivatives','finite-difference'),
                            ('ray_status_counts',{'success':256**2-1,'missing time bracket':1})]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                exercise.check_metrics({**good,field:value},manifest,'corrected',256)
        with patch.dict(os.environ,{},clear=True), self.assertRaises(RuntimeError):
            exercise.require_slurm()

    def test_streaming_face_checks(self):
        for fault,valid,face in [('batch',True,7),('batch-ring',False,7),
                                 ('isw-nonzero',False,0),('isw-nan',False,0)]:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'control'
                subprocess.run([str(TOOL),'self-test',str(root),fault],check=True,capture_output=True)
                result=subprocess.run([str(TOOL),'check-face',str(root/'lightcone'),'4',str(face),'0.1'],
                                      capture_output=True,text=True)
                self.assertEqual(result.returncode == 0,valid,result.stderr)
                if valid:
                    self.assertTrue(json.loads(result.stdout)['pass'])
                    # A missing product cannot be mistaken for a successful ray run.
                    (root/'lightcone'/f'iswmap_batch{face}.fits').unlink()
                    missing=subprocess.run([str(TOOL),'check-face',str(root/'lightcone'),'4',str(face),'0.1'],
                                           capture_output=True,text=True)
                    self.assertNotEqual(missing.returncode,0)


if __name__ == '__main__':
    unittest.main()

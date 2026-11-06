#!/usr/bin/env python3
"""CPU controls for fixture generation and the numerical comparison reporter."""
import json
import contextlib
import io
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import run_static

from run_static import cosmology, synthetic_info

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT/'validation'/'validation-tools'


class WorkflowTests(unittest.TestCase):
    def test_fixture_and_comparator_controls(self):
        for fault in ['', 'difference', 'nan', 'batch', 'batch-ring', 'outside', 'strict', 'strict-difference', 'isw-nonzero', 'isw-nan']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                command = [str(TOOL), 'self-test', str(Path(directory)/'control')]
                if fault:
                    command.append(fault)
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                report = json.loads(result.stdout)
                self.assertEqual(report['pass'], fault not in ['difference','nan','outside','strict-difference','isw-nonzero','isw-nan'])
                self.assertEqual(len(report['maps']), 8)
                # Full sky is logical batch -1, but both tracers use batch0 in filenames.
                batch = 7 if fault in ['batch','batch-ring','outside'] else -1
                self.assertEqual(report['batch'], batch)
                self.assertEqual(report['canonical_ordering'], 'NESTED')
                for producer in ['snapshot', 'lightcone']:
                    suffix = 7 if producer == 'lightcone' and batch == 7 else 0
                    maps = list((Path(directory)/'control'/producer).glob(f'*_batch{suffix}.fits'))
                    self.assertEqual(len(maps), 7 if producer == 'snapshot' and fault not in ['strict','strict-difference'] else 8)
                    self.assertFalse(list((Path(directory)/'control'/producer).glob('*_batch-1.fits')))
                if fault == '':
                    strict = subprocess.run([str(TOOL),'compare-frozen',str(Path(directory)/'control/snapshot'),
                        str(Path(directory)/'control/lightcone'),'4','-1','0.1'],capture_output=True,text=True,check=True)
                    self.assertTrue(json.loads(strict.stdout)['pass'])
                if fault == 'isw-nonzero':
                    self.assertEqual(report['maps'][7]['nonzero_isw_pixels'],1)
                if fault == 'isw-nan':
                    self.assertEqual(report['maps'][7]['invalid_pixels'],1)
                if fault == 'nan':
                    self.assertEqual(report['maps'][0]['invalid_pixels'], 1)
                    self.assertEqual(report['maps'][0]['mask_mismatches'], 1)
                if fault == 'outside':
                    self.assertEqual(report['maps'][0]['outside_batch_nonzero'], 1)
                if fault == 'strict-difference':
                    self.assertGreater(report['maps'][0]['rounding_failures'], 0)
                if fault == 'difference':
                    self.assertGreater(report['maps'][0]['max_abs_difference'], report['maps'][0]['absolute_tolerance'])

    def test_storage_cli_errors(self):
        base = [str(ROOT/'lightcone-raytracer'),'lc','16','--info','unused','--thickness','3',
                '--vertex','0.5','0.5','0.5','--observer-tau','1','--target-z','0.1']
        cases = [(['--storage','ziggurat'],'requires --batch'),
                 (['--slab-cells','16'],'Slab-specific options'),
                 (['--padding-cells','0'],'Slab-specific options'),
                 (['--batch','0','--storage','ziggurat','--padding-cells','-1'],'Invalid slab thickness or padding'),
                 (['--max-lookback-cells','0'],'Look-back cap must be positive'),
                 (['--ordering','unknown'],'Ordering must be nested or ring'),
                 (['--time-derivatives','invalid'],'Time derivatives must be finite-difference or zero')]
        for args, message in cases:
            with self.subTest(args=args):
                result = subprocess.run(base+args,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(message,result.stderr)
        command = base.copy(); command[command.index('--thickness')+1] = '1'
        result = subprocess.run(command,capture_output=True,text=True)
        self.assertIn('Thickness 1 requires explicit',result.stderr)
        result = subprocess.run(command+['--time-derivatives','zero'],capture_output=True,text=True)
        self.assertIn('Cannot open info file',result.stderr)
        command = base.copy(); command[2] = '3'
        result = subprocess.run(command,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('power of two',result.stderr)

    def test_twelve_face_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([str(TOOL),'assembly-self-test',str(Path(directory)/'good')],
                                    capture_output=True,text=True,check=True)
            report = json.loads(result.stdout)
            self.assertTrue(report['pass'])
            self.assertTrue(report['strict'])
            self.assertTrue(all(m['max_abs_difference'] == 0 for m in report['maps']))
            # The generic assembly control has a deliberately nonzero ISW field.
            # Frozen validation must reject it even when reference and candidate agree.
            zero_check = subprocess.run([str(TOOL),'compare-strict',str(Path(directory)/'good/reference'),
                str(Path(directory)/'good/faces/assembled'),'2','-1','0.1','--require-zero-isw'],
                capture_output=True,text=True,check=True)
            zero_report = json.loads(zero_check.stdout)
            self.assertFalse(zero_report['pass'])
            self.assertGreater(zero_report['maps'][7]['nonzero_isw_pixels'],0)
            result = subprocess.run([str(TOOL),'assembly-self-test',str(Path(directory)/'bad'),'outside'],
                                    capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('outside declared base face',result.stderr)

    def test_driver_uses_full_references_for_true_faces(self):
        self.exercise_driver(False)
        self.exercise_driver(True)

    def exercise_driver(self, use_baseline):
        # Exercise orchestration without launching scientific executables. Numerical
        # FITS behavior is independently exercised by the real C++ controls above.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root/'settings.ini'
            settings.write_text('h=0.67\nomega_b=0.021996\nomega_cdm=0.121203\nm_ncdm=0.00868907,0.05\nboxsize=4032\nNgrid=576\nsnapshot file base=test_snap\nsnapshot redshifts=0\n')
            for suffix in ['phi','B']:
                (root/f'test_snap000_{suffix}.h5').touch()
            commands, orderings = [], {}

            def execute(command, cwd, logfile, timeout=None):
                commands.append(list(map(str,command)))
                if Path(command[0]).name == 'lightcone-raytracer':
                    options = list(map(str,command))
                    storage = options[options.index('--storage')+1]
                    ordering = options[options.index('--ordering')+1].upper()
                    orderings[str(cwd)] = ordering
                    run_static.write_json(Path(cwd)/'run-metrics.json', {'storage':storage,'residency_misses':0,'time_derivatives':options[options.index('--time-derivatives')+1],
                        'static_shear_transport':'negative'})
                return {'wall_seconds':0,'max_host_rss_kib':0}

            def compare(command, cwd):
                commands.append(list(map(str,command)))
                if command[1] == 'assemble':
                    batch, target, ordering = -1, float(command[5]), 'NESTED'
                else:
                    batch, target = int(command[5]), float(command[6])
                    ordering = orderings[str(command[3])]
                return {'pass':True,'batch':batch,'target_z':target,
                        'maps':[{'candidate_ordering':ordering,'pass':True} for _ in range(8)]}

            baseline = root/'baseline'; old_full = baseline/'references/z0.15/lightcone'
            old_full.mkdir(parents=True); (old_full/'zmap_batch0.fits').touch()
            run_static.write_json(baseline/'manifest.json',{'nside':16,'snapshot_index':0,'static_shear_transport':'negative',
                'configuration':cosmology(run_static.settings_file(settings)),'targets':[0.15]})
            args = ['run_static.py','--allow-local','--validate-isw','--data-dir',str(root),
                    '--settings',str(settings),'--snapshot','0','--output-root',str(root/'results')]
            if use_baseline:
                args += ['--baseline-run',str(baseline)]
            with patch('sys.argv',args), patch.object(run_static,'run_logged',execute), patch.object(run_static,'compare_logged',compare), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_static.main(),0)
            snapshot_runs = [c for c in commands if Path(c[0]).name == 'snapshot-raytracer']
            self.assertEqual(len(snapshot_runs),1)
            self.assertNotIn('--batch',snapshot_runs[0])
            cones = [c for c in commands if Path(c[0]).name == 'lightcone-raytracer']
            self.assertEqual(len(cones),28)  # both modes: one full reference, twelve slabs, one RING check
            for mode in ['finite-difference','zero']:
                selected = [c for c in cones if c[c.index('--time-derivatives')+1] == mode]
                self.assertEqual(len(selected),14)
                faces = [c for c in selected if '--batch' in c and c[c.index('--ordering')+1] == 'nested']
                self.assertEqual(sorted(int(c[c.index('--batch')+1]) for c in faces),list(range(12)))
                self.assertTrue(all(c[c.index('--storage')+1] == 'ziggurat' for c in faces))
            self.assertEqual(len([c for c in commands if len(c)>1 and c[1]=='assemble']),2)
            self.assertEqual(len([c for c in commands if len(c)>1 and c[1]=='compare-frozen']),2 if use_baseline else 0)
            if use_baseline:
                saved = json.loads((baseline/'manifest.json').read_text())
                del saved['static_shear_transport']
                run_static.write_json(baseline/'manifest.json',saved)
                with patch('sys.argv',args), patch.object(run_static,'run_logged') as execute_mock, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        run_static.main()
                    execute_mock.assert_not_called()

    def test_synthetic_background_and_stencil_support(self):
        config = cosmology({'h': '0.67', 'omega_b': '0.021996', 'omega_cdm': '0.121203',
                            'm_ncdm': '0.00868907,0.05', 'boxsize': '4032', 'Ngrid': '288'})
        self.assertAlmostEqual(config['omega_m'], 0.3204034678236443)
        self.assertAlmostEqual(config['box_mpc']*config['h'], 4032)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'info.bin'
            support = synthetic_info(path, config, 0.3)
            raw = path.read_bytes()
            self.assertEqual(len(raw), 68*support['records'])
            rows = list(struct.iter_unpack('=i8d', raw))
            observer = [row for row in rows if row[1] == support['observer_tau']]
            self.assertEqual(len(observer), 1)
            self.assertEqual(observer[0][2], 1.0)
            subprocess.run([str(TOOL), 'check-fixture', str(path), '288',
                            str(support['observer_tau']), str(support['max_distance_cells'])], check=True)


if __name__ == '__main__':
    unittest.main()

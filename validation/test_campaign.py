#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import campaign as c


def manifest():
    return {'nside':16,'pilot_nside':4,'configuration':{'ngrid':32},'inputs':{'phi':{'path':'phi','bytes':1,'mtime_ns':1}},
            'snapshot_z':.25,'target_z':.5,'ordering':'NESTED','time_derivatives':'zero','scalar_source':'phi',
            'b_layout':'staggered','shell_geometry':'legacy','redshift_tolerance':1e-8,'info_sha256':'a'*64,
            'slab_cells':3,'padding_cells':8,'synthetic':{'observer_tau':1,'vertex':[.5]*3,'lookback_cells':10},
            'variants':{v:dict.fromkeys(c.old.PHYSICS_KEYS,int(v=='corrected')) for v in c.old.VARIANTS}}


def metrics(m,t):
    return {'physics':m['variants'][t['variant']],'time_derivatives':'zero','storage':'ziggurat','ordering':'NESTED',
            'scalar_source':'phi','redshift_tolerance':m['redshift_tolerance'],'nside':t['nside'],'face':t['face'],
            'first_pixel':t['first'],'pixel_count':t['count'],'subdivision_level':t['level'],'part':t['part'],
            'run_id':t['run_id'],'stage':'complete','output_format':'compact-hdf5','residency_misses':0,
            'ray_status_counts':{'success':t['count'],'nonresident storage':0},'effective_lookback_cells':10,
            'io_mode_requested':t.get('io_mode','auto'),
            'io_mode':'tiles' if t.get('io_mode')=='tiles' or (t.get('io_mode','auto')=='auto' and t['face'] in (4,6)) else 'regions'}


class CampaignTests(unittest.TestCase):
    def test_ranges_and_science_identity(self):
        m=manifest();cfg={'level':1,'probe_nside':8,'io_mode':'auto'}
        ts=c.phase_tasks(cfg,m,'production')
        self.assertEqual(len(ts),96)
        for variant in c.old.VARIANTS:
            rows=[t for t in ts if t['variant']==variant]
            self.assertEqual([t['first'] for t in rows],list(range(0,12*16**2,16**2//4)))
        changed=json.loads(json.dumps(m));changed['padding_cells']=99
        self.assertEqual(c.science_id(m,'corrected',16),c.science_id(changed,'corrected',16))
        for key,value in [('redshift_tolerance',1e-4),('target_z',.3),('info_sha256','b'*64)]:
            other={**m,key:value};self.assertNotEqual(c.science_id(m,'corrected',16),c.science_id(other,'corrected',16))
        self.assertNotEqual(c.science_id(m,'corrected',16),c.science_id(m,'legacy',16))

    def test_parent_reuse_and_exact_coverage(self):
        m=manifest();cfg={'level':1,'probe_nside':8,'io_mode':'auto'}
        records=[{'pass':True,'task':t} for t in c.phase_tasks(cfg,m,'pilot')]
        self.assertTrue(c.full_coverage(records,4))
        parent={'pass':True,'task':c.interval(m,'corrected',4,0,0,0)}
        records.append(parent)
        self.assertEqual(len(c.canonical(records,'corrected')),45)
        self.assertIs(c.covered(c.interval(m,'corrected',4,0,1,2),[parent]),parent)
        self.assertTrue(c.full_coverage(records,4))
        records=[r for r in records if not(r['task']['variant']=='legacy' and r['task']['face']==11)]
        self.assertFalse(c.full_coverage(records,4))
        # The saved Nside=4 pilot can never satisfy a production interval.
        self.assertIsNone(c.covered(c.interval(m,'corrected',16,0,1,0),[parent]))

    def test_artifact_and_metric_failures(self):
        m=manifest();t=c.interval(m,'corrected',4,0,1,0);v=metrics(m,t)
        c.check_metrics(v,m,t)
        for key,value in [('stage','writing'),('run_id','b'*64),('residency_misses',1),
                          ('physics',m['variants']['legacy']),('ray_status_counts',{'success':3,'nonresident storage':1})]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):c.check_metrics({**v,key:value},m,t)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'part';p.write_bytes(b'hello')
            r={'pass':True,'task':t,'artifacts':[c.artifact(p)],'part_file':str(p)}
            c.verified(r,m)
            p.write_bytes(b'different')
            with self.assertRaises(RuntimeError):c.verified(r,m)
            p.unlink()
            with self.assertRaises(FileNotFoundError):c.verified(r,m)

    def test_compact_header_binding(self):
        m=manifest();task=c.interval(m,'corrected',4,0,1,0)
        header={key:task[key] for key in ('nside','first','count','face','level','part','run_id')}
        header.update(physics=m['variants']['corrected']);header['pass']=True
        c.check_part_metadata(header,task,m)
        for key,value in [('first',4),('run_id','b'*64),('nside',16),('physics',m['variants']['legacy'])]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                c.check_part_metadata({**header,key:value},task,m)

    def test_claim_exclusion_and_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'SLURM_JOB_ID':'42'}):
            p=Path(d)
            with c.claim(p,False):
                with self.assertRaises(RuntimeError):
                    with c.claim(p,True):pass
            self.assertFalse((p/'claim.json').exists())
            c.atomic(p/'claim.json',{'job':'41'})
            with patch.object(c,'terminal_job',return_value=False),self.assertRaises(RuntimeError):
                with c.claim(p,True):pass
            with patch.object(c,'terminal_job',return_value=True):
                with c.claim(p,True):self.assertEqual(len(list(p.glob('claim-interrupted-*'))),1)
            (p/'attempt-old').mkdir()
            with self.assertRaises(RuntimeError):
                with c.claim(p,False):pass
            with c.claim(p,True):pass

    def test_probe_resource_gates(self):
        m=manifest();m['nside']=16384;cfg={'level':1,'probe_nside':1024,'io_mode':'auto'}
        records=[]
        for t in c.phase_tasks(cfg,m,'probe'):
            v={'field_bytes':20*2**30,'device_bytes':20*2**30+t['count']*52+100000,
               'device_total_bytes':96*2**30,'device_free_bytes':95*2**30,'tracing_seconds':30,'loading_seconds':60}
            records.append({'task':t,'metrics':v})
        with patch.object(c,'completions',return_value=records):
            self.assertTrue(c.report(Path('/unused'),cfg,m,'probe')['pass'])
            records[0]['metrics']['field_bytes']=95*2**30
            self.assertFalse(c.report(Path('/unused'),cfg,m,'probe')['pass'])
            records[0]['metrics']['field_bytes']=20*2**30;records[0]['metrics']['tracing_seconds']=10000
            self.assertFalse(c.report(Path('/unused'),cfg,m,'probe')['pass'])

    def test_run_resume_and_changed_task_table(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'SLURM_JOB_ID':'42','SLURM_PROCID':'0','SLURM_ARRAY_TASK_ID':'0'}):
            root=Path(d);m=manifest();cfg={'original':str(root/'old'),'io_mode':'auto','io_buffer_mib':256,
                                           'binary_sha256':{'lightcone-corrected':'a'*64}}
            c.atomic(root/'campaign.json',cfg)
            task=c.interval(m,'corrected',4,4,1,0);task['io_mode']='auto'
            table=root/'table.json';c.atomic(table,{'phase':'pilot','campaign_sha256':c.old.sha256(root/'campaign.json'),'tasks':[task]})
            c.atomic(table.with_suffix('.sha256'),{'sha256':c.old.sha256(table)})
            args=argparse.Namespace(campaign=root,table=table,resume=True)
            calls=[]
            def execute(command,directory,logfile,timeout=None):
                calls.append(command)
                if Path(command[0]).name=='lightcone-corrected':
                    c.atomic(directory/'run-metrics.json',metrics(m,task));(directory/'ray-part.h5').write_bytes(b'fake-artifact')
                elif command[1]=='check-part':
                    header={key:task[key] for key in ('nside','first','count','face','level','part','run_id')}
                    header.update(physics=m['variants']['corrected']);header['pass']=True
                    c.atomic(logfile,header)
                return {'wall_seconds':1,'max_host_rss_kib':1}
            header={key:task[key] for key in ('nside','first','count','face','level','part','run_id')}
            header.update(physics=m['variants']['corrected']);header['pass']=True
            with patch.object(c,'load',return_value=(cfg,m)),patch.object(c,'prerequisites'),patch.object(c,'run_logged',execute),\
                 patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout=json.dumps(header))),contextlib.redirect_stdout(io.StringIO()):
                c.run(args);self.assertEqual(len(calls),2)
                c.run(args);self.assertEqual(len(calls),2) # Fully verified skip; no retracing.
                saved=c.read(table);saved['tasks'][0]['part']=2;c.atomic(table,saved)
                with self.assertRaises(RuntimeError):c.run(args)

    def test_new_cli_combinations(self):
        base=[str(c.ROOT/'lightcone-raytracer'),'lc','16','--info','unused','--thickness','1',
              '--vertex','.5','.5','.5','--observer-tau','1','--target-z','.1','--time-derivatives','zero','--batch','4']
        cases=[(['--subdivision-level','1'],'Subdivision requires compact-hdf5'),
               (['--part','1'],'Part is outside'),(['--io-mode','tiles'],'Tiled I/O requires ziggurat'),
               (['--io-buffer-mib','0'],'Invalid I/O buffer'),(['--output-format','compact-hdf5'],'Compact output requires'),
               (['--subdivision-level','5'],'level <= log2'),(['--io-mode','bad'],'I/O mode must')]
        for extra,text in cases:
            with self.subTest(extra=extra):
                r=subprocess.run(base+extra,capture_output=True,text=True)
                self.assertNotEqual(r.returncode,0);self.assertIn(text,r.stderr)


if __name__=='__main__':unittest.main()

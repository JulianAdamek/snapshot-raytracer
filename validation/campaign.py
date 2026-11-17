#!/usr/bin/env python3
"""Resumable tiled-preload campaign; scientific I/O runs exclusively under Slurm."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import ode_comparison as old
from run_static import run_logged

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = old.DEFAULT_WORK/'campaigns/tiled-v1'
PHASES = ('diagnostic','pilot','probe','production')
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    with tmp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    tmp.replace(path)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+f'-{time.time_ns()}'


def science_id(manifest, variant, nside):
    # Storage, subdivision and executable identity do not change the science.
    settings={k:manifest[k] for k in ('configuration','inputs','snapshot_z','target_z','ordering',
        'time_derivatives','scalar_source','b_layout','shell_geometry','redshift_tolerance','synthetic')}
    settings.update(nside=nside,physics=manifest['variants'][variant],info_sha256=manifest['info_sha256'],
                    prepared_inputs=manifest.get('prepared_inputs',[]))
    return hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def init(args):
    original=args.work_dir.resolve(); manifest=old.load(original)
    if args.level<1 or args.level>int(math.log2(min(manifest['nside'],manifest['pilot_nside']))):
        raise ValueError('Subdivision level must be positive and supported at pilot resolution')
    root=args.campaign.resolve(); root.mkdir(parents=True,exist_ok=False)
    needed=math.ceil(1.10*(2*12*manifest['nside']**2*68))
    if shutil.disk_usage(root).free<needed: raise RuntimeError(f'Need {needed} free bytes for compact intermediates and final maps')
    (root/'bin').mkdir()
    binaries={}
    if manifest['synthetic']['thickness']!=1: raise RuntimeError('This frozen campaign requires the prepared thickness-1 fixture')
    prepared_inputs=[{'logical':str(original/'fixture'/name),'identity':old.identity(original/'fixture'/name)}
                     for name in ('lc_phi.h5','lc_B1.h5','lc_B2.h5','lc_B3.h5')]
    sources={name:ROOT/'validation/bin'/name for name in ('lightcone-corrected','lightcone-legacy','validation-tools')}
    for name in ('lightcone-tests','lightcone-storage-tests','lightcone-time-tests','lightcone-reference-tests',
                 'lightcone-geometry-tests','lightcone-geometry-gpu','lightcone-bundle-tests','lightcone-preload-tests',
                 'lightcone-preload-gpu','lightcone-preload-gpu-legacy'):
        sources[name]=ROOT/'tests'/name
    for name in ('0000','00','01','10','11'): sources['variant-'+name]=ROOT/'.build'/('variant-'+name)
    for name,path in sources.items():
        shutil.copy2(path,root/'bin'/name);binaries[name]=old.sha256(root/'bin'/name)
    shutil.copy2(ROOT/'validation/bin/build.json',root/'build.json')
    (root/'tables').mkdir()
    config={'schema':1,'created_utc':stamp(),'original':str(original),
            'workflow_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'manifest_sha256':old.sha256(original/'manifest.json'),
            'prepared_inputs':prepared_inputs,'level':args.level,'io_mode':'auto','io_buffer_mib':256,'binary_sha256':binaries,
            'python_sha256':{name:old.sha256(ROOT/'validation'/name) for name in ('campaign.py','ode_comparison.py','run_static.py')},
            'probe_nside':1024,'final_ordering':'NESTED'}
    atomic(root/'campaign.json',config)
    print('Campaign prepared:',root,'\nNo inputs were rewritten and no jobs were submitted.')


def load(root):
    config=read(root/'campaign.json'); original=Path(config['original'])
    if old.sha256(original/'manifest.json')!=config['manifest_sha256']:
        raise RuntimeError('Original prepared manifest changed')
    manifest=old.load(original)
    for item in config['prepared_inputs']:
        if old.identity(item['logical'])!=item['identity']: raise RuntimeError('Prepared component changed: '+item['logical'])
    manifest['prepared_inputs']=config['prepared_inputs']
    for name,digest in config['binary_sha256'].items():
        if old.sha256(root/'bin'/name)!=digest: raise RuntimeError('Campaign binary changed: '+name)
    for name,digest in config['python_sha256'].items():
        if old.sha256(ROOT/'validation'/name)!=digest: raise RuntimeError('Campaign Python changed; initialize a new campaign: '+name)
    return config,manifest


def interval(manifest, variant, nside, face, level, part):
    if not 0<=face<12 or not 0<=level<=int(math.log2(nside)) or not 0<=part<4**level:
        raise ValueError('Invalid NESTED task')
    count=nside*nside//4**level
    return {'variant':variant,'nside':nside,'face':face,'level':level,'part':part,
            'first':face*nside*nside+part*count,'count':count,'run_id':science_id(manifest,variant,nside)}


def task_key(t):
    return f"{t['variant']}/nside-{t['nside']}/face-{t['face']:02d}/level-{t['level']}-part-{t['part']}"


def check_part_metadata(info, task, manifest):
    expected={key:task[key] for key in ('nside','first','count','face','level','part','run_id')}
    expected['physics']=manifest['variants'][task['variant']]
    expected['pass']=True
    if any(info.get(key)!=value for key,value in expected.items()):
        raise RuntimeError('Compact header does not match its completion/task')


def verified(record, manifest, deep=False, tool=None):
    if not record.get('pass'): raise RuntimeError('Invalid completion marker')
    t=record['task']; expected=interval(manifest,t['variant'],t['nside'],t['face'],t['level'],t['part'])
    if any(t.get(k)!=v for k,v in expected.items()): raise RuntimeError('Completion scientific configuration mismatch')
    part_path=str(Path(record['part_file']).resolve(strict=True))
    if not any(item['identity']['path']==part_path for item in record['artifacts']):
        raise RuntimeError('Compact file is not bound to a recorded checksum')
    for artifact in record['artifacts']:
        path=Path(artifact['path'])
        if old.identity(path)!=artifact['identity']: raise RuntimeError('Completed artifact changed: '+str(path))
        if deep and old.sha256(path)!=artifact['sha256']: raise RuntimeError('Completed artifact checksum mismatch: '+str(path))
    if deep:
        old.require_slurm()
        result=subprocess.run([str(tool),'check-part',record['part_file']],check=True,capture_output=True,text=True)
        check_part_metadata(json.loads(result.stdout),t,manifest)
    return record


def completions(root, manifest, nside, phases=None):
    result=[]
    for phase in phases or ('imports','diagnostic','pilot','probe','production'):
        for path in (root/phase).glob('**/success.json'):
            r=verified(read(path),manifest)
            if r['task']['nside']==nside: result.append(r)
    return result


def covered(task, records):
    return next((r for r in records if r['task']['variant']==task['variant'] and r['task']['run_id']==task['run_id'] and
        r['task']['first']<=task['first'] and r['task']['first']+r['task']['count']>=task['first']+task['count']),None)


def canonical(records,variant):
    # Prefer validated parent intervals, then add disjoint children. Partial
    # overlaps cannot occur for aligned NESTED regions and are rejected anyway.
    selected=[]
    for r in sorted((r for r in records if r['task']['variant']==variant),key=lambda r:(-r['task']['count'],r['task']['first'])):
        t=r['task']
        if covered(t,selected): continue
        for other in selected:
            u=other['task']
            if max(t['first'],u['first'])<min(t['first']+t['count'],u['first']+u['count']):
                raise RuntimeError('Inconsistent overlapping completed intervals')
        selected.append(r)
    return sorted(selected,key=lambda r:r['task']['first'])


def full_coverage(records,nside):
    for variant in old.VARIANTS:
        next_pixel=0
        for r in canonical(records,variant):
            t=r['task']
            if t['first']!=next_pixel: return False
            next_pixel+=t['count']
        if next_pixel!=12*nside*nside: return False
    return True


def phase_tasks(config,manifest,phase):
    if phase=='diagnostic':
        return [dict(interval(manifest,'corrected',manifest['pilot_nside'],f,0,0),io_mode='tiles') for f in (4,6,0,5)]
    if phase=='probe':
        return [dict(interval(manifest,v,config['probe_nside'],f,config['level'],0),io_mode='auto') for v in old.VARIANTS for f in (0,4)]
    nside=manifest['pilot_nside'] if phase=='pilot' else manifest['nside']
    return [dict(interval(manifest,v,nside,f,config['level'],p),io_mode=config['io_mode'])
            for v in old.VARIANTS for f in range(12) for p in range(4**config['level'])]


def report(root,config,manifest,phase):
    if phase=='pilot':
        records=completions(root,manifest,manifest['pilot_nside'],('imports','diagnostic','pilot'))
        passed=full_coverage(records,manifest['pilot_nside'])
    else:
        nside=config['probe_nside'] if phase=='probe' else manifest['pilot_nside'] if phase=='diagnostic' else manifest['nside']
        records=completions(root,manifest,nside,(phase,))
        passed=all(covered(t,records) for t in phase_tasks(config,manifest,phase))
    result={'phase':phase,'pass':passed,'completed_records':len(records)}
    if phase=='probe':
        estimates=[]
        for r in records:
            t=r['task'];m=r['metrics']; factor=(manifest['nside']/t['nside'])**2
            required=math.ceil(m['field_bytes']*1.02)+(manifest['nside']**2//4**config['level'])*52
            # Retain all auxiliary allocations in addition to field and ray buffers.
            required+=max(0,m['device_bytes']-m['field_bytes']-t['count']*52)
            reserve=max(2*2**30,math.ceil(.05*m['device_total_bytes']))
            seconds=1.25*m['tracing_seconds']*factor+m['loading_seconds']+3600
            estimates.append({'variant':t['variant'],'face':t['face'],'device_bytes':required,'free_bytes':m['device_free_bytes'],
                              'reserve_bytes':reserve,'projected_hours':seconds/3600,
                              'pass':required+reserve<=m['device_free_bytes'] and seconds<=20*3600})
        result.update(estimates=estimates)
        result['pass']=passed and all(e['pass'] for e in estimates)
    if phase=='production': result['pass']=passed and full_coverage(records,manifest['nside'])
    return result


def prerequisites(root,config,manifest,phase):
    if phase=='production':
        for required in ('diagnostic','pilot','probe'):
            if not report(root,config,manifest,required)['pass']: raise RuntimeError('Production blocked: '+required+' report not passing')
    elif phase in ('pilot','probe') and not report(root,config,manifest,'diagnostic')['pass']:
        raise RuntimeError('Run and validate diagnostics first')
    if phase!='diagnostic' and not read(root/'controls.json')['pass']:
        raise RuntimeError('Missing passing regression controls')


def tasks(args):
    root=args.campaign.resolve();config,manifest=load(root);prerequisites(root,config,manifest,args.phase)
    all_tasks=phase_tasks(config,manifest,args.phase)
    nside=all_tasks[0]['nside']; phases=('imports','diagnostic','pilot') if args.phase=='pilot' else (args.phase,)
    records=completions(root,manifest,nside,phases)
    pending=[t for t in all_tasks if not covered(t,records)]
    if not pending: print('No pending tasks.');return
    table=root/'tables'/(args.phase+'-'+stamp()+'.json')
    atomic(table,{'phase':args.phase,'campaign_sha256':old.sha256(root/'campaign.json'),'tasks':pending})
    atomic(table.with_suffix('.sha256'),{'sha256':old.sha256(table)})
    script={'diagnostic':'diagnostic','pilot':'pilot-v2','probe':'probe','production':'production-v2'}[args.phase]
    groups=(len(pending)+3)//4
    # One debug probe job; normal queues use arrays and stable task snapshots.
    array='' if args.phase in ('diagnostic','probe') else f' --array=0-{groups-1}%'+('2' if args.phase=='production' else '1')
    print(f'sbatch{array} validation/alps-ode-{script}.sbatch --campaign {shlex_quote(str(root))} --table {shlex_quote(str(table))}')


def shlex_quote(value):
    import shlex
    return shlex.quote(value)


def terminal_job(job):
    if not str(job).isdigit(): return False
    active=subprocess.run(['squeue','-h','-j',str(job),'-o','%T'],capture_output=True,text=True)
    if active.returncode or active.stdout.strip(): return False
    accounting=subprocess.run(['sacct','-n','-X','-j',str(job),'--format=State','--parsable2'],capture_output=True,text=True)
    states=[line.split('|')[0].split()[0].rstrip('+') for line in accounting.stdout.splitlines() if line.strip()]
    return accounting.returncode==0 and bool(states) and all(s in TERMINAL for s in states)


@contextlib.contextmanager
def claim(directory,resume):
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Task already claimed: '+str(directory))
        owner=directory/'claim.json'
        if owner.exists():
            previous=read(owner)
            if not resume or not terminal_job(previous['job']): raise RuntimeError('Task has a live or unverified claim: '+str(owner))
            owner.rename(directory/('claim-interrupted-'+stamp()+'.json'))
        attempts=list(directory.glob('attempt-*'))
        if attempts and not resume: raise RuntimeError('Incomplete attempts exist; use --resume')
        atomic(owner,{'job':os.environ['SLURM_JOB_ID'],'pid':os.getpid(),'host':os.uname().nodename,'started':stamp()})
        try: yield
        finally: owner.unlink(missing_ok=True)


def command(root,config,manifest,t):
    invocation=old.command(Path(config['original']),manifest,t['variant'],t['face'],t['nside'])
    invocation[0]=root/'bin'/('lightcone-'+t['variant'])
    invocation += ['--subdivision-level',t['level'],'--part',t['part'],'--output-format','compact-hdf5',
                   '--run-id',t['run_id'],'--io-mode',t.get('io_mode',config['io_mode']),'--io-buffer-mib',config['io_buffer_mib']]
    return list(map(str,invocation))


def check_metrics(metrics,manifest,t):
    expected={'physics':manifest['variants'][t['variant']],'time_derivatives':'zero','storage':'ziggurat','ordering':'NESTED',
              'scalar_source':'phi','redshift_tolerance':manifest['redshift_tolerance'],'nside':t['nside'],'face':t['face'],
              'first_pixel':t['first'],'pixel_count':t['count'],'subdivision_level':t['level'],'part':t['part'],
              'run_id':t['run_id'],'stage':'complete','output_format':'compact-hdf5','residency_misses':0,
              'io_mode_requested':t.get('io_mode','auto')}
    mode=t.get('io_mode','auto')
    expected['io_mode']='tiles' if mode=='tiles' or (mode=='auto' and t['face'] in (4,6)) else 'regions'
    if any(metrics.get(k)!=v for k,v in expected.items()): raise RuntimeError('Wrong region configuration or incomplete run')
    counts=metrics['ray_status_counts']
    if counts.get('success')!=t['count'] or sum(counts.values())!=t['count']: raise RuntimeError('Region has failed rays')
    if abs(metrics['effective_lookback_cells']-manifest['synthetic']['lookback_cells'])>1e-8: raise RuntimeError('Changed integration cap')


def artifact(path):
    return {'path':str(path),'identity':old.identity(path),'sha256':old.sha256(path)}


def run(args):
    old.require_slurm();root=args.campaign.resolve();config,manifest=load(root)
    table=read(args.table)
    if old.sha256(args.table)!=read(args.table.with_suffix('.sha256'))['sha256']: raise RuntimeError('Task table changed')
    if table['campaign_sha256']!=old.sha256(root/'campaign.json'): raise RuntimeError('Wrong campaign task table')
    phase=table['phase']
    if phase not in PHASES: raise RuntimeError('Unknown task-table phase')
    prerequisites(root,config,manifest,phase)
    rank=int(os.environ.get('SLURM_PROCID','0'));group=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    if not 0<=rank<4 or group<0: raise RuntimeError('Expected rank 0..3 and nonnegative array index')
    index=4*group+rank
    if index>=len(table['tasks']): print('No task for this rank');return
    t=table['tasks'][index];expected=interval(manifest,t['variant'],t['nside'],t['face'],t['level'],t['part'])
    if any(t.get(k)!=v for k,v in expected.items()): raise RuntimeError('Invalid task table science')
    directory=root/phase/task_key(t)
    with claim(directory,args.resume):
        success=directory/'success.json'
        if success.exists():
            if not args.resume: raise RuntimeError('Completed task exists; use --resume')
            verified(read(success),manifest,deep=True,tool=root/'bin/validation-tools');print('Verified completed task:',task_key(t));return
        attempt=directory/('attempt-'+os.environ['SLURM_JOB_ID']+'-'+stamp());attempt.mkdir()
        invocation=command(root,config,manifest,t)
        atomic(attempt/'invocation.json',{'command':invocation,'job':os.environ['SLURM_JOB_ID'],'rank':rank,
             'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'task':t})
        resources=run_logged(invocation,attempt,attempt/'lightcone.log')
        metrics=read(attempt/'run-metrics.json');check_metrics(metrics,manifest,t)
        checking=time.monotonic();part=attempt/'ray-part.h5'
        run_logged([root/'bin/validation-tools','check-part',part],attempt,attempt/'fits-check.log')
        check_part_metadata(read(attempt/'fits-check.log'),t,manifest)
        if phase=='diagnostic' and t['face'] in (0,5):
            reference=Path(config['original'])/'pilot/corrected'/f"face-{t['face']:02d}"
            run_logged([root/'bin/validation-tools','compare-part',part,reference],attempt,attempt/'reference-check.log')
        artifacts=[artifact(part),artifact(attempt/'run-metrics.json')]
        result={'pass':True,'task':t,'part_file':str(part),'artifacts':artifacts,'metrics':metrics,'resources':resources,
                'binary_sha256':config['binary_sha256']['lightcone-'+t['variant']], 'checking_seconds':time.monotonic()-checking}
        load(root) # Recheck inputs after the long preload/trace before publishing success.
        atomic(success,result);print('Completed:',phase,task_key(t),flush=True)


def controls(args):
    old.require_slurm();root=args.campaign.resolve();config,manifest=load(root)
    base=root/'controls'
    with claim(base,args.resume):
        if (root/'controls.json').exists():
            saved=read(root/'controls.json')
            if saved.get('pass') and saved.get('campaign_sha256')==old.sha256(root/'campaign.json'): return
            raise RuntimeError('Invalid controls completion')
        directory=base/('attempt-'+stamp());directory.mkdir()
        for name in config['binary_sha256']:
            if (name.startswith('lightcone-') and name not in ('lightcone-corrected','lightcone-legacy')) or name.startswith('variant-'):
                run_logged([root/'bin'/name],directory,directory/(name+'.log'))
        for name in ('test_workflow.py','test_ode_comparison.py','test_campaign.py'):
            run_logged([sys.executable,ROOT/'validation'/name],ROOT,directory/(name+'.log'))
        atomic(root/'controls.json',{'pass':True,'job':os.environ['SLURM_JOB_ID'],'campaign_sha256':old.sha256(root/'campaign.json')})



def imports(args):
    old.require_slurm();root=args.campaign.resolve();config,manifest=load(root)
    original=Path(config['original']); tool=root/'bin/validation-tools'
    for path in sorted((original/'pilot/corrected').glob('face-*/success.json')):
        saved=read(path);face=int(path.parent.name.split('-')[1]);nside=manifest['pilot_nside']
        old.check_metrics(saved['metrics'],manifest,'corrected',nside)
        invocation=read(path.parent/'invocation.json')['command']
        expected=list(map(str,old.command(original,manifest,'corrected',face,nside)))
        if invocation!=expected: raise RuntimeError('Saved pilot invocation does not match fixture: '+str(path))
        t=interval(manifest,'corrected',nside,face,0,0);directory=root/'imports'/task_key(t)
        with claim(directory,args.resume):
            marker=directory/'success.json'
            if marker.exists(): verified(read(marker),manifest,True,tool);continue
            attempt=directory/('attempt-import-'+stamp());attempt.mkdir();part=attempt/'ray-part.h5'
            run_logged([tool,'check-face',path.parent,nside,face,manifest['target_z']],attempt,attempt/'source-check.log')
            run_logged([tool,'import-face',path.parent,part,nside,face,manifest['target_z'],manifest['redshift_tolerance'],
                        t['run_id'],json.dumps({key:manifest['variants']['corrected'][key] for key in old.PHYSICS_KEYS},separators=(',',':'))],attempt,attempt/'import.log')
            run_logged([tool,'check-part',part],attempt,attempt/'part-check.log')
            check_part_metadata(read(attempt/'part-check.log'),t,manifest)
            artifacts=[artifact(part),artifact(path),artifact(path.parent/'invocation.json')]
            artifacts += [artifact(path.parent/(name+f'_batch{face}.fits')) for name in ('dAmap','ellipticity1map','ellipticity2map','rotationmap','deflection1map','deflection2map','zmap','iswmap')]
            atomic(marker,{'pass':True,'task':t,'part_file':str(part),'artifacts':artifacts,'imported':True,'source':str(path)})


def summary(args):
    root=args.campaign.resolve();config,manifest=load(root);result=report(root,config,manifest,args.phase)
    if args.phase=='diagnostic':result['pass']=result['pass'] and (root/'controls.json').exists() and read(root/'controls.json')['pass']
    atomic(root/(args.phase+'-summary.json'),result);print(json.dumps(result,indent=2))
    return 0 if result['pass'] else 2


def assemble(args):
    old.require_slurm();root=args.campaign.resolve();config,manifest=load(root)
    if not report(root,config,manifest,'production')['pass']:raise RuntimeError('Production is incomplete')
    records=completions(root,manifest,manifest['nside'],('production',))
    for variant in old.VARIANTS:
        directory=root/'assembly'/variant
        with claim(directory,args.resume):
            if (directory/'success.json').exists():
                saved=read(directory/'success.json')
                for item in saved['artifacts']:
                    if old.identity(item['path'])!=item['identity'] or old.sha256(item['path'])!=item['sha256']: raise RuntimeError('Assembled output changed')
                continue
            attempt=directory/('attempt-'+stamp());attempt.mkdir()
            selected=canonical(records,variant)
            for record in selected:verified(record,manifest,True,root/'bin/validation-tools')
            resources=run_logged([root/'bin/validation-tools','assemble-parts',attempt/'fullsky',*[r['part_file'] for r in selected]],attempt,attempt/'assembly.log')
            files=sorted((attempt/'fullsky').glob('*.fits'))
            if len(files)!=8: raise RuntimeError('Missing assembled maps')
            # Publish a stable link only after helper verification and checksums.
            result={'pass':True,'variant':variant,'run_id':science_id(manifest,variant,manifest['nside']),
                    'resources':resources,'artifacts':[artifact(p) for p in files],'directory':str(attempt/'fullsky')}
            link=directory/('fullsky-link-'+stamp());link.symlink_to(attempt/'fullsky',target_is_directory=True);link.replace(directory/'fullsky')
            atomic(directory/'success.json',result)
    print('Both full-sky FITS sets validated and assembled.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('init','imports','controls','tasks','run','summary','assemble'))
    parser.add_argument('--campaign',type=Path,default=DEFAULT)
    parser.add_argument('--work-dir',type=Path,default=old.DEFAULT_WORK)
    parser.add_argument('--level',type=int,default=1)
    parser.add_argument('--phase',choices=PHASES,default='pilot')
    parser.add_argument('--table',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='run' and not args.table:parser.error('run requires --table')
    return globals()[{'init':'init','imports':'imports','controls':'controls','tasks':'tasks','run':'run','summary':'summary','assemble':'assemble'}[args.action]](args) or 0


if __name__=='__main__':
    sys.exit(main())

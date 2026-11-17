#!/usr/bin/env python3
"""Frozen 2400^3 ODE comparison. Scientific commands run only inside Slurm.

No Python packages beyond the standard library are required. Preparation streams
HDF5 through validation-tools; all scientific maps retain full NESTED indexing.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys

from run_static import C_KM_S, REDSHIFT_TOLERANCE, cosmology, distance_cells, run_logged, settings_file, simpson, write_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path('/ritom/scratch/cscs/adamek/raytracing/output2400')
DEFAULT_WORK = DEFAULT_DATA/'ode-comparison-z0.5'
PHYSICS_KEYS = ('correct_shear_transport', 'correct_sachs_b', 'nonlinear_scalar', 'correct_observer_distance')
VARIANTS = ('corrected', 'legacy')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run scientific preparation/tracing inside a Slurm allocation')


def static_info(path, config, target, margin=64):
    """Broad overlapping T=1 shells are intentional: the payload is static.

    No artificial time interpolation or boundary-sensitive shell choices affect
    these data. Only a(tau) varies. Include observation as an exact table knot.
    """
    n = config['ngrid']
    kappa = config['box_mpc']*config['h0_km_s_mpc']/C_KM_S
    om, ol = config['omega_m'], config['omega_lambda']
    observer = simpson(lambda u: 2/math.sqrt(om+ol*u**6), 0, 1)/kappa
    depth = distance_cells(target, config)
    steps = math.ceil((depth+margin)/2)
    spacing = 2/n

    def advance(a, dt):
        def f(value):
            if value <= 0:
                raise ValueError('Synthetic background crosses a=0')
            return kappa*math.sqrt(om*value+ol*value**4)
        k1=f(a); k2=f(a+dt*k1/2); k3=f(a+dt*k2/2); k4=f(a+dt*k3)
        return a+dt*(k1+2*k2+2*k3+k4)/6

    values = {0: 1.0}
    for i in range(1, steps+1):
        values[-i] = advance(values[1-i], -spacing)
    for i in range(1, 3):
        values[i] = advance(values[i-1], spacing)
    outer = (2*steps+128)/n
    with Path(path).open('xb') as stream:
        for cycle, index in enumerate(sorted(values)):
            stream.write(struct.pack('=i6d', cycle, observer+index*spacing, values[index], -1.0, outer, -1.0, outer))
    # Match the executable's floating-point subtraction for the cap.
    lookback = (observer-(observer-steps*spacing))*n
    return {'observer_tau': observer, 'observer_a': 1.0, 'vertex': [0.5]*3,
            'thickness': 1, 'spacing_cells': 2, 'records': len(values),
            'background_distance_cells': depth, 'lookback_cells': lookback,
            'extra_lookback_cells': lookback-depth, 'shell_boundaries': [-1.0, outer],
            'shell_policy': 'identical broad shells; later writes replace earlier ones; static payload',
            'background_model': 'flat matter + Lambda, no radiation, matching snapshot-raytracer'}


def prepare(args):
    require_slurm()
    if args.nside < 1 or args.nside & (args.nside-1) or args.padding_cells < 0 or args.slab_cells < 1:
        raise ValueError('Require power-of-two Nside, positive slabs and nonnegative padding')
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    data = args.data_dir.resolve()
    settings = data/'lc_1440box_2400_settings_used.ini'
    config = cosmology(settings_file(settings))
    phi = data/'lc_1440box_2400_snap004_phi.h5'
    vector = data/'lc_1440box_2400_snap004_B.h5'
    inputs = {'phi': identity(phi), 'B': identity(vector), 'settings': identity(settings)}
    snapshot_z = float(settings_file(settings)['snapshot redshifts'].split(',')[4])
    if snapshot_z != 0.25 or config['ngrid'] != 2400:
        raise ValueError('This workflow expects snap004 at z=0.25 and Ngrid=2400')
    # Full-sized zero-filled face maps: 2 variants * 12 faces * 8 maps.
    map_bytes = 2*12*8*12*args.nside**2*4
    conversion_bytes = 3*config['ngrid']**3*4
    needed = math.ceil(1.10*(map_bytes+conversion_bytes))
    if shutil.disk_usage(work).free < needed:
        raise RuntimeError(f'Need at least {needed} free bytes for conversion and maps (including 10% headroom)')
    (work/'bin').mkdir()
    binary_hashes = {}
    for name in ('lightcone-corrected','lightcone-legacy','validation-tools'):
        source = ROOT/'validation/bin'/name
        shutil.copy2(source, work/'bin'/name)
        binary_hashes[name] = sha256(source)
    shutil.copy2(ROOT/'validation/bin/build.json',work/'build.json')
    shutil.copy2(settings, work/'settings_used.ini')
    support = static_info(work/'lc_info.bin',config,args.target_z)
    resources = run_logged([work/'bin/validation-tools','prepare-static',phi,vector,work/'fixture',config['ngrid']],
                           work,work/'prepare.log')
    manifest = {'schema':1, 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'job':os.environ['SLURM_JOB_ID'], 'configuration':config, 'inputs':inputs,
                'snapshot_z':snapshot_z, 'target_z':args.target_z, 'nside':args.nside,
                'pilot_nside':256, 'ordering':'NESTED', 'time_derivatives':'zero', 'scalar_source':'phi',
                'b_layout':'staggered', 'storage':'ziggurat', 'shell_geometry':'legacy',
                'redshift_tolerance':REDSHIFT_TOLERANCE, 'slab_cells':args.slab_cells,
                'padding_cells':args.padding_cells, 'synthetic':support, 'binary_sha256':binary_hashes,
                'info_sha256':sha256(work/'lc_info.bin'), 'map_bytes':map_bytes,
                'conversion_bytes':conversion_bytes, 'prepare_resources':resources,
                'variants':{v:dict.fromkeys(PHYSICS_KEYS,int(v=='corrected')) for v in VARIANTS}}
    write_json(work/'manifest.json',manifest)  # Completion marker: written last.
    print('Prepared',work,flush=True)
    print(f'Production map payload: {map_bytes/2**40:.3f} TiB; B conversion: {conversion_bytes/2**30:.1f} GiB',flush=True)


def load(work):
    manifest = json.loads((work/'manifest.json').read_text())
    for key, saved in manifest['inputs'].items():
        if identity(saved['path']) != saved:
            raise RuntimeError(f'Input {key} changed since preparation')
    for name, digest in manifest['binary_sha256'].items():
        if sha256(work/'bin'/name) != digest:
            raise RuntimeError(f'Prepared binary {name} changed')
    if sha256(work/'lc_info.bin') != manifest['info_sha256']:
        raise RuntimeError('Synthetic info file changed')
    return manifest


def command(work, manifest, variant, face, nside):
    support = manifest['synthetic']
    return [work/'bin'/('lightcone-'+variant),work/'fixture/lc',nside,
            '--info',work/'lc_info.bin','--thickness',1,'--vertex',*support['vertex'],
            '--observer-tau',support['observer_tau'],'--target-z',manifest['target_z'],
            '--batch',face,'--ordering','nested','--storage','ziggurat',
            '--slab-cells',manifest['slab_cells'],'--padding-cells',manifest['padding_cells'],
            '--max-lookback-cells',support['lookback_cells'],'--time-derivatives','zero',
            '--b-layout','staggered','--scalar','phi','--redshift-tol',manifest['redshift_tolerance']]


def check_metrics(metrics, manifest, variant, nside):
    if (metrics['physics'] != manifest['variants'][variant] or metrics['time_derivatives'] != 'zero' or
        metrics['storage'] != 'ziggurat' or metrics['ordering'] != 'NESTED' or
        metrics['scalar_source'] != 'phi' or metrics['redshift_tolerance'] != REDSHIFT_TOLERANCE or
        abs(metrics['effective_lookback_cells']-manifest['synthetic']['lookback_cells']) > 1e-8 or
        metrics['residency_misses'] != 0):
        raise RuntimeError('Wrong run configuration or residency failure')
    counts = metrics['ray_status_counts']
    if counts['success'] != nside*nside or sum(counts.values()) != nside*nside:
        raise RuntimeError(f'Failed rays: {counts}')


def run_face(args):
    require_slurm()
    work = args.work_dir.resolve()
    manifest = load(work)
    pilot = args.phase == 'pilot'
    if not pilot and not json.loads((work/'pilot-summary.json').read_text())['pass']:
        raise RuntimeError('A successful, reviewed pilot summary is required before production')
    nside = manifest['pilot_nside'] if pilot else manifest['nside']
    # Six array elements, each with four tasks/GPUs. Every pair shares its rays,
    # fields, observer, cap and tolerance; only the compile-time ODE flags differ.
    group, rank = args.group, int(os.environ['SLURM_PROCID'])
    if not 0 <= group < 6 or not 0 <= rank < 4:
        raise ValueError('Require group 0..5 and SLURM_PROCID 0..3')
    variant, face = VARIANTS[group//3], (group%3)*4+rank
    directory = work/args.phase/variant/f'face-{face:02d}'
    directory.mkdir(parents=True,exist_ok=False)
    invocation = command(work,manifest,variant,face,nside)
    write_json(directory/'invocation.json', {'command':list(map(str,invocation)), 'job':os.environ['SLURM_JOB_ID'],
        'rank':rank, 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'nside':nside})
    resources = run_logged(invocation,directory,directory/'lightcone.log')
    metrics = json.loads((directory/'run-metrics.json').read_text())
    check_metrics(metrics,manifest,variant,nside)
    run_logged([work/'bin/validation-tools','check-face',directory,nside,face,manifest['target_z']],
               directory,directory/'fits-check.log')
    write_json(directory/'success.json',{'pass':True,'resources':resources,'metrics':metrics})
    print('Completed',variant,'face',face,'Nside',nside,flush=True)


def summarize(args):
    work = args.work_dir.resolve()
    manifest = load(work)
    nside = manifest['pilot_nside'] if args.phase == 'pilot' else manifest['nside']
    entries, missing = [], []
    for variant in VARIANTS:
        for face in range(12):
            path = work/args.phase/variant/f'face-{face:02d}'/'success.json'
            if not path.is_file():
                missing.append(str(path)); continue
            result = json.loads(path.read_text())
            check_metrics(result['metrics'],manifest,variant,nside)
            entries.append({'variant':variant,'face':face,**result})
    factor = (manifest['nside']/nside)**2
    report = {'pass':not missing and all(e['pass'] for e in entries), 'phase':args.phase,
              'missing':missing, 'runs':entries, 'nside':nside,
              'production_tracing_hours_estimate':max((e['metrics']['tracing_seconds']*factor/3600 for e in entries),default=None),
              'production_geometry_hours_estimate':max((e['metrics']['geometry_seconds']*factor/3600 for e in entries),default=None),
              'production_device_bytes_using_pilot_layout':max((e['metrics']['device_bytes']+
                  (manifest['nside']**2-nside**2)*52 for e in entries),default=None),
              'estimate_note':'Pixel-count scaling only; production occupancy, geometry, memory and I/O can differ. Review before submission.'}
    write_json(work/(args.phase+'-summary.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'},indent=2))
    return 0 if report['pass'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run-face','summarize'])
    parser.add_argument('--work-dir',type=Path,default=DEFAULT_WORK)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--nside',type=int,default=16384)
    parser.add_argument('--target-z',type=float,default=0.5)
    parser.add_argument('--slab-cells',type=int,default=16)
    parser.add_argument('--padding-cells',type=int,default=16)
    parser.add_argument('--phase',choices=['pilot','production'],default='pilot')
    parser.add_argument('--group',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args = parser.parse_args()
    if not math.isfinite(args.target_z) or args.target_z <= 0:
        parser.error('Target redshift must be positive and finite')
    if args.action == 'prepare': prepare(args)
    elif args.action == 'run-face': run_face(args)
    else: return summarize(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

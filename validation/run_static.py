#!/usr/bin/env python3
"""Prepare identical-slot fixtures and run the two GPU executables under one Slurm task.

Only the Python standard library is needed. HDF5 and FITS I/O use validation-tools.
Source snapshot files are never modified. Each invocation creates a fresh run directory.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

C_KM_S = 299792.458
MASS_OMEGA_EV = 93.14  # gevolution metadata.hpp


def settings_file(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if '=' in line:
            key, value = line.split('=', 1)
            result[key.strip()] = value.strip()
    return result


def cosmology(settings):
    h = float(settings['h'])
    if 'Omega_m' in settings:
        omega_m = float(settings['Omega_m'])
    else:
        baryons = float(settings.get('Omega_b', float(settings.get('omega_b', '0')) / h**2))
        cold = float(settings.get('Omega_cdm', float(settings.get('omega_cdm', '0')) / h**2))
        masses = [float(x) for x in settings.get('m_ncdm', '').split(',') if x.strip()]
        degrees = [float(x) for x in settings.get('deg_ncdm', '').split(',') if x.strip()] or [1.0] * len(masses)
        if len(degrees) != len(masses):
            raise ValueError('deg_ncdm and m_ncdm length mismatch')
        omega_m = baryons + cold + sum(m*d for m, d in zip(masses, degrees)) / MASS_OMEGA_EV / h**2
    box_mpc_h = float(settings['boxsize'])
    if not (h > 0 and 0 < omega_m < 1 and box_mpc_h > 0):
        raise ValueError('Invalid flat LCDM validation cosmology')
    return {'h': h, 'h0_km_s_mpc': 100*h, 'omega_m': omega_m, 'omega_lambda': 1-omega_m,
            'box_mpc_h': box_mpc_h, 'box_mpc': box_mpc_h/h, 'ngrid': int(settings['Ngrid'])}


def simpson(function, low, high, intervals=4096):
    step = (high-low)/intervals
    total = function(low) + function(high)
    total += sum((4 if i % 2 else 2)*function(low+i*step) for i in range(1, intervals))
    return total*step/3


def distance_cells(z, config):
    integral = simpson(lambda x: 1/math.sqrt(config['omega_m']*(1+x)**3+config['omega_lambda']), 0, z)
    return integral*C_KM_S/config['h0_km_s_mpc'] / (config['box_mpc']/config['ngrid'])


def synthetic_info(path, config, maximum_z):
    """Use the legacy thickness-3 shell equations with eight-cell cycle spacing.

    The width gives >= four cells of time support on either side of the background
    cone, covering the <= sqrt(12)-cell cubic-stencil radius plus a small ray delay.
    This is synthetic support; it is intentionally wider than the simulation's own
    light-cone output may have been. The payload is static everywhere.
    """
    n = config['ngrid']
    kappa = config['box_mpc']*config['h0_km_s_mpc']/C_KM_S
    om, ol = config['omega_m'], config['omega_lambda']
    tau_observer = simpson(lambda u: 2/math.sqrt(om+ol*u**6), 0, 1) / kappa
    spacing = 8/n
    depth = distance_cells(maximum_z, config)
    first = -math.ceil((depth+48)/8)
    last = 2  # background and field support slightly beyond observation

    def scale_at(tau):
        steps = max(1, math.ceil(abs(tau-tau_observer)/1e-5))
        dt = (tau-tau_observer)/steps
        a = 1.0
        def f(value):
            if value <= 0:
                raise ValueError('Requested synthetic table crosses the Big Bang')
            return kappa*math.sqrt(om*value+ol*value**4)
        for _ in range(steps):
            k1 = f(a)
            k2 = f(a+dt*k1/2)
            k3 = f(a+dt*k2/2)
            k4 = f(a+dt*k3)
            a += dt*(k1+2*k2+2*k3+k4)/6
        return a

    with open(path, 'xb') as stream:
        for cycle, k in enumerate(range(first, last+1)):
            tau = tau_observer+k*spacing
            radius = tau_observer-tau
            a = scale_at(tau)
            doubles = (tau, a, radius-spacing/2, radius+spacing/2,
                       radius-2*spacing, radius-spacing/2, radius+spacing/2, radius+1.5*spacing)
            stream.write(struct.pack('=i8d', cycle, *doubles))
    return {'observer_tau': tau_observer, 'observer_a': 1.0, 'spacing_cells': 8,
            'spacing_tau_over_box': spacing, 'records': last-first+1,
            'max_distance_cells': depth, 'thickness': 3}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run_logged(command, directory, logfile, timeout=None):
    print('+', ' '.join(map(str, command)), flush=True)
    resource_log = Path(str(logfile)+'.resources')
    wrapped = list(map(str, command))
    if Path('/usr/bin/time').is_file():
        wrapped = ['/usr/bin/time', '-v', '-o', str(resource_log), *wrapped]
    started = time.monotonic()
    with open(logfile, 'w') as output:
        result = subprocess.run(wrapped, cwd=directory, stdout=output,
                                stderr=subprocess.STDOUT, timeout=timeout)
    peak_rss = None
    if resource_log.is_file():
        for line in resource_log.read_text().splitlines():
            if 'Maximum resident set size (kbytes):' in line:
                peak_rss = int(line.split(':')[-1])
    metrics = {'wall_seconds': time.monotonic()-started, 'max_host_rss_kib': peak_rss}
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {logfile}')
    return metrics


def compare_logged(command, directory):
    result = subprocess.run(list(map(str, command)), text=True, capture_output=True)
    directory = Path(directory)
    (directory/'comparison.stdout').write_text(result.stdout)
    (directory/'comparison.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Comparison failed; see {directory}/comparison.stderr')
    report = json.loads(result.stdout)
    write_json(directory/'comparison.json', report)
    return report


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/ritom/scratch/cscs/adamek/raytracing'))
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--snapshot', type=int)
    parser.add_argument('--nside', type=int, default=16)
    parser.add_argument('--targets', type=float, nargs='+')
    parser.add_argument('--validate-isw', action='store_true', help='Validate finite-difference and zero modes against freshly computed corrected snapshot/light-cone references')
    parser.add_argument('--baseline-run', type=Path, help='Optional saved frozen run with corrected negative static shear transport')
    parser.add_argument('--validate-ziggurat', action='store_true', help='Use the 576 grid and validate all twelve resident slab batches')
    parser.add_argument('--slab-cells', type=int, default=16)
    parser.add_argument('--padding-cells', type=int, default=8)
    parser.add_argument('--max-lookback-cells', type=float)
    parser.add_argument('--batch', type=int, default=7, help='Also compare this batch at the deepest target; -1 disables')
    parser.add_argument('--all-batches', action='store_true', help='Compare all batches at the deepest target instead')
    parser.add_argument('--output-root', type=Path, default=root/'validation'/'results')
    parser.add_argument('--allow-local', action='store_true', help='Allow running on a workstation GPU instead of Slurm')
    args = parser.parse_args()
    if args.validate_isw:
        args.validate_ziggurat = True
    if args.snapshot is None:
        args.snapshot = 11 if args.validate_ziggurat else 10
    if args.targets is None:
        args.targets = [0.15] if args.validate_ziggurat else [0.05, 0.15, 0.3]
    if args.slab_cells <= 0 or args.padding_cells < 0:
        parser.error('Invalid slab thickness or padding')
    if args.max_lookback_cells is not None and (not math.isfinite(args.max_lookback_cells) or args.max_lookback_cells <= 0):
        parser.error('Look-back cap must be finite and positive')
    if not args.allow_local and not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run inside the submitted Slurm job; tests must not run on Alps login nodes')
    if args.nside <= 0 or args.nside > 1024 or args.nside & (args.nside-1) or args.snapshot < 0 or not -1 <= args.batch <= 11:
        parser.error('Invalid nside, snapshot, or batch')
    if not args.targets or any(not math.isfinite(z) or z <= 0 or z > 0.5 for z in args.targets):
        parser.error('Choose target redshifts in (0,0.5] for this debug-queue validation')
    data = args.data_dir.resolve()
    settings = args.settings or data/('unity6_4032box_576_settings_used.ini' if args.validate_ziggurat else 'unity6_4032box_288_settings_used.ini')
    parsed = settings_file(settings)
    config = cosmology(parsed)
    if config['ngrid'] < 8:
        parser.error('Validation needs Ngrid >= 8')
    base = parsed['snapshot file base'] + f'{args.snapshot:03d}'
    potential, vector = data/(base+'_phi.h5'), data/(base+'_B.h5')
    for path in (potential, vector):
        if not path.is_file():
            parser.error(f'Missing input: {path}')
    baseline = None
    if args.baseline_run is not None:
        if not args.validate_isw:
            parser.error('--baseline-run requires --validate-isw')
        baseline = args.baseline_run.resolve()
        if not (baseline/'manifest.json').is_file():
            parser.error(f'Missing frozen baseline manifest: {baseline}')
        saved = json.loads((baseline/'manifest.json').read_text())
        if saved.get('static_shear_transport') != 'negative':
            parser.error('Baseline predates the static shear-sign correction; use a corrected run or omit --baseline-run')
        if (saved['nside'] != args.nside or saved['snapshot_index'] != args.snapshot or
            saved['configuration'] != config or any(z not in saved['targets'] for z in args.targets)):
            parser.error('Baseline Nside, snapshot, cosmology or targets do not match this run')
        for target in args.targets:
            if not (baseline/'references'/f'z{target:g}'/'lightcone'/'zmap_batch0.fits').is_file():
                parser.error('Baseline full-sky FITS maps are missing')
    label = 'slurm-'+os.environ.get('SLURM_JOB_ID', 'local')+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = args.output_root.resolve()/label
    run.mkdir(parents=True, exist_ok=False)
    print('Results directory:', run, flush=True)
    tool = root/'validation'/'validation-tools'
    if not tool.is_file():
        raise RuntimeError('Compile validation/validation-tools first')
    info = run/'lc_info.bin'
    support = synthetic_info(info, config, max(args.targets))
    manifest = {'configuration': config, 'synthetic_lightcone': support, 'snapshot_index': args.snapshot,
                'snapshot_redshift': [float(x) for x in parsed['snapshot redshifts'].split(',')][args.snapshot],
                'targets': args.targets, 'nside': args.nside, 'validate_ziggurat': args.validate_ziggurat,
                'slab_cells': args.slab_cells, 'padding_cells': args.padding_cells, 'max_lookback_cells': args.max_lookback_cells,
                'time_derivative_modes': ['finite-difference','zero'] if args.validate_isw else ['finite-difference'],
                'frozen_baseline': str(baseline) if baseline else None,
                'static_shear_transport': 'negative',
                'background_model': 'flat matter + Lambda; radiation excluded identically from both tracers',
                'input_settings': str(Path(settings).resolve()),
                'inputs': [{'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in (potential, vector)],
                'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in list(root.glob('*.cu'))+list(root.glob('lightcone*.*')) if p.is_file()},
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    write_json(run/'manifest.json', manifest)
    run_logged([tool, 'prepare', potential, vector, run/'fixture', config['ngrid']], root, run/'prepare.log')
    run_logged([tool, 'check-fixture', info, config['ngrid'], support['observer_tau'], support['max_distance_cells']],
               root, run/'coverage.log')
    summary = {'manifest': str(run/'manifest.json'), 'cases': [], 'executions': [], 'pass': True}
    beta = config['box_mpc']/config['ngrid']*config['h0_km_s_mpc']/C_KM_S

    def record(report, directory, kind):
        summary['cases'].append({'directory': str(directory), 'kind': kind, **report})
        summary['pass'] &= report['pass']
        write_json(run/'summary.json', summary)
        print(f"{kind}: z={report['target_z']:g} batch={report['batch']}: "+
              ('PASS' if report['pass'] else 'REVIEW DIFFERENCES'), flush=True)

    def cone_case(directory, target, batch=-1, storage='full', ordering='nested', mode='finite-difference'):
        directory.mkdir(parents=True)
        command = [root/'lightcone-raytracer', run/'fixture'/'lc', args.nside,
                   '--info', info, '--thickness', 3, '--vertex', 0.5, 0.5, 0.5,
                   '--observer-tau', support['observer_tau'], '--target-z', target,
                   '--b-layout', 'staggered', '--storage', storage, '--ordering', ordering,
                   '--time-derivatives', mode]
        if batch >= 0:
            command += ['--batch', batch]
        if storage == 'ziggurat':
            command += ['--slab-cells', args.slab_cells, '--padding-cells', args.padding_cells]
        if args.max_lookback_cells is not None:
            command += ['--max-lookback-cells', args.max_lookback_cells]
        process = run_logged(command, directory, directory/'lightcone.log')
        tracer = json.loads((directory/'run-metrics.json').read_text())
        if tracer.get('static_shear_transport') != 'negative':
            raise RuntimeError(f'Tracer binary lacks the static shear-sign correction in {directory}')
        if tracer['time_derivatives'] != mode:
            raise RuntimeError(f'Incorrect derivative mode in {directory}')
        if tracer['residency_misses']:
            raise RuntimeError(f'Residency misses in {directory}')
        summary['executions'].append({'directory': str(directory), 'process': process, 'tracer': tracer})
        write_json(run/'summary.json', summary)
        return directory

    references = {}
    for target in sorted(set(args.targets)):
        reference = run/'references'/f'z{target:g}'
        snapshot_dir = reference/'snapshot'
        snapshot_dir.mkdir(parents=True)
        process = run_logged([root/'snapshot-raytracer', potential, vector, args.nside,
                              '--target-z', target, '--box-size-mpc', config['box_mpc'],
                              '--omega-m', config['omega_m'], '--omega-lambda', config['omega_lambda'],
                              '--h0', config['h0_km_s_mpc']], snapshot_dir, reference/'snapshot.log')
        summary['executions'].append({'directory': str(snapshot_dir), 'process': process, 'storage': 'snapshot'})
        cone_dir = cone_case(reference/'lightcone', target)
        report = compare_logged([tool, 'compare', snapshot_dir, cone_dir, args.nside, -1, target,
                                 distance_cells(target, config), beta, config['omega_m'], config['omega_lambda']], reference)
        record(report, reference, 'snapshot-vs-full')
        references[target] = (snapshot_dir, cone_dir)

    target = max(args.targets)
    snapshot_dir, full = references[target]
    faces = range(12) if args.validate_ziggurat or args.all_batches else ([args.batch] if args.batch >= 0 else [])
    for face in faces:
        directory = run/'faces'/f'face-{face:02d}'
        cone_case(directory, target, face, 'ziggurat' if args.validate_ziggurat else 'full')
        strict = compare_logged([tool, 'compare-strict', full, directory, args.nside, face, target, '--require-zero-isw'], directory)
        record(strict, directory, 'full-vs-face')
        # Compare the face's sky positions against the full snapshot, never its old RING batch.
        broad_dir = directory/'snapshot-comparison'
        broad_dir.mkdir()
        report = compare_logged([tool, 'compare', snapshot_dir, directory, args.nside, face, target,
                                 distance_cells(target, config), beta, config['omega_m'], config['omega_lambda']], broad_dir)
        record(report, broad_dir, 'snapshot-vs-face')
    if args.validate_ziggurat or args.all_batches:
        directory = run/'assembly-comparison'; directory.mkdir()
        report = compare_logged([tool, 'assemble', full, run/'faces', args.nside, target, '--require-zero-isw'], directory)
        record(report, directory, 'twelve-face-assembly')
    if args.validate_ziggurat:
        directory = run/'ring-output'
        cone_case(directory, target, 7, 'ziggurat', 'ring')
        report = compare_logged([tool, 'compare-strict', full, directory, args.nside, 7, target, '--require-zero-isw'], directory)
        if any(m['candidate_ordering'] != 'RING' for m in report['maps']):
            raise RuntimeError('Requested RING output has incorrect FITS ordering')
        record(report, directory, 'ring-output')
    if args.validate_isw:
        # Only explicitly supplied corrected baselines can be unchanged-map references.
        if baseline is not None:
            for target, (_, full_target) in references.items():
                old_full = baseline/'references'/f'z{target:g}'/'lightcone'
                directory = full_target/'baseline-comparison'; directory.mkdir()
                record(compare_logged([tool,'compare-frozen',old_full,full_target,args.nside,-1,target],directory),
                       directory,'saved-baseline-vs-finite-difference')
        zero_root = run/'zero-mode'
        for target in sorted(references):
            full_target = references[target][1]
            zero = cone_case(zero_root/'references'/f'z{target:g}'/'lightcone',target,mode='zero')
            record(compare_logged([tool,'compare-strict',full_target,zero,args.nside,-1,target,'--require-zero-isw'],zero),
                   zero,'finite-difference-vs-zero')
            if baseline is not None:
                directory = zero/'baseline-comparison'; directory.mkdir()
                old_full = baseline/'references'/f'z{target:g}'/'lightcone'
                record(compare_logged([tool,'compare-frozen',old_full,zero,args.nside,-1,target],directory),
                       directory,'saved-baseline-vs-zero')
        target = max(args.targets); full = references[target][1]
        for face in range(12):
            directory = cone_case(zero_root/'faces'/f'face-{face:02d}',target,face,'ziggurat',mode='zero')
            record(compare_logged([tool,'compare-strict',full,directory,args.nside,face,target,'--require-zero-isw'],directory),
                   directory,'full-vs-zero-face')
        directory = zero_root/'assembly-comparison'; directory.mkdir()
        record(compare_logged([tool,'assemble',full,zero_root/'faces',args.nside,target,'--require-zero-isw'],directory),
               directory,'zero-twelve-face-assembly')
        directory = cone_case(zero_root/'ring-output',target,7,'ziggurat','ring',mode='zero')
        report = compare_logged([tool,'compare-strict',full,directory,args.nside,7,target,'--require-zero-isw'],directory)
        if any(m['candidate_ordering'] != 'RING' for m in report['maps']):
            raise RuntimeError('Zero-mode RING output has incorrect FITS ordering')
        record(report,directory,'zero-ring-output')
    print('Results:', run, flush=True)
    print('Summary:', run/'summary.json', flush=True)
    return 0 if summary['pass'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'Validation failed: {error}', file=sys.stderr)
        raise

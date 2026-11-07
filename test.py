"""
Step 3 of 3: evaluate the trained model on every test cell.

    python test.py [--config config/coma.cfg] [--limit N] [--use_last]

Uses the best-validation EMA weights by default. Per-sample errors go to
<work_dir>/results/<run_name>/<cell>/per_sample.csv, and the summary table (missing-region
L2 in mm, by mask type) to <work_dir>/results/<run_name>/summary.csv. An interrupted run
resumes where it stopped. --limit N evaluates only the first N samples per cell
(quick check).
"""
import os
import sys
import csv
import argparse
import configparser
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CELLS = ('seen_subject', 'unseen_subject', 'unseen_expression', 'unseen_both')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', default=os.path.join(ROOT, 'config', 'coma.cfg'))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--use_last', action='store_true', help='last-epoch weights instead of best-val')
    ap.add_argument('--run_name', default=None, help='evaluate another run (default: run_name in the config)')
    args = ap.parse_args()

    cp = configparser.RawConfigParser()
    cp.read(args.config)
    from config.config import read_config
    work = read_config(args.config)['work_dir']
    predict = cp.get('diffusion parameters', 'predict')
    run_name = args.run_name or cp.get('training parameters', 'run_name', fallback='full')
    cells = [c for c in CELLS if os.path.exists(os.path.join(work, 'benchmark', c, 'index.csv'))]
    if not cells:
        sys.exit(f'No benchmark in {work}. Run: python prepare_data.py')

    from scripts.evaluate_benchmark import main as evaluate
    results = os.path.join(work, 'results', run_name + ('_last' if args.use_last else ''))
    for c in cells:
        print(f'\n######## {c} ########')
        sys.argv = ['evaluate_benchmark.py', '--config_file', args.config,
                    '--param', predict, '--tag', run_name,
                    '--bench_dir', os.path.join(work, 'benchmark', c),
                    '--out_dir', os.path.join(results, c)]
        sys.argv += ['--use_last'] if args.use_last else []
        sys.argv += ['--limit', str(args.limit)] if args.limit else []
        evaluate()

    # summary: mean missing-region error per cell and mask type
    table = {}
    for c in cells:
        err = defaultdict(list)
        with open(os.path.join(results, c, 'per_sample.csv')) as fh:
            for r in csv.DictReader(fh):
                err[r['strategy']].append(float(r['missing_mm']))
                err['all'].append(float(r['missing_mm']))
        table[c] = {k: sum(v) / len(v) for k, v in err.items()}
        table[c]['n'] = len(err['all'])
    cols = [s for s in ('random_scatter', 'bfs_patch', 'plane_cut')
            if any(s in row for row in table.values())] + ['all']
    ref = table.get('seen_subject', {}).get('all')
    print('\nMissing-region L2 error (mm)')
    print(f'{"cell":18s} {"n":>5s} ' + ' '.join(f'{c:>14s}' for c in cols) + '   vs seen')
    with open(os.path.join(results, 'summary.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cell', 'n'] + cols)
        for c, row in table.items():
            w.writerow([c, row['n']] + [f'{row.get(k, float("nan")):.3f}' for k in cols])
            rel = f'{100 * (row["all"] / ref - 1):+.0f}%' if ref and c != 'seen_subject' else ''
            print(f'{c:18s} {row["n"]:5d} ' + ' '.join(f'{row.get(k, float("nan")):14.3f}' for k in cols)
                  + f'   {rel}')
    print('summary:', os.path.join(results, 'summary.csv'))


if __name__ == '__main__':
    main()

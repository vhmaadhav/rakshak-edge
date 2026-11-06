"""Run an explicit unfinished/focused unit selection on Kaggle, using synthetic fixtures only."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from _common import env_report, write_metrics


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(ctx):
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--selection-sha256', required=True)
    args = parser.parse_args(ctx.argv)
    if not torch.cuda.is_available() or not str(ctx.work).startswith('/kaggle/working'):
        raise RuntimeError('private Kaggle T4 unit validation required; never run this job locally')
    repo = Path(ctx.src).parent
    selection_path = repo/'kaggle/test_selection.json'
    if sha(selection_path) != args.selection_sha256:
        raise ValueError('unit selection changed from predeclared snapshot')
    selection = json.loads(selection_path.read_text())
    targets = selection['targets']
    if not targets or len(targets) != len(set(targets)):
        raise ValueError('unique explicit unit selection required')
    full = selection.get('collect_all', False)
    if full and (targets != ['tests'] or selection['expected_tests'] != 'collected'):
        raise ValueError('full unit mode requires every test and collection-derived count')
    for target in ([] if full else targets):
        path = Path(target.split('::')[0])
        if (not str(path).startswith('tests/') or '..' in path.parts or path.suffix != '.py'
                or not (repo/path).is_file()):
            raise ValueError('only bundled synthetic-fixture test targets allowed')
    # Some unit tests exercise git-based source bundling. Use a fresh scratch index
    # of source code only, with no commit, identity override, remote or credentials.
    if (repo/'.git').exists():
        raise RuntimeError('unit scratch repository unexpectedly exists')
    subprocess.run(['git', 'init', '--quiet', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '--', 'src', 'kaggle', 'tests',
                    'pyproject.toml', 'colab'], check=True)
    env = {**os.environ, 'PYTHONPATH': str(repo/'src'), 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    log_path, xml_path = ctx.work/'pytest.log', ctx.work/'pytest-junit.xml'
    expected = selection['expected_tests']
    collected = None
    if full:
        discovery = subprocess.run(
            ['nice', '-n', '19', 'taskset', '-c', '0-3', sys.executable, '-m', 'pytest',
             '--collect-only', '-q', 'tests'], cwd=repo, env=env, capture_output=True, text=True)
        (ctx.work/'pytest-collection.log').write_text(discovery.stdout+discovery.stderr)
        collected = [line for line in discovery.stdout.splitlines()
                     if line.startswith('tests/') and '::' in line]
        if discovery.returncode or len(collected) < selection['minimum_collected']:
            raise ValueError('milestone collection failed or inventory unexpectedly shrank')
        expected = len(collected)
        (ctx.work/'collected_nodes.json').write_text(json.dumps(collected, indent=2)+'\n')
    command = ['nice', '-n', '19', 'taskset', '-c', '0-3', sys.executable, '-m', 'pytest',
               '-q', '--junitxml='+str(xml_path), *targets]
    with log_path.open('w') as stream:
        process = subprocess.run(command, cwd=repo, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT)
    counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
    if xml_path.exists():
        root = ET.parse(xml_path).getroot()
        suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
        counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in counts}
    passed = (process.returncode == 0 and counts['tests'] == expected
              and counts['failures'] == counts['errors'] == 0
              and counts['skipped'] <= selection['maximum_skipped'])
    metrics = {'validation_gate': 'go' if passed else 'kill', 'counts': counts,
               'pytest_exit': process.returncode, 'selection': selection,
               'expected_tests': expected, 'collected_nodes_sha256': (
                   sha(ctx.work/'collected_nodes.json') if collected is not None else None),
               'selection_sha256': args.selection_sha256, 'environment': env_report(),
               'source_tree': ctx.results['tree_sha256'],
               'scope': 'synthetic-fixture unit validation only; no corpus/datasets attached',
               'scratch_git': 'source index only, no commit/identity/remote',
               'completed_local_tests_rerun': False,
               'pytest_version': __import__('importlib.metadata', fromlist=['version']).version(
                   'pytest')}
    write_metrics('metrics.json', metrics)
    ctx.log('\n'.join(log_path.read_text().splitlines()[-24:]))
    ctx.results['summary'] = f'unit validation {metrics["validation_gate"]}; {counts}'
    return passed

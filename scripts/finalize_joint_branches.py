"""Preserve pilot raw data and publish explicit commanded-strike contact labels."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_root', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.raw_root / 'manifest.json').read_text())
    if manifest.get('schema') != 'joint_full_decision_tail_v1':
        raise ValueError('Only pilot v1 data needs finalization; v2 already separates strike/observed contact')
    if manifest['split'] != 'train':
        raise ValueError('Only finalize training pilot data')
    destination = ROOT / 'outputs/joint_branch_data' / datetime.now().strftime('finalized_%Y%m%d_%H%M%S')
    run = destination.with_name('.' + destination.name + '.partial')
    run.mkdir(parents=True)
    manifest.update(schema='joint_full_decision_tail_v2', raw_root=str(args.raw_root.resolve()),
                    contact_definition='physical_contact_during_commanded_strike',
                    pilot_time_definition='first measured physical contact, quantized at 1 ms, only when execution reason is OK')
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    hashes, contacts, misses = {}, 0, 0
    for path in sorted(args.raw_root.glob('execution_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        reason = str(values['execution_reason'])
        if reason not in ('OK', 'NO_CONTACT', 'NO_PHYSICAL_CONTACT'):
            raise ValueError(f'Unclassified pilot outcome {reason}')
        if bool(values['execution_valid']) != (reason == 'OK'):
            raise ValueError('Inconsistent pilot execution label')
        successful_strike = reason == 'OK'
        if successful_strike and not bool(values['contact']):
            raise ValueError('Successful pilot has no measured physical contact')
        values['observed_contact_s'] = values['actual_contact_s'].copy()
        values['contact'] = np.array(successful_strike)
        if not successful_strike:
            values['actual_contact_s'] = np.array(np.nan)
        trajectory = values['actual_trajectory']
        if trajectory[-1, 0] < 1. - 1e-7 or np.any(np.diff(trajectory[:, 0]) <= 0):
            raise ValueError('Missing full decision-relative observation tail')
        np.savez_compressed(run / path.name, **values)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        contacts += successful_strike
        misses += not successful_strike
    (run / 'raw_sha256.json').write_text(json.dumps(hashes, indent=2))
    summary = {'files': len(hashes), 'strike_contacts': contacts, 'strike_misses': misses,
               'raw_data_preserved': True, 'contact_definition': manifest['contact_definition']}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2))
    run.replace(destination)
    print(f'RUN_DIR={destination}')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

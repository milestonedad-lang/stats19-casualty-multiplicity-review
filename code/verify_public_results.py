"""Verify reported aggregate results using only the Python standard library.

This checks report consistency; it does not fit models or validate assumptions.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rows(name):
    with (ROOT / 'tables' / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    table = rows('Table1_source.csv')
    totals = {key: sum(int(r[key]) for r in table) for key in
              ['collisions', 'multiple_KSI_collisions', 'KSI_casualties', 'additional_KSI_casualties']}
    assert totals == {'collisions': 50735, 'multiple_KSI_collisions': 4119,
                      'KSI_casualties': 56153, 'additional_KSI_casualties': 5418}
    assert round(100 * totals['multiple_KSI_collisions'] / totals['collisions'], 2) == 8.12
    assert round(100 * (totals['multiple_KSI_collisions'] + totals['additional_KSI_casualties']) /
                 totals['KSI_casualties'], 2) == 16.98
    coefficients = rows('all_regression_coefficients.csv')
    contrasts = rows('vehicle_contrasts.csv')
    assert len(coefficients) == 4832
    assert len({r['model'] for r in coefficients}) == 62
    assert len(contrasts) == 186
    mapping = {(r['model'], r['parameter']): r for r in coefficients}
    primary = {}
    for r in contrasts:
        c = mapping[(r['model'], r['contrast'])]
        assert abs(float(c['beta']) - float(r['beta'])) < 1e-9
        assert abs(float(c['se']) - float(r['se'])) < 1e-9
        assert abs(math.exp(float(r['beta'])) - float(r['ratio'])) < 1e-9
        if r['model'] == 'M2_primary':
            primary[r['contrast']] = round(float(r['ratio']), 2)
    assert primary == {'exposure=car_hgv': .52, 'exposure=car_motorcycle': .14,
                       'exposure=other_known': .24}
    assert len(rows('per_imputation_model_fit.csv')) == 2904
    assert len(rows('Fig1_source.csv')) == 9
    assert len(rows('Fig2_source.csv')) == 42
    return {'status': 'PASS', 'scope': 'Aggregate report and file-integrity verification only; no model refitting',
            'verified_files': len(manifest['sha256']), 'models': 62, 'vehicle_contrasts': 186,
            'coefficient_rows': 4832, 'fit_records': 2904, 'totals': totals,
            'multiple_collision_percent': 8.12, 'share_of_KSI_people_percent': 16.98,
            'primary_adjusted_odds_ratios_rounded': primary}

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))

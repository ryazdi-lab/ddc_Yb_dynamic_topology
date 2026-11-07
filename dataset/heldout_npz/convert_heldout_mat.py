"""Convert held-out MATLAB matrices using existing training normalization."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from normalize_ablation_datasets import decode_x, encode_x, inverse_dq, sha256

STAT_KEYS = ['yb_asinh_scale', 'input_center', 'input_scale', 'dq_center',
             'dq_scale', 'vmag_center', 'vmag_scale']


def convert(mat_path, normalization_path, output_dir):
    mat_path = Path(mat_path)
    normalization_path = Path(normalization_path)
    output_dir = Path(output_dir)
    output = output_dir / 'dataset_heldout.npz'
    if output.exists():
        raise FileExistsError(output)
    stats = json.loads(normalization_path.read_text(encoding='utf-8'))
    if stats['schema_version'] != 'ddc_yb_train_normalized_v2':
        raise ValueError('Expected existing v2 training statistics')
    data = loadmat(mat_path)
    dc = data['Ddc']
    dq = data['DQ']
    yb = data['Yb']
    if np.iscomplexobj(dc) or np.iscomplexobj(dq):
        raise ValueError('Ddc and DQ must be real')
    if dc.ndim != 3 or dc.shape != dq.shape or dc.shape[0] != dc.shape[1]:
        raise ValueError('Expected matching Ddc and DQ of shape (buses,buses,samples)')
    buses, _, samples = dc.shape
    if len(stats['vmag_center']) != buses:
        raise ValueError('Normalization bus count does not match the MAT file')
    if not np.iscomplexobj(yb) or yb.shape != (buses, buses):
        raise ValueError('Expected one shared complex Yb matrix')
    if not all(np.isfinite(a).all() for a in (dc, dq, yb)):
        raise ValueError('Nonfinite source values; no samples are silently removed')

    dc_batch = np.moveaxis(dc, 2, 0)
    yb_batch = np.broadcast_to(yb, dc_batch.shape)
    x_raw = np.stack([dc_batch, yb_batch.real, yb_batch.imag], axis=1)
    dq_raw = np.moveaxis(dq, 2, 0)[:, None]
    x = encode_x(x_raw, stats)
    y = ((dq_raw - stats['dq_center']) / stats['dq_scale']).astype(np.float32)
    adjacency = (yb_batch.real != 0) | (yb_batch.imag != 0)
    adjacency[:, np.arange(buses), np.arange(buses)] = False
    adjacency |= adjacency.transpose(0, 2, 1)
    arrays = {
        'X': x, 'y': y, 'X_raw': x_raw, 'DQ_raw': dq_raw,
        'Ddc': dc_batch, 'DQ': dq_raw[:, 0], 'Yb': yb_batch.copy(),
        'adjacency': adjacency,
        'source_indices': np.arange(samples, dtype=np.int64),
        'channel_names': np.array(['Ddc', 'Re_Yb', 'Im_Yb']),
        'schema_version': np.array(stats['schema_version']),
        'dataset_id': np.array(mat_path.stem),
        'source_file': np.array(mat_path.name),
        'available_targets': np.array(['dq']),
        'normalization_fit_split': np.array('original_training_set'),
    }
    for key in STAT_KEYS:
        arrays[key] = np.array(stats[key], dtype=np.float64)
    assert np.array_equal(x_raw[:, 0], dc_batch)
    assert np.array_equal(x_raw[:, 1] + 1j*x_raw[:, 2], yb_batch)
    assert np.array_equal(dq_raw[:, 0], np.moveaxis(dq, 2, 0))
    assert np.allclose(decode_x(x, stats), x_raw, rtol=2e-5, atol=2e-5)
    assert np.allclose(inverse_dq(y, stats), dq_raw, rtol=2e-5, atol=1e-5)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    with np.load(output, allow_pickle=False) as saved:
        for key, value in arrays.items():
            assert np.array_equal(saved[key], value), key
    (output_dir/'normalization.json').write_bytes(normalization_path.read_bytes())
    manifest = {
        'source_file': str(mat_path.resolve()), 'source_sha256': sha256(mat_path),
        'samples': samples, 'buses': buses, 'removed_samples': 0,
        'split': 'external_heldout', 'targets': ['dq'],
        'X_shape': list(x.shape), 'y_shape': list(y.shape),
        'source_matrix_layout': 'Ddc/DQ: buses,buses,samples; Yb: buses,buses',
        'raw_output_layout': 'Ddc/DQ/Yb: samples,buses,buses',
        'shared_Yb_broadcast_over_samples': True,
        'normalization_source': str(normalization_path.resolve()),
        'normalization_sha256': sha256(normalization_path),
        'normalization_refitted': False, 'sample_indices': 'local zero-based MAT sample indices',
        'output_file': output.name, 'output_sha256': sha256(output),
        'max_dq_inverse_error': float(np.max(np.abs(inverse_dq(y, stats)-dq_raw))),
        'raw_preservation_verified': True,
    }
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'{output}: X={x.shape}, y={y.shape}, samples={samples}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mat', required=True)
    parser.add_argument('--normalization', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    convert(args.mat, args.normalization, args.out_dir)

"""Quick CPU checks for the triangle model and residual-gate head.

    python test_tri_model.py                       # synthetic checks, ~1 minute
    python test_tri_model.py --data-dir /path/to/case30   # also checks the real dataset constants
"""
import argparse
import torch
from dataset import VARIANTS
from models import AblationModel, head_constants, parameter_hash, nonzero_mask
from triangle_model import TriangleMultiplication

B_BUSES = 12


def synthetic_batch(n=4, buses=B_BUSES, ddc_zero=-0.5):
    x = torch.randn(n, 3, buses, buses)
    zero_cells = torch.rand(n, buses, buses) < 0.4
    x[:, 0][zero_cells] = ddc_zero          # exact "raw Ddc == 0" value
    return x, zero_cells


def gate_kwargs():
    return dict(head='residual_gate', ddc_zero=-0.5, dq_zero=-0.49, skip_a=0.9, skip_b=0.02)


def test_triangle_math():
    for outgoing in (True, False):
        m = TriangleMultiplication(8, 4, outgoing)
        a, b = torch.randn(2, 5, 5, 4), torch.randn(2, 5, 5, 4)
        got = m.mix(a, b)
        want = torch.zeros_like(got)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    want[:, i, j] += (a[:, i, k] * b[:, j, k]) if outgoing else (a[:, k, i] * b[:, k, j])
        assert torch.allclose(got, want, atol=1e-5), outgoing
    print('ok  triangle update matches explicit loops (outgoing and incoming)')


def test_models(model_type, head):
    kwargs = gate_kwargs() if head == 'residual_gate' else {}
    hashes, counts = [], []
    for variant, channels in VARIANTS.items():
        torch.manual_seed(7)
        model = AblationModel(model_type, variant, 'dq', B_BUSES, 16, 2, 4, 0.1, 2., **kwargs).eval()
        hashes.append(parameter_hash(model)); counts.append(sum(p.numel() for p in model.parameters()))
        x, zero_cells = synthetic_batch()
        x.requires_grad_(True)
        out = model(x)
        assert out.shape == (4, 1, B_BUSES, B_BUSES) and torch.isfinite(out).all()
        out.square().sum().backward()
        excluded = [c for c in range(3) if c not in channels]
        if excluded:
            assert torch.count_nonzero(x.grad[:, excluded]) == 0, 'gradient reached an excluded channel'
            changed = x.detach().clone(); changed[:, excluded] += 123.
            with torch.no_grad():
                assert torch.equal(model(changed), model(x.detach())), 'excluded values changed output'
        if head == 'residual_gate':
            with torch.no_grad():
                out = model(x.detach())
                assert (out[:, 0][zero_cells] == -0.49).all(), 'gate must give the exact zero constant'
                # zero-initialised last layer: output is exactly the linear skip on kept entries
                kept = ~zero_cells
                want = 0.9 * x.detach()[:, 0] + 0.02
                assert torch.allclose(out[:, 0][kept], want[kept], atol=1e-6), 'step-0 output is not the baseline'
    assert len(set(hashes)) == 1 and len(set(counts)) == 1, 'variants must share init and parameter count'
    print(f'ok  {model_type:5s} head={head:13s} shapes, gradients, exclusion, gate/baseline, paired init')


def test_vmag():
    for model_type in ('axial', 'dq', 'tri'):
        model = AblationModel(model_type, 'ddc_yb', 'vmag', B_BUSES, 16, 1, 4, 0.1, 2.).eval()
        assert model(synthetic_batch()[0]).shape == (4, B_BUSES)
    print('ok  vmag heads (axial, dq, tri)')


def test_overfit_step():
    torch.manual_seed(0)
    model = AblationModel('tri', 'ddc_yb', 'dq', B_BUSES, 16, 2, 4, 0., 2., **gate_kwargs()).train()
    x, _ = synthetic_batch(8)
    target = torch.randn(8, 1, B_BUSES, B_BUSES)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        opt.zero_grad(); loss = (model(x) - target).square().mean(); loss.backward(); opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.7 * first, (first, float(loss))
    print(f'ok  tri model can fit a batch (loss {first:.3f} -> {float(loss):.3f})')


def test_real_dataset(data_dir):
    from pathlib import Path
    from dataset import DependencyMatrixDataset
    train = DependencyMatrixDataset(Path(data_dir) / 'dataset_train.npz', 'dq')
    const = head_constants(train)
    keep = nonzero_mask(train.X[:, :1], train.stats)
    print('ok  real data constants', {k: round(v, 5) for k, v in const.items()}, 'kept fraction', round(float(keep.float().mean()), 4))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data-dir')
    a = p.parse_args(); torch.set_num_threads(4)
    test_triangle_math()
    for model_type in ('axial', 'dq', 'tri'):
        for head in ('plain', 'residual_gate'):
            test_models(model_type, head)
    test_vmag(); test_overfit_step()
    if a.data_dir: test_real_dataset(a.data_dir)
    print('all checks passed')

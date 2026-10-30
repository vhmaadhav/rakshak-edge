"""GPU smoke job: torch sees a CUDA device and a small matmul runs on it and matches the CPU result.

Launch: kaggle/launch.sh gpu_smoke --gpu t4 --internet
Returns True when torch imports, CUDA is available and the GPU matmul agrees with a float64 CPU
reference.
Writes metrics.json.
"""

from _common import env_report, seed_everything, timed, write_metrics

SIZE = 256
TOLERANCE = 1e-2  # absolute, on values of order 10 from a float32 matmul


def run(ctx):
    seed_everything(0)
    env = env_report()
    ctx.log(f'env: {env}')
    torch_info = env['torch'] or {}
    torch_present = bool(torch_info) and 'import_error' not in torch_info
    cuda = bool(torch_info.get('cuda_available'))
    device = torch_info.get('device')
    ctx.log(f"torch {torch_info.get('version')} cuda_build {torch_info.get('cuda_build')} "
            f'cuda_available {cuda} device {device}')

    timings = {}
    matmul = {'ok': False, 'shape': [SIZE, SIZE], 'device': device, 'max_abs_err': None}
    if torch_present and cuda:
        import torch

        with timed('matmul_s', timings):
            torch.manual_seed(0)
            a = torch.randn(SIZE, SIZE, device='cuda')
            b = torch.randn(SIZE, SIZE, device='cuda')
            c = (a @ b).cpu().double()
            torch.cuda.synchronize()
            ref = a.cpu().double() @ b.cpu().double()
            err = (c - ref).abs().max().item()
        matmul.update(ok=err < TOLERANCE, max_abs_err=err, device=torch.cuda.get_device_name(0))
        ctx.log(f"matmul {SIZE}x{SIZE} on {matmul['device']}: max abs err {err:.3g}")

    checks = {
        'torch_present': torch_present,
        'cuda_available': cuda,
        'gpu_matmul_matches_cpu': matmul['ok'],
    }
    passed = all(checks.values())
    gpu_names = ', '.join(row['name'] for row in env['gpu'] or []) or 'none'
    metrics = {
        'job': 'gpu_smoke',
        'env': env,
        'checks': checks,
        'matmul': matmul,
        'timings': timings,
        'passed': passed,
    }
    write_metrics('metrics.json', metrics)
    status = 'ok' if passed else 'fail'
    ctx.results['summary'] = f'cuda {status} gpu {gpu_names} torch {torch_info.get("version")}'
    ctx.log(f'wrote metrics.json: checks {checks}')
    return passed

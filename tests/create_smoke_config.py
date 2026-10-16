"""Tiny deterministic dual-model input for testing the bundled .exe on Windows CI."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_config import save_json

E, nu, rho = 70e9, 0.3, 2700
G = E / (2 * (1 + nu))
config = {
    'a': 0.6, 'b': 0.5, 'nx': 4, 'ny': 4,
    'clpt': {'nx': 2, 'ny': 2},
    'kw': 1e7, 'kp': 1e5,
    'edges': {edge: 'simply_supported' for edge in ('left', 'right', 'bottom', 'top')},
    'layers': [{
        'name': 'Aluminum', 'h': 0.008, 'theta': 0,
        'E1': E, 'E2': E, 'G12': G, 'G13': G, 'G23': G,
        'nu12': nu, 'rho': rho, 'is_visco': False,
    }],
    'load': {
        'kind': 'pulse', 'spatial': 'patch', 'amplitude': 10,
        'duration': 0.001, 'x': 0.3, 'y': 0.25,
        'patch_width': 0.06, 'patch_height': 0.05,
    },
    'simulation': {'total_time': 0.001, 'dt': 0.0001, 'calculate_stress': False},
}
if __name__ == '__main__':
    path = ROOT / 'smoke_config.json'
    save_json(config, path)
    print(f'Created {path}')

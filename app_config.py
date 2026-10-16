"""GUI-side input checks and config IO; SI units throughout."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import engine

BC_OPTIONS = {'Sharnirli (S)': 'simply_supported',
              'Qistirilgan (C)': 'clamped', 'Erkin (F)': 'free'}
BC_LOOKUP = {v: k for k, v in BC_OPTIONS.items()}


def initial_config():
    """Editable example ONLY; never start a simulation without user clicking Run."""
    config = engine.demo_config()
    config['clpt'] = {'nx': 4, 'ny': 4}
    config['simulation']['calculate_stress'] = True
    config['simulation']['record_energy'] = False
    return config


def validate(config, model='both'):
    """Validate before starting subprocess. Underlying engine checks further details."""
    if model not in ('clpt', 'fsdt', 'both'):
        raise ValueError('Model turi: CLPT, FSDT yoki ikkalasi.')
    if not isinstance(config, dict):
        raise ValueError('JSON obyekt bo\'lishi kerak.')
    cfg = engine.normalize_config(config)
    for key in ('a', 'b'):
        engine.finite(cfg[key], key, positive=True)
    for key in ('nx', 'ny'):
        value = cfg.get(key, 8)
        if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 100:
            raise ValueError(f'{key}: to\'r 2..100 oralig\'idagi butun son bo\'lsin.')
    ritz = cfg.get('clpt', {})
    for key in ('nx', 'ny'):
        value = ritz.get(key, 4)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
            raise ValueError(f'CLPT {key}: Ritz modalari 1..16 oralig\'ida bo\'lsin.')
    for key in ('kw', 'kp'):
        v = engine.finite(cfg.get(key, 0), key)
        if v < 0:
            raise ValueError(f'{key} manfiy bo\'lishi mumkin emas.')
    cfg['edges'] = engine.normalize_edges(cfg.get('edges'))
    if cfg.get('in_plane', 'fixed') not in ('fixed', 'normal', 'free'):
        raise ValueError('in_plane = fixed, normal, free')
    if not cfg.get('layers'):
        raise ValueError('Kamida bitta qatlam kerak.')
    if len(cfg['layers']) > 40:
        raise ValueError('40 tadan ko\'p qatlam kiritmang.')
    layers = [engine.OrthotropicLayer(**ly) for ly in cfg['layers']]
    load = engine.Load(**cfg['load'])
    for xy, dim in (('x', 'a'), ('y', 'b')):
        if load.spatial != 'uniform':
            value = getattr(load, xy)
            if value is None:
                value = cfg[dim]/2
            if not 0 <= value <= cfg[dim]:
                raise ValueError(f'Yuk {xy} plastinka chegarasidan tashqarida.')
    if load.spatial == 'patch':
        if load.patch_width is None or load.patch_height is None:
            raise ValueError('Kontakt eni va bo\'yi ko\'rsatilishi shart.')
        x, y = (load.x if load.x is not None else cfg['a']/2,
                load.y if load.y is not None else cfg['b']/2)
        if (x-load.patch_width/2 < 0 or x+load.patch_width/2 > cfg['a'] or
                y-load.patch_height/2 < 0 or y+load.patch_height/2 > cfg['b']):
            raise ValueError('Kontakt to\'rtburchagi plastinka ichida bo\'lsin.')
    sim = cfg['simulation']
    total = engine.finite(sim['total_time'], 'total_time', positive=True)
    dt = sim.get('dt')
    if dt is not None:
        dt = engine.finite(dt, 'dt', positive=True)
        if total/dt > 300000:
            raise ValueError('Vaqt qadamlari 300000 dan oshyapti; dt ni oshiring.')
        if model in ('clpt', 'both') and total/dt > 10000:
            raise ValueError('CLPT uchun maksimal 10000 qadam; dt ni oshiring.')
    if sim.get('memory_initial', 'instantaneous') not in ('instantaneous', 'relaxed'):
        raise ValueError('Boshlang\'ich xotira holati noto\'g\'ri.')
    if model in ('clpt', 'both') and sim.get('memory_initial', 'instantaneous') == 'relaxed' and (
       sim.get('w0_mm', 0) or sim.get('v0', 0)):
        raise ValueError('CLPT da relaksatsiyalangan boshlang\'ich xotira + boshlang\'ich siljish qo\'llanmaydi.')
    return cfg


def load_json(path):
    with Path(path).open('r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return engine.normalize_config(data)


def save_json(config, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    return p

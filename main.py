"""Desktop launcher and isolated compute worker for CLPT + FSDT + RK.

Normal launch: graphical desktop UI. --worker: invisible compute child process.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path


def run_worker(args):
    # Force off-screen figures before engine imports pyplot. No server/network needed.
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg', force=True)
    import engine
    from app_config import load_json, validate

    config = validate(load_json(args.config), args.model)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    print('Haqiqiy parametrlar yuklandi:', args.config, flush=True)
    if args.model == 'fsdt':
        engine.run_config(config, target / 'fsdt', show=False, check_dt=args.check_dt)
    elif args.model == 'clpt':
        engine.run_clpt_config(config, target / 'clpt', show=False, check_dt=args.check_dt)
    else:
        _, result_f = engine.run_config(config, target/'fsdt', show=False,
                                        check_dt=args.check_dt)
        _, result_c = engine.run_clpt_config(config, target/'clpt', show=False,
                                             check_dt=args.check_dt)
        engine.compare_models(result_f, result_c, target, show=False)
    print('YAKUNIY HOLAT: OK', flush=True)


def main():
    parser = argparse.ArgumentParser(description='CLPT + FSDT + RK Windows desktop dasturi')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--config', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--model', choices=('clpt','fsdt','both'), default='both')
    parser.add_argument('--check-dt', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        import engine
        engine.self_test_unified()
        return 0
    if args.worker:
        if not args.config or not args.output:
            parser.error('--worker uchun --config va --output kerak')
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        # A --windowed PyInstaller EXE may have sys.stdout/sys.stderr=None.
        # Always write worker output to a real log file; the GUI polls this file.
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        class LogWriter:
            def __init__(self, file, stream):
                self.file, self.stream = file, stream
            def write(self, message):
                self.file.write(message)
                self.file.flush()
                if self.stream is not None:
                    try:
                        self.stream.write(message)
                        self.stream.flush()
                    except OSError:
                        pass
                return len(message)
            def flush(self):
                self.file.flush()
            def isatty(self):
                return False
        with (output_dir/'run.log').open('w', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(LogWriter(log, original_stdout)):
                with contextlib.redirect_stderr(LogWriter(log, original_stderr)):
                    try:
                        run_worker(args)
                        return 0
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        return 1
    from gui import Application
    from tkinter import Tk
    root = Tk()
    Application(root)
    root.mainloop()
    return 0


if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        import multiprocessing
        multiprocessing.freeze_support()
    raise SystemExit(main())

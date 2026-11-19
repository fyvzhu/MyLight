"""Build/install libsumo 1.16.0 against an existing source-built SUMO library.

Run with the Unicorn interpreter after installing swig==4.2.1. Requires the
source tree (headers, build/src/config.h and bin/libsumocpp.so) in SUMO_HOME.
No project test code, test dependencies or prebuilt SUMO wheel is used.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def install(sumo, log_dir):
    sumo = sumo.resolve()
    library = sumo / 'bin/libsumocpp.so'
    for path in [library, sumo / 'src/libsumo/libsumo.i', sumo / 'build/src/config.h']:
        if not path.is_file():
            raise FileNotFoundError(f'Required source-build file missing: {path}')
    for package in ['traci', 'sumolib']:
        if importlib.metadata.version(package) != '1.16.0':
            raise RuntimeError(f'{package}==1.16.0 is required before building libsumo')
    version = subprocess.check_output([str(sumo / 'bin/sumo'), '--version'], text=True)
    if not version.splitlines()[0].endswith('Version 1.16.0'):
        raise RuntimeError(f'Expected SUMO 1.16.0, got {version.splitlines()[0]}')
    # Use this interpreter's SWIG package; do not pick another project's tool.
    import swig
    swig_root = Path(swig.__file__).resolve().parent / 'data'
    swig_binary = swig_root / 'bin/swig'
    swig_lib = next((swig_root / 'share/swig').iterdir())
    build_env = {**os.environ, 'SWIG_LIB': str(swig_lib)}
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    with tempfile.TemporaryDirectory(prefix='unicorn-libsumo-') as tmp:
        tmp = Path(tmp)
        package = tmp / 'libsumo'
        package.mkdir()
        commands += [[str(swig_binary), '-c++', '-python', '-keyword', '-DHAVE_LIBSUMOGUI',
                      '-I' + str(sumo / 'src/libsumo'), '-outdir', str(package),
                      '-o', str(tmp / 'wrapper.cpp'), str(sumo / 'src/libsumo/libsumo.i')]]
        commands += [['g++', '-std=c++11', '-shared', '-fPIC', '-O2', '-DHAVE_LIBSUMOGUI',
                      '-I' + sysconfig.get_path('include'), '-I' + str(sumo / 'src'),
                      '-I' + str(sumo / 'build/src'), str(tmp / 'wrapper.cpp'),
                      '-L' + str(library.parent), '-lsumocpp',
                      '-Wl,-rpath,' + str(library.parent), '-o', str(package / '_libsumo.so')]]
        shutil.copy2(sumo / 'tools/libsumo/__init__.py', package / '__init__.py')
        (tmp / 'setup.py').write_text(
            'from setuptools import setup, Distribution\n'
            'class BinaryDistribution(Distribution):\n'
            '    def has_ext_modules(self): return True\n'
            "setup(name='libsumo', version='1.16.0', packages=['libsumo'],\n"
            "      package_data={'libsumo': ['_libsumo.so']},\n"
            "      install_requires=['traci==1.16.0', 'sumolib==1.16.0'],\n"
            '      distclass=BinaryDistribution)\n')
        commands += [[sys.executable, 'setup.py', 'bdist_wheel']]
        with (log_dir / 'build.log').open('w') as log:
            for command in commands:
                print('Running:', command[0], flush=True)
                log.write(json.dumps(command) + '\n'); log.flush()
                subprocess.run(command, cwd=tmp, env=build_env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        wheel = next((tmp / 'dist').glob('*.whl'))
        subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, '--no-deps',
                        str(wheel)], check=True)
    check_env = {**os.environ, 'SUMO_HOME': str(sumo)}
    check = subprocess.check_output([sys.executable, '-c',
        'import libsumo; print(libsumo.__file__); print(libsumo.getVersion())'],
        env=check_env, text=True)
    print(check, flush=True)
    if "SUMO 1.16.0" not in check:
        raise RuntimeError('Installed libsumo version verification failed')
    (log_dir / 'installation.json').write_text(json.dumps({
        'python': sys.executable, 'sumo_home': str(sumo), 'library': str(library),
        'commands': commands, 'verification': check}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sumo-home', type=Path, default=os.environ.get('SUMO_HOME'))
    parser.add_argument('--log-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.sumo_home is None:
        parser.error('Set SUMO_HOME or provide --sumo-home')
    install(args.sumo_home, args.log_dir)

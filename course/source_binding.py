"""Pure file binding for policy, environment, control contract and assets.

This module does not import a simulator, policy, or inference library. The
evaluator binds its own implementation separately with evaluator_sha256.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
ACTOR_FILES=('policy.py','policy_gyro_v4.py','policy_yaw_phase_v10.py','policy_yaw_him_v25.py')

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def files_sha256(files):
    """Canonical hash of relative filename -> content hash, without paths to a PC."""
    data=json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def collect_source_binding(root=None):
    root=Path(root).resolve() if root is not None else ROOT
    required=[root/'course/runtime.py',root/'course/source_binding.py',
              *[root/'scratch_policy'/name for name in ACTOR_FILES]]
    for directory,pattern in (('tinker_policy','*.py'),('kudada_motrix','*.py'),('references/tinker','*.yaml')):
        required.extend(sorted((root/directory).glob(pattern)))
    required.extend(sorted(p for p in (root/'assets/kudada').rglob('*')
                           if p.is_file() and '__pycache__' not in p.parts
                           and p.suffix.lower() in {'.xml','.urdf','.stl','.obj','.ply','.msh','.py'}))
    versions={}
    for package in ('motrix_env_core','motrix_env_motrixsim'):
        folder=root/'vendor'/package
        pyproject=folder/'pyproject.toml'
        required.append(pyproject)
        required.extend(sorted((folder/'src').rglob('*.py')))
        text=pyproject.read_text(encoding='utf-8-sig')
        project=text.split('[project]',1)[1].split('\n[',1)[0]
        match=re.search(r'^version\s*=\s*"([^"]+)"',project,re.MULTILINE)
        if match is None: raise ValueError(f'Missing vendor version: {package}')
        versions[package]=match.group(1)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError('Missing source-binding files: '+', '.join(str(p) for p in required if not p.is_file()))
    files={p.relative_to(root).as_posix():file_sha256(p) for p in sorted(set(required))}
    return {'schema':'standing_course_source_binding_v1','files':files,
            'canonical_sha256':files_sha256(files),'vendor_versions':versions,
            'scope':'Policy loading/Actor, environment, hardware YAML, robot assets and vendored Motrix source. Evaluator identity is bound separately.'}

def validate_source_binding(value):
    """Validate stored hashes without pretending to re-execute an old evaluation."""
    if not isinstance(value,dict) or value.get('schema')!='standing_course_source_binding_v1':
        raise ValueError('Unknown source binding schema')
    files=value.get('files')
    if not isinstance(files,dict) or not files:
        raise ValueError('Source binding is empty')
    for name,digest in files.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):
            raise ValueError('Invalid source binding filename or content hash')
    if files_sha256(files)!=value.get('canonical_sha256'):
        raise ValueError('Source binding canonical hash does not match files')
    if not all(name in files for name in ('course/runtime.py','course/source_binding.py',
            'scratch_policy/policy.py','scratch_policy/policy_yaw_him_v25.py','tinker_policy/env.py',
            'kudada_motrix/env.py','references/tinker/param_gait.yaml','references/tinker/param_robot.yaml',
            'assets/kudada/kudada_lingzu.xml','vendor/motrix_env_core/pyproject.toml','vendor/motrix_env_motrixsim/pyproject.toml')):
        raise ValueError('Source binding lacks a required component')
    return value

"""Load the actual owned policy; fixed hardware IO and deterministic inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

class Policy:
    def __init__(self, checkpoint=None, onnx=None, device='cpu'):
        if (checkpoint is None) == (onnx is None):
            raise ValueError('Choose one actual checkpoint or ONNX')
        self.path = Path(checkpoint or onnx).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.identity = {'path':str(self.path), 'sha256':sha256(self.path), 'kind':'checkpoint' if checkpoint else 'onnx'}
        self.actor = self.session = None
        if checkpoint:
            import torch
            from scratch_policy.policy_yaw_him_v25 import ARCHITECTURE as HIM_ARCHITECTURE, load_actor as load_him_actor
            from scratch_policy.policy import load_actor as load_base_actor
            from scratch_policy.policy import state_digest
            data = torch.load(self.path, map_location='cpu', weights_only=False)
            architecture=data['actor_config'].get('architecture','scratch_base_mlp')
            if architecture==HIM_ARCHITECTURE:
                self.actor=load_him_actor(data).to(device).eval()
            elif architecture=='scratch_base_mlp':
                self.actor=load_base_actor(data).to(device).eval()
            else:
                raise ValueError(f'Unsupported classroom architecture: {architecture}')
            self.device = device
            self.identity.update(actor_sha256=state_digest(self.actor.state_dict()),
                                 architecture=architecture,
                                 completed_updates=data['completed_updates'])
        else:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=['CPUExecutionProvider'])
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if len(inputs)!=1 or len(outputs)!=1 or inputs[0].shape!=[1,924] or outputs[0].shape!=[1,11]:
                raise ValueError('Course deployment IO must be exactly [1,924] -> [1,11]')
            if inputs[0].type!='tensor(float)' or outputs[0].type!='tensor(float)':
                raise ValueError('Policy IO must be float32')
            self.input_name, self.output_name = inputs[0].name, outputs[0].name
            self.identity.update(onnxruntime_version=ort.__version__)

    def __call__(self, observations):
        raw = np.ascontiguousarray(observations, dtype=np.float32)
        if raw.ndim!=2 or raw.shape[1]!=924 or not np.isfinite(raw).all():
            raise ValueError('Expected finite [N,924] input')
        if self.actor is not None:
            import torch
            with torch.inference_mode():
                result = self.actor.hardware_output(torch.from_numpy(raw).to(self.device)).cpu().numpy()
        else:
            result = np.concatenate([self.session.run([self.output_name], {self.input_name:row[None]})[0] for row in raw])
        if result.shape!=(len(raw),11) or not np.isfinite(result).all():
            raise ValueError('Invalid policy output')
        return result

def export(checkpoint, output, observations=None):
    import torch
    import onnx
    import onnxruntime as ort
    from onnx import helper
    from scratch_policy.policy import HardwareActor
    from scratch_policy.export import observation_cases
    policy = Policy(checkpoint=checkpoint)
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_suffix('.staged.onnx')
    torch.onnx.export(HardwareActor(policy.actor), torch.zeros(1,924), str(staged),
                      input_names=['input'], output_names=['output'], opset_version=17, dynamo=False)
    model = onnx.load(str(staged))
    model.ir_version = 8
    helper.set_model_props(model, {'history_order':'oldest_first_newest_last', 'startup_history':'zero_fill',
        'checkpoint_sha256':policy.identity['sha256'], 'actor_sha256':policy.identity['actor_sha256'],
        'policy_origin':'independent_random_initialization', 'uses_original_onnx':'false',
        'reserved_output':'index10_zero_ignored_by_motion_ctrl'})
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(staged))
    converted = Policy(onnx=staged)
    cases = observation_cases(Path(observations) if observations else None)
    # Cover first-frame-only startup, plus explicit chronology with unequal frames.
    chronology = np.zeros((3,21,44), dtype=np.float32)
    chronology[0,-1,13] = -1.
    chronology[1,:,8] = np.linspace(-.15,.15,21)
    chronology[2,:,8] = chronology[1,::-1,8]
    cases = np.concatenate((cases, chronology.reshape(3,924)))
    maximum = 0.
    for row in cases:
        expected, actual = policy(row[None]), converted(row[None])
        maximum = max(maximum, float(np.max(np.abs(expected-actual))))
        if actual[0,10]!=0:
            raise ValueError('Reserved output must be zero')
    if maximum>=1e-4:
        raise ValueError(f'ONNX numerical parity failed: {maximum}')
    staged.replace(destination)
    result = dict(passed=True, **policy.identity, onnx=str(destination), onnx_sha256=sha256(destination),
                  input_shape=[1,924], output_shape=[1,11], dtype='float32',
                  history_frames=21, frame_width=44, history_order='oldest_first_newest_last',
                  startup_history='zero_fill_then_append_one_frame', reserved_output_zero=True,
                  parity_cases=len(cases), max_abs_error=maximum, onnxruntime_version=ort.__version__,
                  opset=17, ir_version=8, hardware_tested=False,
                  scope='Interface and numerical parity. Physical performance requires separate evaluation.')
    write_json(destination.with_suffix('.verification.json'), result)
    return result

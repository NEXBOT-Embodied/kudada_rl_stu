"""Repository-local course entry point; explicit models and reproducible outputs."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_WAIT_POLICY','PASSIVE')
os.environ.setdefault('GOMP_SPINCOUNT','0')

def source(parser,required=False):
    group=parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--checkpoint',type=Path)
    group.add_argument('--onnx',type=Path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor')
    play=commands.add_parser('play')
    play.add_argument('mode',choices=['pd','stand','walk','transition'])
    source(play)
    play.add_argument('--seconds',type=float,default=None)
    play.add_argument('--headless',action='store_true')
    play.add_argument('--video',type=Path,help='Write actual simulation replay to MP4 at 25 fps')
    play.add_argument('--kp-scale',type=float,default=4.)
    play.add_argument('--policy-kp-scale',type=float,default=None,help='策略模式的 KP 倍率，范围 [0.25, 4]；仅用于仿真教学')
    play.add_argument('--kd-scale',type=float,default=None,help='所有模式的 KD 倍率，范围 [0.25, 4]；仅用于仿真教学')
    play.add_argument('--action-scale',type=float,default=None,help='动作缩放绝对值，范围 (0, 0.5]；仅用于仿真教学')
    play.add_argument('--action-filter-rate',type=float,default=None,help='动作低通速率绝对值，范围 [5, 200]，单位 1/s；仅用于仿真教学')
    play.add_argument('--signals',type=Path,help='Save aligned action/target/position samples for one joint to CSV and an identity JSON')
    play.add_argument('--signal-joint',type=int,default=0,help='Joint index for the signal record (0-9, default 0)')
    ev=commands.add_parser('evaluate')
    source(ev)
    ev.add_argument('--output',type=Path,required=True)
    ev.add_argument('--modes',nargs='+',choices=['stand','walk','transition'],default=['stand','walk','transition'])
    ev.add_argument('--seed',type=int,default=9101)
    ev.add_argument('--conditions',type=int,default=5)
    ev.add_argument('--seconds',type=float,help='Smoke override; omit for full classroom protocol')
    ex=commands.add_parser('export')
    ex.add_argument('--checkpoint',type=Path,required=True)
    ex.add_argument('--output',type=Path,required=True)
    ex.add_argument('--observations',type=Path)
    train=commands.add_parser('train')
    train.add_argument('--config',type=Path,required=True)
    train.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.command=='play':
        if args.mode=='pd' and args.policy_kp_scale is not None:
            parser.error('--policy-kp-scale 仅用于策略模式（stand/walk/transition），不能用于 pd 模式')
        for name,lower,upper,bounds in (
            ('policy_kp_scale',.25,4.,'[0.25, 4]'),
            ('kd_scale',.25,4.,'[0.25, 4]'),
            ('action_scale',0.,.5,'(0, 0.5]'),
            ('action_filter_rate',5.,200.,'[5, 200]'),
        ):
            value=getattr(args,name)
            if value is not None and (not math.isfinite(value) or value<=0. or not lower<=value<=upper):
                parser.error(f"--{name.replace('_','-')} 必须是有限且大于 0 的数，范围为 {bounds}")
    from course.runtime import Policy,export,write_json
    if args.command=='doctor':
        import importlib
        import importlib.metadata
        packages=['numpy','torch','onnx','onnxruntime','mujoco','motrixsim','motrix_env_core','motrix_env_motrixsim']
        result={'python':sys.executable,'repository':str(ROOT),'packages':{},'missing':[]}
        for name in packages:
            try:
                module=importlib.import_module(name)
                result['packages'][name]={'version':getattr(module,'__version__','see package metadata'),'path':getattr(module,'__file__',None)}
            except Exception as error: result['missing'].append({'name':name,'error':str(error)})
        if not result['missing']:
            import torch
            from tinker_policy.env import hardware_contract
            result.update(cuda_available=torch.cuda.is_available(),cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          hardware_contract=hardware_contract())
        result['default_checkpoint_present']=(ROOT/'models/teaching_start.pt').is_file()
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['missing']: raise SystemExit(1)
    elif args.command=='train':
        from course.train import main as train_main
        forwarded=['--config',str(args.config)]
        if args.run_dir: forwarded+=['--run-dir',str(args.run_dir)]
        sys.argv=[str(ROOT/'course/train.py'),*forwarded]
        train_main()
    elif args.command=='export':
        print(json.dumps(export(args.checkpoint,args.output,args.observations),ensure_ascii=False,indent=2))
    else:
        checkpoint=args.checkpoint
        if checkpoint is None and args.onnx is None and not (args.command=='play' and args.mode=='pd'):
            checkpoint=ROOT/'models/teaching_start.pt'
        policy=None if args.command=='play' and args.mode=='pd' else Policy(checkpoint=checkpoint,onnx=args.onnx)
        if args.command=='play':
            if args.mode=='pd' and (args.checkpoint or args.onnx): parser.error('PD diagnostic has no policy; choose stand to load a model')
            from course.mujoco_play import play
            duration=args.seconds or (40. if args.mode=='transition' else 30. if args.mode=='walk' else 60.)
            if args.mode=='transition' and duration!=40.: parser.error('Transition replay duration is exactly 40 seconds')
            result=play(policy,args.mode,duration,args.headless,args.kp_scale,args.video,args.signals,args.signal_joint,
                        policy_kp_scale=args.policy_kp_scale,kd_scale=args.kd_scale,
                        action_scale=args.action_scale,action_filter_rate=args.action_filter_rate)
            if args.video:
                from course.runtime import sha256
                result['video_sha256']=sha256(args.video)
                write_json(args.video.with_suffix('.json'),result)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            from course.evaluate import evaluate
            result=evaluate(policy,args.output,args.modes,args.seed,args.conditions,args.seconds)
            print(f"Saved: {args.output/'evaluation.json'}")
            for mode,report in result['modes'].items():
                print(f"{mode}: survived {report['survived']}/{report['conditions']}; reference gates {report['reference_passed']}/{report['conditions']}")

if __name__=='__main__': main()

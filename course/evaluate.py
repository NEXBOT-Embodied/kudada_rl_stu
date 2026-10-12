"""Five paired conditions; only the first physical episode is scored."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
import time
from pathlib import Path
import numpy as np
from course.runtime import Policy, sha256, write_json
from course.source_binding import collect_source_binding

EVALUATION_SEED = 9101
PROTOCOL = {'stand':(60.,[(0.,60.,0.)]), 'walk':(30.,[(0.,30.,.2)]),
            'transition':(40.,[(0.,10.,0.),(10.,30.,.2),(30.,40.,0.)])}

def runtime_versions():
    result={'python':platform.python_version()}
    for package in ('motrixsim','motrixsim-core','motrix-env-core','motrix-env-motrixsim','numpy','torch','onnxruntime'):
        try: result[package]=version(package)
        except PackageNotFoundError: result[package]='not installed'
    return result

def summary(rows, initial, initial_yaw, mode, horizon, condition, termination):
    if not rows:
        raise RuntimeError('No physical transition captured')
    times = np.asarray([r['t'] for r in rows])
    position = np.asarray([r['position'] for r in rows])
    velocity = np.asarray([r['velocity'] for r in rows])
    gravity = np.asarray([r['gravity'] for r in rows])
    output = np.asarray([r['output'][:10] for r in rows])
    yaw = np.asarray([r['yaw'] for r in rows])
    delta = position[:,:2]-initial[:2]
    lateral = -np.sin(initial_yaw)*delta[:,0]+np.cos(initial_yaw)*delta[:,1]
    forward_speed = np.cos(initial_yaw)*velocity[:,0]+np.sin(initial_yaw)*velocity[:,1]
    planar = np.linalg.norm(velocity[:,:2], axis=1)
    roll = np.rad2deg(np.arctan2(-gravity[:,1],-gravity[:,2]))
    pitch = np.rad2deg(np.arcsin(np.clip(gravity[:,0],-1,1)))
    steady = times > 5.+1e-8
    relative_yaw = (yaw-initial_yaw+np.pi)%(2*np.pi)-np.pi
    completed = times[-1]+1e-6>=horizon and not termination
    def mean(values, mask):
        return float(np.mean(values[mask])) if mask.any() else None
    protected = any(r['protected'] for r in rows)
    outside = any(r['outside'] for r in rows)
    mean_speed = mean(planar, steady)
    result = dict(condition=condition, survived=bool(completed), duration_seconds=float(times[-1]),
        stop_reason='duration_completed' if completed else termination or 'incomplete',
        startup_seconds=5., planar_speed_startup_mean=mean(planar,~steady),
        planar_travel_startup_m=float(np.sum(planar[~steady])*.02),
        max_drift_startup_m=float(np.linalg.norm(delta[~steady],axis=1).max()) if (~steady).any() else None,
        planar_travel_total_m=float(np.sum(planar)*.02),
        planar_speed_steady_mean=mean_speed, forward_speed_steady_mean=mean(forward_speed,steady),
        max_drift_m=float(np.linalg.norm(delta,axis=1).max()), final_drift_m=float(np.linalg.norm(delta[-1])),
        max_lateral_m=float(np.abs(lateral).max()), max_yaw_deg=float(np.rad2deg(np.abs(relative_yaw).max())),
        max_roll_deg=float(np.abs(roll).max()), max_pitch_deg=float(np.abs(pitch).max()),
        minimum_height_m=float(position[:,2].min()),
        action_delta_rms=float(np.sqrt(np.mean(np.diff(output,axis=0)**2))) if len(output)>1 else None,
        action_delta_steady_rms=float(np.sqrt(np.mean(np.diff(output[steady],axis=0)**2))) if steady.sum()>1 else None,
        velocity_protection_triggered=protected, target_outside_physical_range=outside,
        max_feedback_velocity_deg_s=max(r['feedback'] for r in rows),
        max_command_velocity_deg_s=max(r['command_velocity'] for r in rows))
    if mode=='transition':
        result['segments'] = []
        for start,end,speed in PROTOCOL[mode][1]:
            mask = (times>start+2.+1e-8)&(times<=end+1e-8)
            last = np.flatnonzero(times<=start+1e-8)
            origin = position[last[-1],:2] if len(last) else initial[:2]
            allmask=(times>start+1e-8)&(times<=end+1e-8)
            result['segments'].append(dict(start_seconds=start,end_seconds=end,command_vx=speed,
                completed=bool(times[-1]+1e-6>=end),
                planar_speed_steady_mean=mean(planar,mask),forward_speed_steady_mean=mean(forward_speed,mask),
                max_drift_from_segment_start_m=float(np.linalg.norm(position[allmask,:2]-origin,axis=1).max()) if allmask.any() else None))
    gates=dict(completed=completed,no_velocity_protection=not protected,no_target_range_exceedance=not outside,
               minimum_height=result['minimum_height_m']>=.15,
               roll_pitch=max(result['max_roll_deg'],result['max_pitch_deg'])<=20.)
    if mode=='stand':
        gates.update(stationary=mean_speed is not None and mean_speed<.03,drift=result['max_drift_m']<=.25)
    elif mode=='walk':
        vx=result['forward_speed_steady_mean']
        gates.update(speed=vx is not None and .15<=vx<=.25,lateral=result['max_lateral_m']<=.30,yaw=result['max_yaw_deg']<=15.)
    result.update(reference_gates=gates,reference_passed=bool(all(gates.values())))
    return result

def evaluate_mode(policy, mode, output, seed=EVALUATION_SEED, conditions=5, seconds=None):
    from tinker_policy.env import TinkerCfg,TinkerEnv,mild_randomization
    horizon,schedule=PROTOCOL[mode]
    if seconds is not None:
        if mode=='transition' and seconds!=40.:
            raise ValueError('Transition protocol is exactly 10/20/10 seconds')
        horizon=float(seconds)
    cfg=TinkerCfg(seed=seed, fixed_command=(schedule[0][2],0.,0.), max_episode_seconds=horizon+2.,
                  command_resampling_seconds=horizon+2., add_noise=False, randomization=mild_randomization())
    env=TinkerEnv(cfg,conditions)
    started=time.perf_counter()
    env.init_state()
    initial=env.sim_data['base_pos'].copy()
    x,y,z,w=env.sim_data['base_quat'].T
    initial_yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    initial_data=dict(position=initial.tolist(),q=env.sim_data['q'].tolist(),
                      velocity_world=env.sim_data['world_lin_vel'].tolist(),
                      friction=env.physical['friction'].tolist(),
                      initial_policy_input_sha256=hashlib.sha256(env.policy_input().tobytes()).hexdigest())
    active=np.ones(conditions,dtype=bool)
    rows=[[] for _ in range(conditions)]
    reasons=[None]*conditions
    trace={name:[] for name in ('time','active','position','velocity_world','projected_gravity','yaw','output','protected','outside')}
    dt=cfg.ctrl_dt
    try:
        for step in range(round(horizon/dt)):
            t=step*dt
            for begin,end,speed in schedule:
                if begin<=t<end or mode!='transition':
                    env.commands[:]=[speed,0.,0.]
                    break
            mask=active.copy()
            result=policy(env.policy_input())
            env.step(np.clip(result[:,:10],-20.,20.))
            tr=env.transition_data
            for i in np.flatnonzero(mask):
                row=dict(t=(step+1)*dt,position=tr['position'][i].tolist(),velocity=tr['velocity_world'][i].tolist(),
                    gravity=tr['projected_gravity'][i].tolist(),yaw=float(tr['yaw'][i]),output=result[i].tolist(),
                    protected=bool(tr['velocity_protection_triggered'][i]), outside=bool(tr['target_outside_physical_range'][i]),
                    feedback=float(tr['max_feedback_velocity_deg_s'][i]),command_velocity=float(tr['max_command_velocity_deg_s'][i]))
                rows[i].append(row)
                tilt=np.linalg.norm(tr['projected_gravity'][i,:2])
                if row['protected']:
                    reasons[i]='joint_velocity_protection'
                elif tr['terminated'][i]:
                    failure=[name for name,values in tr['termination_reasons'].items() if values[i]]
                    reasons[i]='physical_failure:'+','.join(failure)
                elif tilt>np.sin(np.deg2rad(45.)):
                    reasons[i]='attitude_45deg'
                elif tr['truncated'][i]:
                    reasons[i]='unexpected_timeout'
                if reasons[i]:
                    active[i]=False
            trace['time'].append((step+1)*dt)
            trace['active'].append(mask)
            for target,key in [('position','position'),('velocity_world','velocity_world'),('projected_gravity','projected_gravity'),
                               ('yaw','yaw'),('protected','velocity_protection_triggered'),('outside','target_outside_physical_range')]:
                trace[target].append(np.array(tr[key],copy=True))
            trace['output'].append(result.copy())
            if (step+1)%250==0:
                print(f'EVAL {mode} t={(step+1)*dt:.0f}s alive={active.sum()}/{conditions}',flush=True)
            if not active.any():
                break
    finally:
        close=getattr(env,'close',None)
        if callable(close): close()
    path=Path(output)
    path.mkdir(parents=True,exist_ok=True)
    trace_file=path/f'{mode}.trace.npz'
    np.savez_compressed(trace_file,**{k:np.asarray(v) for k,v in trace.items()},initial_position=initial,initial_yaw=initial_yaw)
    items=[summary(rows[i],initial[i],initial_yaw[i],mode,horizon,f'seed{seed}_row{i}',reasons[i]) for i in range(conditions)]
    report=dict(mode=mode,engine='MotrixLab/MotrixSim',seed=seed,conditions=conditions,requested_seconds=horizon,
                control_dt=dt,physics_dt=cfg.sim.dt,schedule=schedule,steady_start_seconds=5.,initial_conditions=initial_data,
                policy=policy.identity,rows=items,survived=sum(r['survived'] for r in items),
                reference_passed=sum(r['reference_passed'] for r in items),trace=str(trace_file),trace_sha256=sha256(trace_file),
                evaluator_sha256=sha256(__file__),wall_seconds=time.perf_counter()-started,
                runtime_versions=runtime_versions(),
                scoring='First physical episode only; failure step retained; no post-reset or fallen-static averaging.',
                hardware_tested=False)
    write_json(path/f'{mode}.json',report)
    return report

def evaluate(policy, output, modes=('stand','walk','transition'), seed=EVALUATION_SEED, conditions=5, seconds=None):
    binding=collect_source_binding()
    if conditions<5:
        print('SMOKE ONLY: fewer than 5 conditions does not satisfy classroom evaluation.',flush=True)
    results={mode:evaluate_mode(policy,mode,output,seed,conditions,seconds) for mode in modes}
    report=dict(schema='standing_course_eval_v1',policy=policy.identity,protocol_seed=seed,conditions=conditions,
                modes=results,complete_classroom_protocol=conditions>=5 and seconds is None and set(modes)==set(PROTOCOL),
                source_binding=binding,binding_capture='in_report')
    write_json(Path(output)/'evaluation.json',report)
    return report

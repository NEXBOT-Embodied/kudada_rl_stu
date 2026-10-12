"""MuJoCo visual replay of the actual policy and inspected controller contract."""
from __future__ import annotations
from contextlib import nullcontext,ExitStack
from pathlib import Path
import time
import numpy as np
from course.evaluate import summary

def play(policy,mode,seconds=60.,headless=False,kp_scale=4.,video=None):
    import mujoco
    from kudada_motrix.constants import MODEL,JOINT_NAMES,FEET
    from tinker_policy.env import hardware_contract
    contract=hardware_contract()
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    model.opt.timestep=.001
    model.vis.global_.offwidth=768
    model.vis.global_.offheight=576
    data=mujoco.MjData(model)
    joint_ids=np.array([model.joint(n).id for n in JOINT_NAMES])
    qids=model.jnt_qposadr[joint_ids]
    vids=model.jnt_dofadr[joint_ids]
    aids=np.array([model.actuator(n).id for n in JOINT_NAMES])
    base=model.body('base_link').id
    bodies=[base,*[model.body(n).id for n in FEET]]
    root=model.joint('floating_base_joint').id
    qroot,vroot=int(model.jnt_qposadr[root]),int(model.jnt_dofadr[root])
    default=np.asarray(contract['default_q'],dtype=np.float32)
    data.qpos[qroot:qroot+7]=[0.,0.,.28,1.,0.,0.,0.]
    data.qpos[qids]=default
    mujoco.mj_forward(model,data)
    initial=data.xpos[base].copy()
    kp=np.asarray(contract['kp'])*(kp_scale if mode=='pd' else 1.)
    kd=np.asarray(contract['kd'])
    gains=np.asarray(contract['action_gains'])
    torque_limits=np.asarray(contract['torque_limits'])
    ranges=np.asarray(contract['target_ranges'])
    history=np.zeros((21,44),dtype=np.float32)
    command=np.zeros(3,dtype=np.float32)
    action=np.zeros(10,dtype=np.float32)
    filtered=np.zeros(10,dtype=np.float32)
    command_velocity=np.zeros(10)
    previous_target=default.copy()
    phase=0.
    velocity_timer=0.
    rows=[]
    reason=None
    if not headless:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(model,data)
        viewer.cam.distance=1.25
        viewer.cam.elevation=-15.
        viewer.cam.azimuth=125.
        context=viewer
    else:
        viewer=None
        context=nullcontext()
    tick=time.perf_counter()
    print('ZERO-ACTION PD diagnostic, no learned policy.' if mode=='pd' else f'Actual {mode} policy: {policy.identity}',flush=True)
    with ExitStack() as resources:
        resources.enter_context(context)
        renderer=writer=None
        if video:
            import imageio.v2 as imageio
            video=Path(video)
            video.parent.mkdir(parents=True,exist_ok=True)
            renderer=mujoco.Renderer(model,height=576,width=768)
            resources.callback(renderer.close)
            writer=imageio.get_writer(video,fps=25,codec='libx264')
            resources.callback(writer.close)
            camera=mujoco.MjvCamera()
            camera.distance=1.25
            camera.elevation=-15.
            camera.azimuth=125.
        for step in range(round(seconds/.02)):
            if viewer and not viewer.is_running():
                reason='viewer_closed'
                break
            speed=.2 if mode=='walk' or (mode=='transition' and 10.<=data.time<30.) else 0.
            command+=contract['command_filter_alpha']*(np.array([speed,0.,0.])-command)
            advanced=(phase+.02/contract['gait_period'])%1.
            phase=0. if np.all(np.abs(command)<.1) and advanced<.1 else advanced
            rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,data.qpos[qroot+3:qroot+7])
            gravity=rotation.reshape(3,3).T@np.array([0.,0.,-1.])
            frame=np.concatenate((np.zeros(3),command*np.array(contract['command_scale']),
                [np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)],data.qvel[vroot+3:vroot+6]*.25,gravity,
                data.qpos[qids]-default,data.qvel[vids]*.05,action)).astype(np.float32)
            history[:-1]=history[1:].copy()
            history[-1]=np.clip(frame,-100.,100.)
            output=np.zeros(11,dtype=np.float32) if mode=='pd' else policy(history.reshape(1,924))[0]
            action=np.clip(output[:10],-20.,20.)
            protected=outside=False
            maxfeedback=maxcommand=0.
            for _ in range(20):
                filtered+=contract['action_filter_rate']*.001*(action-filtered)
                target=np.clip(default+contract['action_scale']*gains*filtered,ranges[:,0],ranges[:,1])
                rawspeed=np.abs(target-previous_target)*57.3/.001
                command_velocity+=.001/.101*(rawspeed-command_velocity)
                previous_target[:]=target
                outside|=bool(np.any((target<model.jnt_range[joint_ids,0])|(target>model.jnt_range[joint_ids,1])))
                data.ctrl[aids]=np.clip(kp*(target-data.qpos[qids])-kd*data.qvel[vids],-torque_limits,torque_limits)
                mujoco.mj_step(model,data)
                feedback=np.max(np.abs(data.qvel[vids])*57.3)
                maxfeedback=max(maxfeedback,float(feedback))
                maxcommand=max(maxcommand,float(command_velocity.max()))
                exceeded=feedback>1000. or command_velocity.max()>1000.
                velocity_timer=velocity_timer+.001 if exceeded else 0.
                protected|=velocity_timer>.03
            mujoco.mj_forward(model,data)
            position=data.xpos[base].copy()
            mujoco.mju_quat2Mat(rotation,data.qpos[qroot+3:qroot+7])
            gravity=rotation.reshape(3,3).T@np.array([0.,0.,-1.])
            w,x,y,z=data.qpos[qroot+3:qroot+7]
            yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            contact_base=False
            for contact in data.contact:
                if model.geom_bodyid[contact.geom1]==base or model.geom_bodyid[contact.geom2]==base:
                    contact_base=True
            rows.append(dict(t=float(data.time),position=position.tolist(),velocity=data.qvel[vroot:vroot+3].tolist(),
                gravity=gravity.tolist(),yaw=float(yaw),output=output.tolist(),protected=protected,outside=outside,
                feedback=maxfeedback,command_velocity=maxcommand))
            if protected: reason='joint_velocity_protection'
            elif position[2]<.15 or np.linalg.norm(gravity[:2])>.8 or contact_base: reason='physical_failure'
            elif not np.isfinite(data.qpos).all(): reason='nonfinite_physics'
            if viewer:
                viewer.cam.lookat[:]=position
                viewer.sync()
                tick+=.02
                delay=tick-time.perf_counter()
                if delay>0: time.sleep(delay)
                else: tick=time.perf_counter()
            if writer and step%2==0:
                camera.lookat[:]=position
                renderer.update_scene(data,camera=camera)
                writer.append_data(renderer.render())
            if reason: break
    if not rows: return {'stop_reason':reason,'survived':False}
    result=summary(rows,initial,0.,'stand' if mode=='pd' else mode,seconds,'nominal_mujoco',reason)
    result.update(engine='MuJoCo',policy=None if policy is None else policy.identity,
                  diagnostic_kp_scale=kp_scale if mode=='pd' else None,hardware_tested=False,
                  validation_scope='Visual replay; classroom paired evaluation uses MotrixLab.')
    if video: result.update(video=str(video),video_fps=25)
    return result

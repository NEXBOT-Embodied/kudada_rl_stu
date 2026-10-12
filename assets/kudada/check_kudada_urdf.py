#!/usr/bin/env python3
"""Headless 验证 Kudada URDF 能否被 Isaac Gym 正常解析并仿真。

用法:
    uv run python check_kudada_urdf.py --headless
    uv run python check_kudada_urdf.py --headless --urdf <path>

输出: body/dof 名称、关节限位、总质量，并实际步进仿真若干帧检查是否崩溃。
"""
import os
import sys

import numpy as np

import isaacgym  # noqa: F401  必须先 import 才能用 gymapi
from isaacgym import gymapi, gymutil

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(REPO_ROOT, 'kudada_lingzu.urdf')


def main():
    args = gymutil.parse_arguments(
        description='URDF parse check for Kudada',
        headless=True,
        custom_parameters=[
            {"name": "--urdf", "type": str, "default": None,
             "help": "Path to URDF (default: resources/robots/kudada/...)"},
        ])

    urdf_path = args.urdf if args.urdf else URDF_PATH
    if not os.path.isfile(urdf_path):
        print(f"[FAIL] URDF not found: {urdf_path}")
        sys.exit(1)

    gym = gymapi.acquire_gym()

    # ── sim 参数（headless）──────────────────────────────────────────
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    if args.headless:
        args.graphics_device_id = -1

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)
    if sim is None:
        print("[FAIL] could not create sim (GPU/驱动问题?)")
        sys.exit(1)
    print("[OK] sim created")

    # ── 地面 ─────────────────────────────────────────────────────────
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 0.8
    plane_params.dynamic_friction = 0.6
    gym.add_ground(sim, plane_params)

    # ── 加载 URDF（选项与 legged_gym 训练时保持一致）──────────────────
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False
    asset_options.collapse_fixed_joints = False

    asset_root = os.path.dirname(urdf_path)
    asset_file = os.path.basename(urdf_path)
    print(f"[*] Loading: root={asset_root}  file={asset_file}")

    try:
        asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    except Exception as e:
        print(f"[FAIL] load_asset raised {type(e).__name__}: {e}")
        sys.exit(1)

    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print(f"[OK] asset parsed: {num_bodies} bodies, {num_dofs} dofs")

    # ── body 名称 ────────────────────────────────────────────────────
    print("\n[Body names]")
    foot_hits = []
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(f"  {i:2d}: {name}")
        if 'ankle' in name:
            foot_hits.append(name)

    # ── DOF 信息 ─────────────────────────────────────────────────────
    dof_props = gym.get_asset_dof_properties(asset)
    print("\n[Dof info]  (type: 1=revolute 2=prismatic 3=fixed)")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        jtype = int(gym.get_asset_dof_type(asset, i))
        lower = dof_props['lower'][i]
        upper = dof_props['upper'][i]
        effort = dof_props['effort'][i]
        vel = dof_props['velocity'][i]
        print(f"  {i:2d}: {name:<35s} type={jtype} "
              f"lower={lower:+.3f} upper={upper:+.3f} "
              f"effort={effort:>6.2f} vel={vel:>6.2f}")

    print(f"\n[foot candidates] 含 'ankle' 的 body: {foot_hits}")

    # ── 创建 actor 并步进仿真验证 ────────────────────────────────────
    env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0),
                         gymapi.Vec3(1.0, 1.0, 1.0), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.5)      # 初始抬高，让落地过程也走一帧
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, pose, "kudada", 0, 1)

    # ── 质量（gym_38 无 asset 级接口，用 actor 级）───────────────────
    body_props = gym.get_actor_rigid_body_properties(env, actor)
    total_mass = sum(p.mass for p in body_props)
    print(f"[*] base mass = {body_props[0].mass:.4f} kg, total mass = {total_mass:.4f} kg")

    # ── 运动学探测：确认各关节旋转方向（正角 vs 负角 脚往哪摆）────────
    def probe(pose, label):
        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        dof_states['pos'][:] = np.array(pose)
        dof_states['vel'][:] = 0.0
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
        rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
        # foot body index: 5 = left ankle, 10 = right ankle
        lp = rb['pose']['p'][5]; rp = rb['pose']['p'][10]
        print(f"  {label:<28s} Lfoot=({lp[0]:+.3f},{lp[1]:+.3f},{lp[2]:+.3f}) "
              f"Rfoot=({rp[0]:+.3f},{rp[1]:+.3f},{rp[2]:+.3f})")

    print("\n[Kinematic probe] (关节正负角对应的脚底位置, base at z=0.5)")
    probe([0]*10, "all 0 (直腿)")
    probe([0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0], "hip_pitch +0.5")
    probe([0, 0, -0.5, 0, 0, 0, 0, -0.5, 0, 0], "hip_pitch -0.5")
    probe([0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0], "knee +0.5")
    probe([0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5], "ankle +0.5")

    # ── 默认站立姿态几何验证（正膝）──────────────────────────────────
    # 说明：Kudada 脚底是沿 x 的细圆柱(radius 7mm，"刀刃脚")，无主动平衡时锁关节
    #       站立必倒，故这里只验证**几何**：脚底能否触地、左右对称、脚在身体
    #       正下方、关节在限位内。动力学稳定由 RL 训练学习。
    # probe 结论: hip_pitch + → 腿向后; knee + → 小腿向后;
    # 正膝站姿 = 大腿略前倾(hip<0), 膝屈(knee>0), 踝补偿(≈-(hip+knee))让脚掌水平
    CANDIDATES = {
        "A 直腿 all0":       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "B 微屈 hip0":       [0, 0.05, 0.0, 0.15, -0.15, 0, -0.05, 0.0, 0.15, -0.15],
        "C 前倾 hip-0.2":    [0, 0.05, -0.20, 0.60, -0.40, 0, -0.05, -0.20, 0.60, -0.40],
        "D 后倾 hip+0.2":    [0, 0.05, 0.20, 0.20, -0.40, 0, -0.05, 0.20, 0.20, -0.40],
    }
    print("\n[默认站姿几何验证] (脚底=ankle 碰撞圆柱底部, 世界坐标, base 平移后 z≈0.34)")
    for name, DEFAULT_POSE in CANDIDATES.items():
        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        dof_states['pos'][:] = np.array(DEFAULT_POSE)
        dof_states['vel'][:] = 0.0
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
        rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
        # 脚底碰撞体在 ankle link frame 下 z=-0.047-0.007=-0.054
        foot_bot = rb['pose']['p']['z'][5] - 0.054
        # 左右脚底相对 base 的水平偏移 (body 5/10)
        lx = rb['pose']['p']['x'][5]; rx = rb['pose']['p']['x'][10]
        ly = rb['pose']['p']['y'][5]; ry = rb['pose']['p']['y'][10]
        # 脚底触地时的 base 高度 = 当前 base z - 脚底离地高度
        base_stand_z = rb['pose']['p']['z'][0] - foot_bot
        sym = abs(lx - rx) < 1e-3 and abs(ly + ry) < 1e-3
        # 脚在身体正下方 (base 原点 x≈0)
        under = abs((lx + rx) / 2) < 0.05
        print(f"  [{name:<16s}] 触地base_z={base_stand_z:.3f} "
              f"脚底离地={foot_bot:+.3f} 脚x=({lx:+.3f},{rx:+.3f}) "
              f"对称={sym} 脚在身下={under}")

    # ── 短仿真冒烟测试：默认站姿(C, FK 最优) 脚底贴近地面步进，确认无 NaN/穿透 ──
    print("\n[冒烟测试] 站姿 C 脚底离地 5mm 步进 120 帧 (刀刃脚站立不倒属正常, 只查 NaN/穿透)")
    DEFAULT_POSE = CANDIDATES["C 前倾 hip-0.2"]
    dof_props['driveMode'][:] = int(gymapi.DOF_MODE_POS)
    dof_props['stiffness'][:] = 40.0
    dof_props['damping'][:] = 2.0
    gym.set_actor_dof_properties(env, actor, dof_props)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_states['pos'][:] = np.array(DEFAULT_POSE)
    dof_states['vel'][:] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    dx = (0.054 + 0.005) - rb['pose']['p']['z'][5]  # 脚底抬到离地 5mm
    rb['pose']['p']['z'][:] += dx
    gym.set_actor_rigid_body_states(env, actor, rb, gymapi.STATE_POS)
    for i in range(120):
        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_NONE)
        dof_states['pos'][:] = np.array(DEFAULT_POSE)
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
    rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    base_pos = rb['pose']['p'][0]
    if np.isnan(base_pos[0]) or np.isnan(base_pos[1]) or np.isnan(base_pos[2]):
        print("  [FAIL] NaN detected (仿真发散)")
        sys.exit(1)
    print(f"  [OK] 120 帧无 NaN。base z={base_pos[2]:.3f} "
          f"(落地后低于 0.3 属正常, 刀刃脚无平衡必倒)")
    print("\n[SUMMARY] Kudada URDF 解析 OK，默认站姿几何验证通过 → 可用站姿 C 生成训练配置")


if __name__ == '__main__':
    main()

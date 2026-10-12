# Tinker deployment binary contract audit

Audit is read-only. It does not execute ARM64 code, alter original binaries or install dependencies into either training environment. Dependencies are loaded from `reports/tinker_onnx/inspection_deps`.

`motion_ctrl/control_task` SHA256: `01379eb81e77337e95852ae6d8d4532b5f3ac01a82300049e3d252cae02178bb`.

The ELF includes application function symbols and file names. The available DWARF compilation units belong only to `/workspace/control_task/qpOASES/src/*.cpp`; no application structure/member definitions or source line information were found. Relevant application file symbols are `rl_tinker_lingzu_onnx.cpp`, `hardware_interface.cpp`, `motor_data_convert.cpp`, `robot_param.cpp`, `locomotion_sfm.cpp`.

## Confirmed contract

| Item | Confirmed behavior | Evidence addresses in control_task |
|---|---|---|
| Inference entry | `Gait_RL_Update_2(float dt)`, symbol `_Z16Gait_RL_Update_2f` | `0xc1188`, size 7396 bytes |
| Frame/history | 44 floats × 21 frames = 924; chronological oldest first, current frame appended last | `num_single_obs=44` at `0x24e588`, `frame_stack=21` at `0x24e58c`; shift/append `0xc2970–0xc2b88` |
| Initial history | All 924 floats zeroed on first model activation; no initial-frame repetition | `0xc65b8–0xc6654` |
| Reserved obs | obs[0:3] always zero | `0xc1c24–0xc1c28` |
| Command obs | obs[3]=cmd_x×cmd_vel_scale, obs[4]=cmd_y×same scale, obs[5]=cmd_rate×0.25. YAML cmd_vel_scale is the single linear command observation multiplier, currently 2.0 | scale read at `0x85694–0x856b0`, applied `0xc1bd4–0xc1c30` |
| Command smoothing | On inference updates, command ← 0.97×previous + 0.03×target; target magnitudes below .01 become zero first | `smooth_rc=.03` at `0x24e584`, `dead_zone_rc=.01` at `0x24e580`; `0xc1858–0xc193c` |
| Gait phase | phase←fmod(phase+elapsed_inference_time/rl_gait_duty,1). If all filtered command magnitudes <.1 and wrapped phase<.1, phase is reset0, sine0/cosine1. Output[10] does not control phase | phase update `0xc1910–0xc1954`; freeze `0xc28b8–0xc2900`; sine/cosine `0xc1978–0xc19c4` |
| Policy timing | Control thread schedules a 1 ms loop and passes measured dt; inference timer triggers when timer > onnx_run_dt. Current YAML nominal .02 s, actual interval includes floating-point accumulation and loop jitter | thread `0x76f10`, nanosecond increment `0x76f8c–0x76fd4`; gate `0xc11bc–0xc126c` |
| Joint order | Motor indices [0,1,2,3,4,7,8,9,10,11]: first five left, second five right | id_list at `0x24e540`; embedded interface log docs; q loads `0xc19cc–0xc1aec` |
| Joint obs | Position in degrees /57.3 minus default angles in rad; velocity degrees/sec /57.3 ×.05 | `0xc1c44–0xc1ebc` |
| IMU obs | Supplied vmc rates are ordered pitch,roll,yaw; policy omega = [-roll,pitch,yaw]/57.3 ×.25. Gravity Euler conversion receives [-roll,pitch,-yaw]/57.3, then adds configured x/y gravity bias | `0xc1af4–0xc1b80`, gravity bias `0xc1be4–0xc1c40` |
| Previous action obs | Previous clipped raw ONNX action, before low-pass and action gains | `0xc1ec0–0xc1ee4` |
| ONNX output | Only output[0:10] consumed. Output[10] classifier result is ignored in this binary; no gait selector state feedback | output buffer loads at `0xc2bec–0xc2c6c`, offsets 0,8,0x10,0x18,0x20, never offset0x28 |
| Action clamp | Each of ten outputs clipped to [-20,20] | `0xc2cb0–0xc2d54` |
| Action filter | alpha=50×dt for 0≤dt≤.005; alpha=.25 for dt>.005; alpha=0 for negative dt. Runs at each control update | `0xc1284–0xc1348`, capped branch `0xc1734–0xc173c` |
| Gains | [1.75,1.0625,1.0625,1.0625,1.4] repeated both legs; optional 10-entry YAML override | action_scale_gains at `0x24e518`; override loader `0xc4bcc–0xc4e74` |
| Target decode | q_target_rad = default + filtered_action×action_scale×gain; ×57.3 produces motor degrees; finite checks and per-joint configured target limits follow | `0xc134c–0xc170c`, limit branches `0xc211c–0xc2770` |
| Default rad | [0,-.05,-.23,.3,-.14,0,.05,-.23,.3,-.14] | provided YAML `def_act0:4`, `def_act7:11` |
| RL PD | Nominal active RL KP [8,16,16,16,10]×2, KD [.7]×10. Active RL sets joint stiffness arrays to1.0, overriding stand-mode stiffness2.0. KP/KD also multiplied by global soft weight, normal value1.0 | YAML KP stores leg+0x604 at `0x82970–0x8298c`, KD leg+0x6a4 at `0x82d10–0x82d2c`; RL stiffness1 `0xc62b4–0xc62f4`; transmission `0x8137c–0x815b4`; soft weight reset1 `0xb6b30–0xb6b8c` |
| Velocity protection | A shared any-joint timer accumulates while feedback magnitude>1000 deg/s or low-pass target command speed>1000 deg/s; continuously for>.03 s while motors enabled and gait_ww==2 triggers SAFE_HOLD, keeping motors on | YAML loads `0x8575c–0x857e8`; safe check `0xb6fd8–0xb71cc`; SAFE_HOLD return2 `0xb70c0–0xb70f4` |
| Command speed protection filter | command_speed_raw=abs(q_set_deg−previous_q_set_deg)/dt; filtered speed←previous+dt/(.1+dt)×(raw−previous). safe_check clamps dt to [0,.0055] | `0xb6bb4–0xb6bd8`, `0xb6fc8–0xb7064` |
| Fall protection | Pitch attitude deviation>35° or roll>25° continuously>.02 s, relative to stored target attitude, triggers protection outside stand/safe-hold state | `0xb6bf0–0xb6db8`; .02 constant at `0x1f8860` |

For the simulation, body-frame omega and projected gravity should represent the physical robot's policy coordinates. The observed roll sign change is the hardware sensor mapping, not justification for negating arbitrary simulator body-frame axes a second time.

The policy clock is command-driven, not ONNX gait-ID-driven. Exporting an [1,11] output with an unused constant eleventh float is compatible with this inspected controller; preserving the original classifier is optional for this binary. Other controller versions might use it differently.

## ONNX loader

`motion_ctrl/lib_files/liblite_arm64.so` exports `Onnx::onnxClass` methods. Controller calls `init(model_path, "tinker_lingzu")` at `0xc66f8–0xc6728`.

The loader implementation at `0x4560` obtains first session input/output names dynamically from ONNX Runtime, using OrtApi SessionGetInputName (index36, table offset0x120) at `0x467c–0x46bc` and SessionGetOutputName (index37, offset0x128) at `0x46f0–0x4724`. Thus "tinker_lingzu" is a stored model name, not a hardcoded tensor name. Preserve existing tensor names when exporting anyway, as this has no cost. It obtains tensor dimensions/types from the model and constructs CPU float tensors. Keep one float32 input containing924 elements and one float32 output containing11 elements.

## Remaining limits

- Physical IMU mounting, encoder calibration/zero positions, actuator command signs, and actual robot communication are not established by simulation. `param_motors.yaml` contains per-leg motor direction flags; these are applied below the policy contract.
- Normal RL gains are established from controller storage/transmission. The full motor driver transfer function is outside this control_task audit.
- The low-pass command state and phase may persist across RL reactivation; only first model load definitely zeroes history/actions. No policy-driven gait-ID state exists in the inspected inference routine.
- At startup or reactivation, phase is stored in BSS and first initialized to0 by process startup; no general reset of phase was found in Gait_RL_Active. Training should include zero history startup and command switches.
- Actual loop dt is measured, nominal control1ms and inference20ms. Bounded timing variation should be tested rather than assuming exact scheduling on hardware.

Final simulation evaluation should check actual feedback joint velocities, the low-pass derivative of the *clamped target positions* and continuous .03s overrun timer at each physical control tick; raw action differences are not this protection signal. Also record clipped target angles, torque saturation, finite input/output, and pitch/roll deviations. These are concrete controller hooks, not speculative new restrictions.

## Reproduce

From project root:

```powershell
MotrixLab/.venv/Scripts/python.exe -X utf8 kudada_rl_motrix/tools/audit_tinker_binary.py --pattern '^_Z[^N]|^vmc_all$|^leg_motor_all$' --output reports/tinker_retrain/contract_audit/app
MotrixLab/.venv/Scripts/python.exe -X utf8 kudada_rl_motrix/tools/audit_tinker_binary.py --binary motion_ctrl/lib_files/liblite_arm64.so --pattern 'Onnx|onnx|tinker|input|output' --output reports/tinker_retrain/contract_audit/onnx_loader
MotrixLab/.venv/Scripts/python.exe -X utf8 reports/tinker_retrain/contract_audit/audit_dwarf.py
```

Each binary audit emits symbol addresses/data, matching embedded strings, and annotated AArch64 disassembly. The conclusions above are manual static control/data-flow analysis, not equivalence proof or hardware validation.

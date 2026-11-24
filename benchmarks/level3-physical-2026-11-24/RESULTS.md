# Level 3 物理约束 MPC：2026-11-24 本地验证

冻结 PPO 权重，保留多门联合路线；增加空气阻力补偿、垂直推力偏差估计、MPC 内部门框修正与单电机余量，并使用 SQP_RTI 提高实时性。

先用 64 条独立赛道选择版本，再固定参数，用另外 32 条新赛道最终复核。每条赛道重复三种条件，不能计作 96 条独立赛道。

| 最终复核条件 | 旧版完赛 | 新版完赛 | 旧版成功圈中位数 | 新版成功圈中位数 | 共同完赛配对提速中位数 |
|---|---:|---:|---:|---:|---:|
| 标准课程 | 30/32 (93.8%) | 32/32 (100.0%) | 9.92 s | 9.24 s | +0.44 s |
| 固定额外延迟 + 推力下降/噪声 | 25/32 (78.1%) | 25/32 (78.1%) | 10.02 s | 9.12 s | +0.54 s |
| 变化额外延迟 + 推力下降/噪声 | 30/32 (93.8%) | 28/32 (87.5%) | 9.78 s | 9.37 s | +0.49 s |

成功圈中位数来自各自成功样本；配对提速仅比较两版均成功的相同赛道。样本数、95% 区间、失败/挽回种子见 report.json。

| 串行超时复核（另 10 条赛道） | 旧版完赛 | 新版完赛 | 旧版控制 P99 | 新版控制 P99 |
|---|---:|---:|---:|---:|
| one | 7/10 | 7/10 | 19.98 ms | 13.31 ms |
| mixed | 9/10 | 8/10 | 16.34 ms | 12.70 ms |
| nominal | 8/10 | 9/10 | 15.83 ms | 12.73 ms |

串行压力测试是否全部达到本轮完赛/姿态门槛：否。新入口保留为可选仿真实验版本，默认入口保持不变。

| 同 10 条赛道、不计超时的对照 | 旧版完赛 | 新版完赛 |
|---|---:|---:|
| one | 10/10 | 6/10 |
| mixed | 9/10 | 8/10 |
| nominal | 8/10 | 9/10 |

超时复核在计算超过 20ms 时继续推进物理仿真，保持旧指令或交付已排队指令。测试串行运行，计时来自本机，不能替代飞机平台实测。

复现中的 175ms 峰值有约 170ms 来自 Python 第二代垃圾回收，MPC 求解约 2.6ms。诊断包装前后指令哈希完全一致。固定延迟同赛道对照中新版普通评估只完赛 6/10，因此不能把所有失败归因于计算超时；运行时尖峰和延迟跟踪都还需要改进。

核心文件：

- lsy_drone_racing/control/level3_physical_controller.py：可单独选择的新控制器。
- lsy_drone_racing/control/attitude_mpc.py：可选模型、约束与推力参考映射接口，保留旧版默认行为。
- lsy_drone_racing/control/level3_delay_controller.py：将同一扰动估计传入延迟预测。
- config/level3_mpc_physical.toml：新的可选入口。
- tests/unit/test_mpc_physical.py：推力偏差、单电机余量、首帧/未来姿态约束检查。

代码整理前后，100 组符号模型比较完全一致；96 圈新控制器重放、24 圈旧版重放的逐帧比较见 port-equivalence.json。

在 WSL 仓库根目录运行仿真：

```bash
export PYTHONPATH="$PWD" ACADOS_SOURCE_DIR="$PWD/acados"
export LD_LIBRARY_PATH="$PWD/acados/lib:$PWD/.pixi/envs/default/lib"
.pixi/envs/default/bin/python scripts/sim.py --config level3_mpc_physical.toml --render=True
```

复现批量测试可使用本目录的 record.py：位置参数依次为配置文件路径、全新输出 CSV 路径、起始种子和数量。使用 --link nominal/one/mixed；压力条件另加 --noise --thrust-scale .98 --thrust-end .90。--wall-clock 按实时节奏运行；--deadline-aware 额外在计算超时时推进物理仿真。

5 秒级整圈目标尚未达到。未进行实机飞行或硬件安全认证；Handout、课程模型参数和工程裕度分别记录在 report.json。最终复核样本仍较少，不能把估计完赛率当作保证。

中途额度限制使工具调用中断；已启动验证仍完成运行，恢复后按用户指示补完复核。未提交或推送 Git。

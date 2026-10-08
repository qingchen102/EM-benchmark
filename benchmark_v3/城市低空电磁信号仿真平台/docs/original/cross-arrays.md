# 8+8 十字接收阵及空间元数据

2026-09-14：根据用户最新要求，默认接收阵列从四通道线阵升级为水平 8＋垂直 8 的十字阵，16 个不同阵元。中途讨论的 2×2 面阵仍作为可选 rectangular 类型支持，不作为默认。

## 定义

- 本地阵面为 yz，名义正面法向 +x。
- 水平臂通道 0–7：沿本地 +y，位置由负到正。
- 垂直臂通道 8–15：沿本地 -z，位置由上到下。
- 两臂均为偶数阵元，不在中心放阵元，不存在共享中心通道。
- 默认水平／垂直间距均 0.5 波长。2.4 GHz 时约 6.246 cm；每臂端到端长 43.720 cm。
- 支持更改两臂偶数阵元数和独立间距，总阵元数上限 16。rectangular 类型可配置行列数。

配置：

```json
{
  "geometry_type": "cross",
  "horizontal_elements": 8,
  "vertical_elements": 8,
  "horizontal_spacing_wavelengths": 0.5,
  "vertical_spacing_wavelengths": 0.5,
  "pattern": "iso",
  "polarization": "V"
}
```

每个接收站自己的 `position_m` 是阵列中心位置，`orientation_rad` 为 yaw、pitch、roll，局部到全局旋转为 Rz(yaw)Ry(pitch)Rx(roll)。二者原先已经记录，本次将几何关联明确到每个通道。

## 公开记录

在 `public/observations.json` 中每个接收站保存：

| 字段 | 含义 |
|---|---|
| coordinate_frame | 右手场景笛卡尔坐标，m，+z 向上；不冒充地理经纬度 |
| array_pose.center_world_m | 阵列中心全局坐标 |
| array_pose.orientation_rad | yaw/pitch/roll，rad |
| array_pose.rotation_local_to_world | 3×3 旋转矩阵 |
| array_pose.normal_world | 全局阵面正面单位法向 |
| array_geometry | 阵型、阵元数、间距、通道到臂／索引映射 |
| element_offsets_local_m | [16,3] 阵元局部坐标 |
| element_offsets_world_m | [16,3] 全局轴系中的相对中心偏移 |
| element_positions_world_m | [16,3] 每个阵元的绝对全局位置 |

变换：`p_world = center_world + R_local_to_world @ p_local`。这些是合法采集先验，不包含发射源真值。

## 传播与旧 Agent

RT 使用 Sionna `AntennaArray` 的显式 normalized_positions，接收实际十字形坐标，不用 2×8 的矩形冒充十字阵。自由空间后端使用同一几何函数。

旧 Agent 的 MUSIC 只支持线阵。本次将十字阵／面阵在旧 manifest 中标记为 custom，检查工具不宣称 DOA 可用；完整三维几何保存在 provenance。频谱等非空间工具仍可用。未实现新的二维 MUSIC 或定位器。

共面阵列仍有镜像／前后歧义，后续二维测向需要明确角域先验、方向图或其他站点信息，不能仅因增加阵元就声称唯一定位。

## 兼容

- 新配置与公开输出 schema_version=1.2。
- 带 version=1.0 的旧配置保留 1×4 默认线阵；旧 elements/spacing_wavelengths 字段继续映射为线阵。
- 带 version=1.1 的旧配置，包括只填写 spacing/pattern 的部分配置，保留 2×2 默认矩形阵。
- 新默认为 8+8 十字阵。不同类型的尺寸字段不可混用，零尺寸及奇数十字臂配置拒绝。
- 旧结果目录未覆盖。核心几何改变后，旧版本精确重放需对应版本源码；本版本应重放新的 cross16 结果。

## 验收

正式结果：`results/munich_cross16_verified/`。

- 2 个通信目标、2 个干扰设备、3 个 16 通道站；50 ms、2 MS/s，每站 shape=[16,100000]，总原始 IQ 38.4 MB。
- 真实 CPU RT 求解 240 条路径；已保存场景图、信号图和 array_geometry.png。
- Windows：60 passed，4 个 opt-in RT 测试跳过。
- WSL：64 passed，含 8+8 十字阵任意旋转／非零仰角的真实 RT 相位验证，以及 2×3 矩形与 Sionna 原生坐标逐元素核对。
- 新增迁移、通道编号、中心、间距、无重复点、旋转、全局绝对位置及旧 Agent DOA 不误用测试。
- `results/munich_cross16_replay/` 三站 IQ SHA256 与正式结果逐个相同。
- 三站旧 Agent inspect_iq 和 compute_spectrum 实际调用成功。

查看几何：

```powershell
wsl --exec /root/.venvs/static-city-rf/bin/python tools/render_arrays.py results/munich_cross16_verified/public --output results/munich_cross16_verified/array_geometry.png
```

# 相干多径怎样分开，怎样得到可信的路径方向
类型：方法
分类：空间与源数
关键词：解相干、空间平滑、前后向平滑、MUSIC、十字阵、多径、方向配对

## 1. 目的与触发条件

一个源经过几栋楼反射，同一波形会从不同方向到达。对单音或时延很接近的窄带路径，这些副本可能高度相干。协方差看起来近秩一，但最高空间谱峰不一定是任何一条真实路径。若单平面波拟合差、各子带角度不稳，或拿主峰去反射定位总不一致，应先检查这里的问题。

目标是得到“哪些路径方向能解释阵列数据”，不是把空间模态数当成设备数，也不是保证所有路径都能分开。

## 2. 输入与适用条件

输入是已经选定同一成分的复 IQ `xband[M,N]`、RF 频率、实际阵元坐标和阵列姿态。先核对频带内是否混入其他源；空域处理不能替代成分关联。

**调用前对齐输入：** 用某子带算空间谱时，fit_path_candidates 也必须接收同一子带的 IQ，而不是原始全带数组。否则残差包含其他源，不能解释为当前成分的多径。保存输入频带、时间窗和功率；对比 K=1/K>1 或前后两次结果时保持这些条件相同。实际方向相同但选频不同，残差下降不能归因于方向改善。详见[计算有效性](computation_validity.md)。

本页实现适用于均匀线阵，或十字阵中分别取出的均匀直线臂。必须保持阵元沿同一轴的等间距顺序和增益/相位标定，不能把十字阵的 16 通道连起来当一条线阵。跨子阵要求近似平面波、相同阵元响应；空间平滑缩短有效孔径，对近方向路径可能无效。

路径数 K 是待比较的模型，不由原始秩一直接设成 1。保守起点要求子阵长度 P>K、平移子阵数 M−P+1≥K；这不是充分保证，重合的方向投影仍不可区分。可比较 K=1、2、3，以及不同子阵长度与时间窗。

## 3. 具体处理操作

1. 保存未平滑的协方差、特征值、单方向拟合残差。不要先平均复数阵元，否则可能抵消某个方向的信号。
2. 对一条臂选 P 个连续阵元构成子阵，沿阵列平移，计算并平均各子阵协方差：`Rss = mean_j(Xj Xjᴴ/N)`。前后向平滑再计算 `(Rss + J Rss* J)/2`，J 表示倒序。它利用空间平移来改善相干路径的可分性，不是增加实际测量孔径。
3. 对平滑后的矩阵运行 MUSIC，输出沿该臂轴向的方向余弦 μ。线阵的一维 μ 不是完整方位/俯仰角；最高 K 个峰只是当前 K 假设下的候选。
4. 对局部 yz 平面的十字阵，分别得到 μy、μz；保留满足 `μy²+μz²≤1` 的配对，构造 `u=(±sqrt(1−μy²−μz²), μy, μz)`，再用阵列旋转矩阵变到世界坐标。垂直臂若按 −z 排列，应倒序或取反投影。± 两个法向分量都是合法镜像，不能只留一个。
5. 不按两臂峰的名次直接配对。对候选方向集合构造完整阵列导向矩阵 A，用全部通道拟合；比较归一化残差 `tr[(I−AA†)R]/tr(R)`。同时记录 A 的条件数。镜像可能给完全相同的残差，应保留给后续地图定位。
6. 更换子阵长度、分窗和窄子带，检查方向与残差是否稳定。更多自由度通常有利于降低训练残差，因此不能只选残差最小的最大 K；还要考虑模型复杂度、噪声对照与未参与选择的时间窗。重复数据或同一矩阵派生指标不是独立证据。
7. 若仍不能稳定分开，保留合成响应及候选集合，改查成分污染、阵列校准或联合多径拟合，不将不稳定峰送入定位当作确定方向。
   路径结果保存为“路径组合假设”：包括K、同一组完整 directions、candidate_indices、残差、条件数与验证窗口。不要只抄每站几个角度进入下一步；镜像配对与不同路径的共存关系也必须保留。K改变后不同组合可能不相容，不能把各组最低残差的单个方向随意拼成新模型。

### 选择模型是单独的一步

按[模型比较](question_model_selection.md)保存每个尚有证据的 K、子阵长度及完整方向组合。models 的排序不是推荐顺序；既不能机械取第一项，也不能看到单路径解释了大部分能量就忽略多路径候选。几何对角度很敏感，较小的能量失配仍可能意味着错误墙面。

把仍有竞争力的完整模型送入相同几何条件比较；若初选无候选，回查这些模型，而不是固定其方向再扩大角峰组合。比较阶段保留数据签名、误差、稳定性及几何结果，选择理由不能只有“简单”或“残差最小”。多个已读模型中若只挑了一个，交代其余为何暂缓/被排除。

## 4. 结果判读

| 结果 | 可以怎样理解 | 还不能声称什么 |
|---|---|---|
| 平滑后出现两个稳定方向、完整阵列残差明显下降 | 两路径模型值得保留 | 必然是两个设备，或方向一定对应主反射径 |
| 两臂各有两个峰，但某些配对残差很大 | 不合适的二维配对可被削弱 | 剩余配对已消除前后镜像 |
| 加 K 后残差下降，但分窗角度漂移 | 可能拟合噪声、混合或模型失配 | 已发现额外稳定路径 |
| 两路径投影重合或阵列孔径不足 | 当前方法不能可靠分开 | 不存在多径 |
| 世界方向相反但 abs(dot)=1 | 仅说明反向，不能按一致接收 | 阵面镜像等价；正确镜像为 u−2(u·n)n |

## 5. 后续调查

稳定路径进入[多路径几何与距离定位](multipath_geometry_localization.md)。同一成分的路径需要共享波形/活动等关联依据；不能仅因都在一个频带就归为一源。空间模态解释见[空间阶数](spatial_count.md)，混合问题见[同频分离](cochannel_components.md)。

## 6. 示例与实现

### 封装使用方式

band：已选频SignalBlock，必须携带真实阵元坐标和姿态。当前path_hypotheses适用两条等长的local-y/local-z均匀直线臂；不是任意阵列。

公共记录的创建与保存见[信号数据对象](signal_record.md)，本方法的调用与返回说明如下。

<!-- checked-example:start -->
```python
from rfkit.analysis_api import single_path_fit, path_hypotheses, save_result, save_signal
single = single_path_fit(band, grid_step_deg=2.)
paths = path_hypotheses(band, path_counts=(1, 2, 3), subarray_sizes=(4, 5, 6))
save_result('work/path_models.json', paths)
save_signal('work/path_band.npz', band)  # 每站/频带使用不同文件名
for model in paths['models']:
    print(model['path_count'], model['subarray_size'],
          model['training_residual'], model['validation_residual'])
```
<!-- checked-example:end -->

models为list[dict]，每项是一组同时存在的路径，含directions[K,3]世界单位向量、normal_world、path_count、training_residual、validation_residual、condition_number、signal_signature及subarray_size。函数自动分臂排序/姿态变换，拒绝非法P/K和不均匀阵列。models按K后按残差排，不代表第一项最佳。没有zenith_deg字段；需要角度时由向量计算。保留整个model给定位，勿只抄角度。

### 推导参考与低层实现

已实现 `lib/rfkit/multipath.py`，可在 Agent 的 python 工具中直接导入。下面针对本项目 8+8、前 8 个沿 +y、后 8 个沿 −z 的十字阵；`offsets_world_m` 和 `rotation_local_to_world` 来自观测说明，`carrier_frequency_hz` 用实际 RF 参考。输入 xband 是选频后的 16 通道数据，示例比较两条路径，不自动判定 K。

```python
from rfkit.multipath import (
    smooth_ula_covariance, ula_music,
    cross_direction_candidates, fit_path_candidates,
)
ry = smooth_ula_covariance(xband[:8], subarray_size=5)
rz = smooth_ula_covariance(xband[8:16][::-1], subarray_size=5)
sy = ula_music(ry, spacing_wavelengths=0.5, path_count=2)
sz = ula_music(rz, spacing_wavelengths=0.5, path_count=2)
directions = cross_direction_candidates(
    sy['direction_cosines'], sz['direction_cosines'], rotation_local_to_world)
hypotheses = fit_path_candidates(
    xband, offsets_world_m, carrier_frequency_hz, directions, path_count=2)
# hypotheses: directions / residual_fraction / condition_number
# sy、sz 保留完整谱与特征值；保存多组镜像/配对候选，不只第一项。
```

`fit_path_candidates` 使用协方差计算完整阵列投影残差，无需逐组合重新读取全部样点；默认最多 10000 个组合、返回 20 项。这个残差不是定位置信度，也未自动完成路径阶数选择、时间窗稳健性或 RF 校准。

## 7. 常见误判与来源

空间平滑适用的阵列结构必须满足，不能对任意阵列直接切连续行。近场、增益/相位误差、混合成分和多个路径在某一臂上的投影重合都会影响结果。单站共面阵本身的镜像不能靠此方法消除。

阵元关于阵心对称不意味着 uy、uz 的任意符号都等价。复数通道保留顺序和相位；只有经实际导向矢量检验等价的方向才能组成同一路径的模糊候选。低残差多路径组合中的一条路径，单独拟合可能很差；这既不能单独否定它，也不能将它包装成已验证的单路径。

参考：[DOATools 空间平滑及前后向平滑实现](https://morriswmz.github.io/doatools.py/references/doatools.estimation.preprocessing.html)，其引用 Pillai 与 Kwon 的相干信号空间平滑研究。本项目实现不复制该库代码；已用完全相干双路径和十字阵配对合成数据测试，真实城市检出/定位能力需另行验证。

# 怎样用两条或多条路径，结合建筑与距离定位同一个源
类型：方法
分类：空间与源数
关键词：单站多径、多站、多路径、反射、距离差、TDOA、镜像源、定位、可见性

## 1. 目的与触发条件

直达路径被遮挡时，到达接收机的方向可能指向一面墙，而不是发射机。若已得到同一源的两条或多条可信路径，可以利用建筑表面把这些方向反向追踪，再寻找共同的源位置。路径可以来自同一接收站，也可以来自多个站。

“两个峰”不等于“两条已确认路径”。本页的前置步骤是[路径模型验证](question_path_model.md)及[解相干与路径方向](coherent_multipath_directions.md)。先排除合成峰、旁瓣、阵面镜像和混合源，并保留不能排除的关联方案；一条路径也可能没有被正确分离。地图搜索不能弥补输入方向未验证。

## 2. 输入与适用条件

输入为各路径的接收站位置、世界到达方向候选、角度容差，以及同坐标系建筑三角面。方向均从接收站指向来波路径。还可提供总传播长度或路径长度差及误差容限。

输入必须来自同一个可接受的路径组合假设。每条独立路径是一条 observation，该条 directions 是“二选一”的方向候选，不是两条同时存在的路径。以下结构含义不同：

```text
错误（把两条真实路径变成二选一）：
observations = [{receiver_id: R, directions: [路径1方向, 路径2方向]}]
正确（两条路径同时约束位置）：
observations = [
  {receiver_id: R, position_m: 站位置, directions: [路径1方向, 路径1镜像]},
  {receiver_id: R, position_m: 站位置, directions: [路径2方向, 路径2镜像]}
]
```

镜像用阵面法向计算 u−2(u·n)n，不是随意反转方位或平面内投影。若方向没有该镜像歧义，不要人为增加镜像。多个可接受的全阵列路径组合应分别构建定位问题，不能把不同组合的单个角度混合成“最佳集合”。局部坐标到世界坐标只变换一次。

绝对长度 `L=cτ` 需要已知发射时间及收发时间基准；仅有接收时间不够。同一源两路径的时延差可消除共同未知发射时刻，但需路径可分、延迟已消除周期模糊；跨站还需同步/偏差校准。CW 载波相位差一般不能直接当绝对距离或无模糊时差。没有可信时延时只用几何，并如实报告歧义。

两条同站反射径有机会提供位置约束，是因为它们反向追踪后从不同墙面点出发；若路径重复、近平行、反射面关联错误或几何条件差，仍不能唯一定位。

## 3. 具体处理操作

1. 为每个“观测路径”保留独立条目，记录原站点、频带/活动、方向候选、关联依据和角度误差。同站可有多条，单个峰的两个阵面镜像应放在同一条目的候选里，不能当两条独立测量。
2. 从接收站沿候选方向找到第一处建筑交点 p。直达分支只允许在第一遮挡之前；反射分支沿 `d'=d−2(d·n)n` 回溯。保留反射点和接收站到它的长度，继续检查下一遮挡。
3. 枚举合理路径关联，用反向射线交会求位置候选。每条都要检查正向距离、角残差、交会条件数、有效射线长度和遮挡。不要用 abs(dot) 把反方向当一致，也不要忽略反射线段之外的下一栋楼。
4. 计算完整传播长度：直达为 `|源−接收站|`；一次反射为 `|源−反射点|+|反射点−接收站|`。反射点到源的距离不是总长度，发射源到接收站的欧氏距离也不能代替反射传播长度。
5. 有距离证据时，比较预测长度与测量，或 `L_i−L_j` 与实测路径长度差。按已校准误差容限拒绝冲突候选、排序剩余候选；不要为留下某坐标临时放宽容限。检查延迟闭合、相关峰显著性与周期歧义。
6. 对角度、时间及反射面关联做扰动检查，并用未参与筛选的路径/时间段或原始阵列响应复核。多路径来自同一批数据，证据可能相关，不能机械按数量增加置信度。
   容差与角度变更必须有分窗变化、校准或拟合误差依据。无解后手工移动仰角或把6°改为12°只能算敏感性探索，不是获得了更可靠测量。记录改动理由，候选回到原IQ全阵列拟合；不得仅凭新容差下的低几何残差写成定位成功。
7. 输出全部仍受支持的位置簇、对应路径和长度残差。几何只能约束一条射线或一个区域时，报告该约束；不能强填一个点。

### 无候选时怎样继续

先读[模型选择与反证回查](question_model_selection.md)，把本次完整输入组合和失败写回 TODO。优先在相同观测、地图和误差标准下比较尚未排除的上游模型，例如另一组路径数/配对、另一个有证据的成分关联。不是机械增加路径数，而是区分哪项输入假设与几何相容。

如果仍无解，检查方向稳定性、地图范围、传播阶数或几何实现限制。手写交会/搜索可以作为有目的的对照，但要先说明它检验什么，不能把同一批有疑点的方向换个程序算出坐标就算解决了问题。手工改变高度或放宽角度得到的位置，必须回到原误差约定和原 IQ 核验。

### 如何接受、保留或否决位置

位置候选要同时交代：采用哪些站/路径、每条采用哪种传播解释、镜像是否仍有竞争、角误差/遮挡检查以及未解释观测。候选与某站无遮挡且方向吻合，只支持这一条可能解释；该站镜像方向也可能由反射形成。几何可见也不等于一定能检出直达波，声称“应看到却没有”还需要接收响应与灵敏度依据。

不能将解释不了其他用于交会的站点归为“可能多跳”后仍宣称定位完成。需要多跳的候选可保留为待检验分支，实际验证之前不升级为受到多站支持的首选。已被实际约束否决的点与仍成立的位置候选分开报告。仅有一站方向时比较直达段和可用反射段，不把来波射线默认当作源位置射线；代表点说明其条件，不能作为几个独立定位解。

## 4. 结果判读

| 结果 | 支持什么 | 下一步 |
|---|---|---|
| 多条可见反射径交会且距离证据相容 | 当前路径/几何模型下的位置候选 | 角度扰动和原始 IQ 验证 |
| 角度拟合好但完整长度差冲突 | 位置或路径关联不成立，或测距有误 | 检查信号分离、延迟定义与误差模型 |
| 候选需要穿过建筑才能到反射点或源 | 不符合当前传播模型 | 拒绝，不用小角残差救回 |
| 两路径近平行、同一镜像被重复计数 | 信息退化或伪重复证据 | 寻找独立路径/站点/距离约束 |
| 几何与距离都匹配多个簇 | 尚有歧义 | 保留多个簇并说明额外测量需求 |
| 无候选 | 输入角度、关联或传播模型不能同时解释 | 不能直接说没有源或无法定位 |

## 5. 后续调查

方向不可靠先回到[相干多径方向分解](coherent_multipath_directions.md)，不同站可能看到不同源时查[跨站关联证据](association_evidence.md)。单站只有一个可见成分时不要强行借用其他站的合作信号做交会。基础几何与阵面约定见[反射定位](reflection_localization.md)、[建筑几何](building_geometry.md)。

## 6. 示例与实现

### 封装使用方式

geometry：同坐标系建筑Geometry；selections为每站一个(band,该站完整model)的列表。model直接来自路径拟合；angle_tolerance_deg由分窗/校准误差支持。

公共记录的创建与保存见[信号数据对象](signal_record.md)，本方法的调用与返回说明如下。

<!-- checked-example:start -->
```python
from rfkit.analysis_api import locate_paths, save_result
result = locate_paths(geometry, selections,
                      angle_tolerance_deg=angle_tolerance_deg,
                      path_differences=(), height_range=(0., 200.))
save_result('work/location_candidates.json', result)
print(result['status'], len(result['observations']), len(result['hypotheses']))
```
<!-- checked-example:end -->

每个model中的每条物理路径会自动成为独立observation，镜像由真实normal_world生成；拒绝跨数据签名/同站不相容model混用。hypotheses含position_m、paths、角度/长度残差和score，均为条件性候选。距离差可填{i,j,difference_m,tolerance_m}，索引对应observations，符号为Li-Lj；没有证据传空。不自动放宽容差或增加角度。

### 从路径检验到定位：跨工具调用保留原始模型

Python 工具每次启动新进程，上一轮变量不会自动保留。完成分离时，针对每站、每个分析频带分别保存 `save_signal` 的 IQ 对象和 `save_result` 的完整 `paths`；不要用四舍五入的方向摘要替代模型。用留出残差、不同子阵长度下的稳定性和模型复杂度选择候选，记录所选 `models` 下标及理由。下标不是源编号，第一项也不一定最好。

下一次定位调用的衔接如下。`selected_files` 是已完成路径检验后选择的 `(IQ文件, 路径结果文件, 模型下标)` 列表，每站一个联合模型；`angle_tolerance_deg` 来自方向稳定性/校准证据。更换竞争模型时应完整重算另一组定位假设，而不是手工拼它们的个别方向。

<!-- handoff-example:start -->
```python
import json
from rfkit.analysis_api import Geometry, load_signal, locate_paths, save_result
selections = []
for signal_path, models_path, model_index in selected_files:
    band = load_signal(signal_path)
    with open(models_path, encoding='utf-8') as f:
        paths = json.load(f)
    model = paths['models'][model_index]
    selections.append((band, model))
geometry = Geometry.load(geometry_path)
result = locate_paths(geometry, selections,
                      angle_tolerance_deg=angle_tolerance_deg,
                      height_range=(0., 200.))
save_result('work/location_candidates.json', result)
```
<!-- handoff-example:end -->

这一步不需要自行构造 directions、stations 或做阵面镜像配对。完整 model 中的方向组、阵面法向和数据签名都会保留。复数和 NumPy 数据使用 `save_result` 保存；`brief` 仅供显示，不能保存它来替代计算结果。

### 推导参考与低层实现

已实现 `lib/rfkit/multipath.py` 的 `localize_multipath`，允许同一 receiver_id 出现在多条观测中。每条的 `directions` 是该路径的竞争方向，包括未消除的阵面镜像。示例输入 `observations` 与可选 `path_differences` 均由实际检验得到：

```python
from rfkit.multipath import localize_multipath
result = localize_multipath(
    geometry, observations,
    path_differences=path_differences,
    max_angle_deg=5., height_range=(0.,200.))
# observations 每项：receiver_id、position_m、directions。
# 可选 total_length_m、length_tolerance_m：完整路径绝对长度和容限。
# path_differences 每项：i、j、difference_m、tolerance_m。
# i/j 为 observations 下标，difference_m 定义为 L_i-L_j；没有时延证据时传 []。
# result['hypotheses'] 包含位置、各路径分支、完整长度、角度与距离残差。
```

本版先在固定方向/反射点上交会，再用距离筛选排序，**尚未联合优化有误差的角度、反射点和距离**。默认最多 10000 组合、返回 20 项；每条角误差均需通过，不只看平均误差。评分是按容限归一化的残差，不是置信概率。

## 7. 常见误判与来源

当前实现只含直达/一次镜面反射；不自动估计 DOA、时延、源身份或路径关联，也没有多跳/绕射反演。模型不包括的传播可能导致无候选。路径长度差不能直接套直达距离差方程。

调用成功只说明几何程序运行完成。提交前按[计算有效性](computation_validity.md)核对路径条目、镜像、角度来源和未使用的路径。某方向属于多路径组合，不代表它可独立解释全阵数据；选中一条路径后不能丢掉其他已验证路径而仍声称做了联合多径定位。

参考：[地图与镜像源辅助多径 DPD 研究](https://publica.fraunhofer.de/entities/publication/e23cd789-0a5a-40ba-a04a-92d23d917722)、[Sionna RT 路径建模](https://nvlabs.github.io/sionna/rt/tech-report/index.html)。本页是几何候选验证方法，不是上述 DPD 论文的复现。已验证同站双反射合成交会、正确/错误长度差筛选及平行路径退化；不把真值方向对照当成盲定位成功。

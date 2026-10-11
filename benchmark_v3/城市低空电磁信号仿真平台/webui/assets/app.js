import {SceneMap} from './map.js?v=hz4';
import {plot} from './charts.js';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)],clone=o=>structuredClone(o);
let config,boot,selected={type:'global'},currentRun=null,resultTab='spectra',poll=null,runData=null,toastTimer;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(t){$('#toast').textContent=t;$('#toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('show'),5000);}
async function api(path,data){const r=await fetch('/api/'+path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const json=await r.json();if(!r.ok)throw Error(typeof json.detail==='string'?json.detail:JSON.stringify(json.detail));return json;}
async function attempt(fn){try{await fn()}catch(e){toast(e.message);console.error(e)}}
const specs={
 name:['场景名称'],seed:['随机种子'],duration_s:['观测时长 · ms',1000,'最长 500 ms。采样点数 = 采样率 × 时长；跳频/突发要看清结构就加长窗口'],sample_rate_hz:['采样率 · MS/s',1e-6],carrier_frequency_hz:['参考载频 · GHz',1e-9],max_working_mb:['DSP 内存预算 · MB',1,'不含 RT 求解器额外内存'],
 id:['设备 ID',1,'字母开头，仅英文、数字、_、-'],platform_id:['平台关联 ID',1,'可留空；用于关联同一平台'],position_m:['世界位置 · m'],orientation_rad:['朝向 · °',180/Math.PI,'yaw / pitch / roll；Rz·Ry·Rx，阵面正面 +x'],
 role:['设备角色'],power_dbm:['名义发射功率 · dBm',1,'组合参考波形功率；门控、滤波后平均功率可能下降'],rf_offset_hz:['相对参考载频 · MHz',1e-6,'该设备自己的射频，不再强行都放在参考载频上'],velocity_mps:['速度 · m/s',1,'X/Y/Z；短窗内按径向速度产生多普勒'],kind:['波形类型'],weight:['成分功率权重'],offset_hz:['中心偏移 · kHz',.001],bandwidth_hz:['带宽 · kHz',.001,'发射端为带限参数；不等同于实测 99% 占用带宽'],symbol_rate_hz:['符号率 · ksym/s',.001,'用于 BPSK/QPSK/FSK；OFDM 内部速率由带宽决定'],phase_rad:['初相 · °',180/Math.PI],pulse_period_s:['脉冲周期 · ms',1000,'留空为连续发射；占空比仅在周期非空时生效'],duty_cycle:['占空比',1,'0 < duty ≤ 1'],pulse_start_s:['脉冲起点 · ms',1000],chirp_period_s:['扫频周期 · ms',1000],
 geometry_type:['阵列几何'],num_rows:['矩形阵行数'],num_cols:['矩形阵列数'],horizontal_elements:['十字阵水平臂阵元数'],vertical_elements:['十字阵垂直臂阵元数'],vertical_spacing_wavelengths:['垂直间距 · λ'],horizontal_spacing_wavelengths:['水平间距 · λ'],pattern:['天线方向图'],polarization:['极化'],gain_error_std_db:['阵元增益误差 · dB',1,'0 表示各阵元幅度一致'],phase_error_std_deg:['阵元相位误差 · °',1,'0 表示各阵元相位校准'],tuning_offset_hz:['调谐偏移 · kHz',.001],
 noise_figure_db:['噪声系数 · dB'],temperature_k:['温度 · K'],noise_enabled:['热噪声（白噪声）'],colored_noise_enabled:['有色/带限底噪'],colored_noise_db:['有色噪声相对热噪声 · dB'],colored_bandwidth_hz:['有色噪声带宽 · kHz',.001],impulse_noise_enabled:['脉冲/冲击噪声'],impulse_rate_hz:['脉冲出现率 · 次/秒'],impulse_power_db:['脉冲相对热噪声 · dB'],impulse_width_us:['脉冲宽度 · μs'],phase_noise_enabled:['本振相位噪声'],phase_noise_deg:['相位噪声步进 · °/样点'],correlated_noise_enabled:['阵元相关噪声'],noise_correlation:['阵元噪声相关系数',1,'0 完全独立，1 各阵元同一份噪声'],imd3_enabled:['三阶互调'],iip3_dbm:['输入三阶截点 · dBm',1,'越小，强信号产生的互调越明显'],gain_db:['固定增益 · dB'],soft_limit_input_dbm:['输入软限幅阈值 · dBm',1,'留空关闭非线性压缩'],filter_taps:['接收滤波器抽头数',1,'17–1025 之间的奇数'],
backend:['传播后端',1,'sionna_rt 走真实城市射线追踪（慢）；free_space / tapped 不考虑建筑（快），此时传播场景不影响结果。'],rt_variant:['RT 计算设备',1,'CUDA 需机器上有可用 GPU 与 OptiX 运行时。'],scene:['传播场景',1,'仅 Sionna RT 使用：munich = 内置慕尼黑；自定义城市 = priors/<城市>_geometry/scene.xml（如杭州）；empty = 无建筑空场景。与右上角城市网格下拉联动。'],max_depth:['最大路径交互深度',1,'每条射线最多与建筑交互几次（镜面反射 / 绕射 / 折射）。0 = 只算直达；默认 3；每加 1 计算量近似翻倍，实际路径数还受「每源路径预算」限制。'],samples_per_src:['每源射线采样数',1,'发射的射线条数：越多越不容易漏掉弱路径，也越慢。'],max_num_paths_per_src:['每源路径预算',1,'每条源-站链路保留的最大路径数，超出按功率截断。'],los:['直达路径',1,'关闭后只保留反射 / 绕射等多径。'],specular_reflection:['镜面反射'],diffuse_reflection:['漫反射',1,'开启漫反射模型，通常显著变慢。'],refraction:['折射'],diffraction:['绕射',1,'开启后建筑边缘会产生绕射路径，代价明显增加。'],tap_delays_s:['抽头时延 · μs',1e6],tap_gains_real:['抽头增益实部'],tap_gains_imag:['抽头增益虚部'],
receiver_ids:['参与监测站 ID',1,'只有列表里的站参与该任务，并出现在公开观测里。'],desired_emitter_ids:['合作 / 受保护发射源 ID',1,'声明「应当正常占用保护频带」的合作源；它们会作为已知先验出现在公开观测中。'],protected_center_offset_hz:['保护频带中心 · kHz',.001,'保护频带 = 该任务声明合作源应占用的频率窗口，用于把「受保护业务」与「未知占用」分开；只影响公开元数据，不改变仿真波形。'],protected_bandwidth_hz:['保护带宽 · kHz',.001,'保护频带宽度（默认 100 kHz）。'],instruction:['分析任务说明',1,'给分析方（或 Agent）的自然语言任务描述，会写进公开观测。']
};
const enums={role:{communication:'通信源',interference:'干扰源',background:'背景源'},kind:{tone:'单音 Tone',noise:'带限噪声',bpsk:'BPSK',qpsk:'QPSK',fsk:'FSK',ofdm:'OFDM',chirp:'线性扫频 Chirp'},backend:{sionna_rt:'Sionna RT · 真实城市',free_space:'自由空间 · 不考虑建筑',tapped:'指定多径抽头 · 不考虑建筑'},rt_variant:{llvm_ad_mono_polarized:'LLVM · CPU',cuda_ad_mono_polarized:'CUDA · GPU（需 OptiX）'},scene:{munich:'慕尼黑完整城市',empty:'空场景'},geometry_type:{cross:'十字阵',rectangular:'矩形面阵'},pattern:{iso:'各向同性'},polarization:{V:'垂直极化'}};
let map;
try{map=new SceneMap($('#map'),select,(type,index,pos,final)=>{if(pos){config[type][index].position_m=pos;refresh(false)}if(final)renderInspector()});await map.load('munich').catch(e=>toast('地图网格加载失败：'+e.message));}catch(e){$('#map').textContent='WebGL 不可用；仍可编辑配置和运行。'+e.message;}
function select(type,index){selected={type,index};renderInspector();refresh(false);}
function changed(obj,key,value){
 const prev=obj[key];obj[key]=value;
 if(key==='scene'&&obj===config.propagation)attempt(()=>applyScene(value));
 if(key==='geometry_type'){
  if(value==='cross'){obj.num_rows=null;obj.num_cols=null;obj.horizontal_elements=8;obj.vertical_elements=8;}
  else{obj.horizontal_elements=null;obj.vertical_elements=null;obj.num_rows=4;obj.num_cols=4;}
  renderInspector();
 }
 if(key==='id'&&selected.type!=='tasks')for(const task of config.tasks)for(const k of ['receiver_ids','desired_emitter_ids'])task[k]=task[k].map(v=>v===prev?value:v);
 refresh(false);
}
const CITY_SCENE=/^priors\/([A-Za-z0-9_-]+)_geometry\//;
const PRESET_LABELS={'benchmark_v3/d1_logistics_airway.json':'D1 · 物流航线基线（单站）','benchmark_v3/d2_residential_coexistence.json':'D2 · 居民区共存（四站）','benchmark_v3/d3_urban_canyon.json':'D3 · 城市峡谷多径（四站）','benchmark_v3/d4_adversarial_contest.json':'D4 · 对抗演习（四站）'};
function sceneForMesh(id){return id==='munich'?'munich':`priors/${id}_geometry/scene.xml`}
function meshForScene(scene){if(scene==='munich')return 'munich';const m=CITY_SCENE.exec(scene||'');return m?m[1]:null}
function sceneOptions(meshes){const out={munich:'慕尼黑完整城市',empty:'空场景 · 无建筑'};for(const m of meshes||[]){const v=sceneForMesh(m.id);if(!(v in out))out[v]=`${m.name} · 自定义城市场景`}return out}
async function applyScene(scene,notify=true){
 if(!config)return;config.propagation.scene=scene;
 const id=meshForScene(scene),ms=$('#mesh-select');
 if(id&&ms&&[...ms.options].some(o=>o.value===id)&&ms.value!==id){ms.value=id;await map?.load(id);const m=(boot?.meshes||[]).find(x=>x.id===id);if(m)$('#scene-title').textContent=`${m.name} · 城市电磁空间`}
 if(selected.type==='propagation')renderInspector();else refresh(false);
 if(notify)toast(id?`传播场景已切到 ${id}：地图网格与 RT 场景同步`:'传播场景已更新为「空场景」：Sionna RT 不使用建筑');
}
function field(parent,obj,key){
 const value=obj[key],[title,scale=1,baseHint]=specs[key]||[key];const hint=key==='bandwidth_hz'&&obj.frontend?'接收滤波器带宽，必须满足采样带宽约束':baseHint;const label=document.createElement('label');label.className='form-field';const span=document.createElement('span');span.textContent=title;label.append(span);parent.append(label);
 if(typeof value==='boolean'){
  label.classList.add('check');const input=document.createElement('input');input.type='checkbox';input.checked=value;input.onchange=()=>changed(obj,key,input.checked);label.append(input);
 }else if(key==='position_m'||key==='orientation_rad'){
  const box=document.createElement('div');box.className='tuple';const names=key==='position_m'?['X','Y','Z']:['Yaw','Pitch','Roll'];value.forEach((v,i)=>{const l=document.createElement('label');l.textContent=names[i];const input=document.createElement('input');input.type='number';input.step='any';input.value=+(v*scale).toFixed(4);input.setAttribute('aria-label',`${title} ${names[i]}`);input.oninput=()=>{if(input.value!==''&&Number.isFinite(input.valueAsNumber)){obj[key][i]=input.valueAsNumber/scale;refresh(false)}};l.append(input);box.append(l)});label.append(box);
 }else if(enums[key]){
  const input=document.createElement('select');for(const [v,t]of Object.entries(enums[key]))input.add(new Option(t,v));if(![...input.options].some(o=>o.value===String(value)))input.add(new Option(`${value}（不在枚举中，保留原值）`,String(value)));input.value=value;input.onchange=()=>changed(obj,key,input.value);label.append(input);
 }else if(Array.isArray(value)){
  const input=document.createElement('input');input.value=value.map(v=>typeof v==='number'?+(v*scale).toPrecision(8):v).join(', ');input.placeholder='逗号分隔';input.onchange=()=>{let v=input.value.split(',').map(s=>s.trim()).filter(Boolean);if(key.startsWith('tap_')){v=v.map(Number).map(x=>x/scale);if(!v.every(Number.isFinite)){toast('请输入逗号分隔的数值');return}}changed(obj,key,v)};label.append(input);
 }else{
  const input=document.createElement(key==='instruction'?'textarea':'input');if(key==='instruction')input.style.minHeight='100px';
  const numeric=typeof value==='number'||['pulse_period_s','soft_limit_input_dbm'].includes(key);input.type=numeric?'number':'text';if(numeric)input.step='any';input.value=value===null?'':numeric?+(value*scale).toPrecision(10):value;input.placeholder=value===null?'留空禁用 / 未设置':'';
  input.onchange=()=>{let v=input.value;if(numeric){if(v===''){if(['pulse_period_s','soft_limit_input_dbm'].includes(key))v=null;else{input.value=obj[key]*scale;return}}else{v=Number(v)/scale;if(!Number.isFinite(v))return}}else if(key==='platform_id'&&v==='')v=null;changed(obj,key,v)};label.append(input);
 }
 if(hint){const small=document.createElement('small');small.textContent=hint;label.append(small)}
}
function group(parent,title,obj,keys){const box=document.createElement('section');box.className='form-group';if(title){const h=document.createElement('h3');h.textContent=title;box.append(h)}for(const k of keys)field(box,obj,k);parent.append(box);return box;}
function button(parent,text,fn,cls=''){const b=document.createElement('button');b.textContent=text;b.className=cls;b.onclick=fn;parent.append(b);return b;}
function note(parent,text){const n=document.createElement('div');n.className='info-box';n.textContent=text;parent.append(n);}
function renderInspector(){
 const form=$('#inspector-form');form.replaceChildren();const {type,index}=selected;
 const title={global:'全局采集参数',propagation:'信道与传播',emitters:'发射源属性',receivers:'监测站属性',tasks:'任务与保护频带',json:'完整 JSON 配置'}[type];$('#inspector-title').textContent=title;
 $('#inspector-description').textContent=type==='emitters'?'一个物理发射设备可以叠加多个波形成分。':type==='receivers'?'位置与朝向均写入观测元数据。':type==='propagation'?'显示范围不裁剪传播网格；RT 使用完整地图。':'编辑后请校验；每次运行保存独立配置快照。';
 if(type==='global'){for(const k of ['name','seed','carrier_frequency_hz','sample_rate_hz','duration_s','max_working_mb'])field(form,config,k);note(form,'当前窗口采样点数 = 采样率 × 观测时长。例如 2 MS/s × 50 ms = 100000 点；跳频默认驻留只有几十个样点，窗口太短会看不到跳变，可把时长加到 100–500 ms。');}
 if(type==='propagation'){
  for(const k of ['backend','scene','rt_variant','max_depth','samples_per_src','max_num_paths_per_src','los','specular_reflection','diffuse_reflection','refraction','diffraction'])field(form,config.propagation,k);
  group(form,'指定抽头后端参数',config.propagation,['tap_delays_s','tap_gains_real','tap_gains_imag']);note(form,'地图上的虚线只表示 TX/RX 配对，不代表求得的射线路径。RT 端点需在地图范围内且无遮挡天空。');
  note(form,'传播场景（scene）只对 Sionna RT 后端生效，并与右上角城市网格下拉联动：切场景会把地图网格一起切过去。自由空间 / 抽头后端不使用建筑，改场景不会影响仿真结果。');
 }
 if(type==='emitters'){
  const d=config.emitters[index];if(!d)return select('global');for(const k of ['id','role','platform_id','position_m','velocity_mps','orientation_rad','power_dbm','rf_offset_hz'])field(form,d,k);
  // 信号源模式：预设(v3 辐射体) / 自由(调制+参数)
  const modeBox=document.createElement('section');modeBox.className='form-group';const mh=document.createElement('h3');mh.textContent='信号源模式';modeBox.append(mh);
  const mSel=document.createElement('select');mSel.add(new Option('预设 · v3 辐射体','preset'));mSel.add(new Option('自由 · 调制+参数','free'));
  mSel.value=d.preset?'preset':'free';
  mSel.onchange=()=>{d.preset=mSel.value==='preset'?(boot.radiators&&boot.radiators[0]?boot.radiators[0].name:'Bluetooth_FHSS'):null;renderInspector();refresh(false)};
  const ml=document.createElement('label');ml.className='form-field';ml.append(document.createTextNode('信号源'));ml.append(mSel);modeBox.append(ml);
  if(d.preset){
   const ps=document.createElement('select');for(const r of boot.radiators||[])ps.add(new Option(`${r.name} · ${r.scene||''}`,r.name));
   if(![...ps.options].some(o=>o.value===d.preset)){d.preset=(boot.radiators&&boot.radiators[0]?boot.radiators[0].name:'Bluetooth_FHSS');}
   ps.value=d.preset;ps.onchange=()=>{d.preset=ps.value;renderInspector();refresh(false)};
   const pl=document.createElement('label');pl.className='form-field';pl.append(document.createTextNode('辐射体类型'));pl.append(ps);modeBox.append(pl);
   note(modeBox,'预设只锁定调制和时域结构（跳频、突发、脉冲）。功率、频偏、带宽和脉冲门控仍可改。');
   const c=d.components[0];
   for(const k of ['offset_hz','bandwidth_hz','pulse_period_s','duty_cycle','pulse_start_s'])field(modeBox,c,k);
   const nSamp=Math.round((config.sample_rate_hz||0)*(config.duration_s||0));
   note(modeBox,`当前观测窗 ${nSamp.toLocaleString()} 点（${(config.sample_rate_hz/1e6).toFixed(2)} MS/s × ${(config.duration_s*1e3).toFixed(0)} ms）。改点数请到「全局采集参数」调采样率或观测时长。`);
  }
  form.append(modeBox);
  if(!d.preset){
   d.components.forEach((c,i)=>{const box=document.createElement('div');box.className='component';const h=document.createElement('h3');h.textContent=`波形成分 ${String(i+1).padStart(2,'0')}`;button(h,'删除',()=>{if(d.components.length===1)return toast('至少保留一个波形成分');d.components.splice(i,1);renderInspector();refresh(false)});box.append(h);for(const k of ['kind','weight','offset_hz','bandwidth_hz','symbol_rate_hz','phase_rad','pulse_period_s','duty_cycle','pulse_start_s','chirp_period_s'])field(box,c,k);form.append(box)});
   button(form,'＋ 添加波形成分',()=>{d.components.push(clone(boot.defaults.component));renderInspector();refresh(false)});
   note(form,'OFDM 为固定 FFT64 / CP16 / 32 个 QPSK 子载波代理；不是完整厂商协议。成分相干组合后统一标定名义功率。');
  }
  const a=document.createElement('div');a.className='form-actions';button(a,'复制设备',()=>duplicate('emitters',index));button(a,'删除设备',()=>remove('emitters',index),'danger');form.append(a);
 }
 if(type==='receivers'){
  const d=config.receivers[index];if(!d)return select('global');
  for(const [k,v] of Object.entries(boot?.defaults?.receiver?.frontend||{})) if(!(k in d.frontend)) d.frontend[k]=v;
  for(const k of ['id','position_m','orientation_rad','tuning_offset_hz','bandwidth_hz'])field(form,d,k);
  group(form,'接收阵列',d.antenna,['geometry_type',...(d.antenna.geometry_type==='cross'?['horizontal_elements','vertical_elements']:['num_rows','num_cols']),'horizontal_spacing_wavelengths','vertical_spacing_wavelengths','pattern','polarization','gain_error_std_db','phase_error_std_deg']);
  button(form,'将此阵列规格应用到所有站',()=>{for(const r of config.receivers)r.antenna=clone(d.antenna);refresh(false);toast('已同步阵列规格；各站位置和朝向独立保留。')});
  note(form,'每站最多 16 阵元；十字阵双臂需为偶数。Sionna RT 当前要求所有接收站阵列规格一致。地图箭头表示阵面正面方向。');
  group(form,'接收机前端',d.frontend,Object.keys(d.frontend));const a=document.createElement('div');a.className='form-actions';button(a,'复制监测站',()=>duplicate('receivers',index));button(a,'删除监测站',()=>remove('receivers',index),'danger');form.append(a);
 }
 if(type==='tasks'){
  if(!config.tasks.length)note(form,'当前场景没有配置保护任务（tasks 为空）。D1–D4 基准数据集的场景就是这样：只用发射源与监测站描述物理环境，不声明受保护业务，所以公开观测里不会出现合作源先验。点下方「＋ 添加保护任务」可添加一条。');
  config.tasks.forEach((t,i)=>{const box=document.createElement('div');box.className='component';for(const k of Object.keys(t))field(box,t,k);button(box,'删除任务',()=>{config.tasks.splice(i,1);renderInspector()},'danger');form.append(box)});
  button(form,'＋ 添加保护任务',()=>{const t=clone(boot.defaults.task);t.id='task_'+(config.tasks.length+1);t.receiver_ids=config.receivers.map(x=>x.id);t.desired_emitter_ids=[config.emitters[0].id];config.tasks.push(t);renderInspector()});note(form,'任务定义公开合作源和保护频带。未配置任务时，公开观测不暴露未知源位置和干扰数量。ID 列表以逗号分隔。');
 }
 if(type==='json'){
  const ta=document.createElement('textarea');ta.id='json-editor';ta.value=JSON.stringify(config,null,2);ta.spellcheck=false;form.append(ta);
  button(form,'校验并应用 JSON',()=>attempt(async()=>{const data=await api('validate',JSON.parse(ta.value));config=data.config;refresh(true);toast('JSON 已校验并应用')}),'primary');note(form,'完整编辑入口与表单使用相同 Scenario schema。未知字段、非法带宽、重复 ID 会被后端拒绝。');
 }
}
function uniqueId(prefix){const ids=new Set([...config.emitters,...config.receivers].map(d=>d.id));let i=1;while(ids.has(prefix+i))i++;return prefix+i;}
function duplicate(type,index){const d=clone(config[type][index]);d.id=uniqueId(type==='emitters'?'tx_':'rx_');d.position_m[0]+=20;config[type].push(d);select(type,config[type].length-1)}
function remove(type,index){if(config[type].length===1)return toast('至少保留一个发射源与一个接收站');const id=config[type][index].id;config[type].splice(index,1);for(const t of config.tasks){t.receiver_ids=t.receiver_ids.filter(x=>x!==id);t.desired_emitter_ids=t.desired_emitter_ids.filter(x=>x!==id)}const n=config.tasks.length;config.tasks=config.tasks.filter(t=>t.receiver_ids.length&&t.desired_emitter_ids.length);select('global');toast('设备已移除'+(n!==config.tasks.length?'；已移除因此失效的保护任务':''));}
function refresh(inspector=true){
 if(!config)return;$('#scenario-name').textContent=config.name;$('#tx-count').textContent=config.emitters.length;$('#rx-count').textContent=config.receivers.length;
 for(const [type,selector]of [['emitters','#tx-list'],['receivers','#rx-list']]){$(selector).innerHTML=config[type].map((d,i)=>`<div class="object ${type==='receivers'?'rx':d.role==='interference'?'jam':''} ${selected.type===type&&selected.index===i?'active':''}" data-type="${type}" data-index="${i}" role="button" tabindex="0"><div class="symbol">${type==='receivers'?'▲':'◉'}</div><div><strong>${esc(d.id)}</strong><small>${type==='receivers'?`${d.antenna.geometry_type==='cross'?'十字阵':'面阵'} · ${count(d)} ch`:d.preset?('预设 · '+d.preset):d.components.map(c=>c.kind.toUpperCase()).join(' + ')}</small></div></div>`).join('');}
 $$('.object').forEach(el=>{el.onclick=()=>select(el.dataset.type,Number(el.dataset.index));el.onkeydown=e=>{if(e.key==='Enter')el.click()}});
 $$('[data-panel]').forEach(b=>b.classList.toggle('active',b.dataset.panel===selected.type));
 const channels=config.receivers.reduce((n,r)=>n+count(r),0);$('#m-carrier').textContent=(config.carrier_frequency_hz/1e9).toFixed(2)+' GHz';$('#m-rate').textContent=(config.sample_rate_hz/1e6).toFixed(2)+' MS/s';$('#m-duration').textContent=+(config.duration_s*1e3).toFixed(3)+' ms';$('#m-channels').textContent=channels+' ch';$('#m-size').textContent=(channels*config.sample_rate_hz*config.duration_s*8/1e6).toFixed(1)+' MB';
 $('#backend-tag').textContent={sionna_rt:'Sionna RT · '+config.propagation.scene,free_space:'自由空间 · 建筑不参与传播',tapped:'抽头信道 · 建筑不参与传播'}[config.propagation.backend];
 map?.update(config,selected);if(inspector)renderInspector();
}
function count(r){return r.antenna.geometry_type==='cross'?(r.antenna.horizontal_elements||0)+(r.antenna.vertical_elements||0):(r.antenna.num_rows||0)*(r.antenna.num_cols||0)}
async function loadRuns(){const runs=await api('runs');$('#run-select').innerHTML=runs.map(r=>`<option value="${esc(r.id)}">${esc(r.name)} · ${{complete:'完成',running:'运行中',queued:'等待',failed:'失败',interrupted:'已中断'}[r.status]||r.status}</option>`).join('');if(currentRun&&runs.some(r=>r.id===currentRun))$('#run-select').value=currentRun;else if(runs.length){currentRun=runs.find(r=>r.id==='example_case01')?.id||runs[0].id;$('#run-select').value=currentRun}if(currentRun)await loadRun();}
async function loadRun(){if(!currentRun)return;const id=currentRun;const data=await api('runs/'+id);if(id!==currentRun)return;runData=data;const option=[...$('#run-select').options].find(o=>o.value===id);if(option)option.textContent=data.name+' · '+({complete:'完成',running:'运行中',queued:'等待',failed:'失败',interrupted:'已中断'}[data.status]||data.status);renderResults();clearTimeout(poll);if(['running','queued'].includes(data.status))poll=setTimeout(()=>attempt(loadRun),2000);$('#run').disabled=['running','queued'].includes(data.status);$('#run').textContent=$('#run').disabled?'◌ 仿真进行中':'▶ 运行仿真';}
function renderResults(){
 const d=runData;if(!d)return;$('#job-status').textContent=(d.imported?'示例记录 · ':'')+({complete:'已完成',running:'运行中',queued:'等待执行',failed:'运行失败',interrupted:'已中断'}[d.status]||d.status);const el=$('#result-content');
 if(resultTab==='log'){el.innerHTML=`<pre>${esc((d.error?d.error+'\n\n':'')+d.log)}</pre>`;return}
 if(resultTab==='files'){el.innerHTML=`<p class="result-note">public/ 为混合观测与采集元数据；private/ 为仿真真值。交给分析 Agent 时仅提供 public/。</p><div class="files">${d.files.map(f=>`<a class="file-row" href="/api/runs/${d.id}/files/${f.split('/').map(encodeURIComponent).join('/')}"><span>${esc(f)}</span><small>下载 ↗</small></a>`).join('')}</div>`;return}
 if(resultTab==='truth'){
  if(!d.truth){el.innerHTML='<div class="empty">仿真完成后显示每条链路的路径数量和接收功率。</div>';return}
  el.innerHTML=`<div class="result-note">仿真真值，仅供建模与评估。路径数不等于物理源数；零路径不等于绝对不可见。下表功率为前端处理前的逐源相干多径贡献。</div><table><thead><tr><th>发射源 → 监测站</th><th>路径数</th><th>平均端口功率</th></tr></thead><tbody>${d.truth.stations.flatMap(s=>s.sources.map(t=>`<tr><td>${esc(t.emitter_id)} → ${esc(s.receiver_id)}</td><td>${t.path_count}</td><td>${(10*Math.log10(Math.max(t.mean_port_power_w.reduce((a,b)=>a+b,0)/t.mean_port_power_w.length,1e-30))+30).toFixed(2)} dBm</td></tr>`)).join('')}</tbody></table><div class="form-actions"><button id="restore-run">将该记录配置载入编辑器</button></div>`;$('#restore-run').onclick=()=>{config=clone(d.config);select('global');map?.reset();toast('已载入该记录配置，尚未重新运行。')};return;
 }
 if(!d.preview){el.innerHTML=`<div class="empty">${d.status==='failed'?esc(d.error):'仿真运行后生成频谱与时频图。'}<br><br>请查看运行日志。</div>`;return}
 const summary=d.summary;el.innerHTML=`<p class="result-note">${esc(d.config?.name||d.name)} · ${summary?.backend||''} · ${summary?.path_count??'—'} 条求解路径 · ${summary?.total_seconds?.toFixed(1)??'—'} s 仿真<br>以下为所选运行记录；修改上方配置不会改变已保存结果。频率相对各接收机调谐中心，逐阵元功率谱平均。</p>`+d.preview.stations.map((r,i)=>`<div class="station-card"><div class="station-title"><strong>▲ ${esc(r.id)}</strong><span>${r.channels} ch · ${r.samples.toLocaleString()} samples · ${r.power_dbm.toFixed(1)} dBm</span></div><div class="charts"><div><div class="chart-title">功率谱密度 · Welch</div><canvas data-chart="${i}" data-kind="psd"></canvas></div><div><div class="chart-title">时频分布 · Spectrogram</div><canvas data-chart="${i}" data-kind="heat"></canvas></div></div></div>`).join('');
 $$('[data-chart]').forEach(c=>plot(c,d.preview.stations[+c.dataset.chart],c.dataset.kind==='heat'));
}
$$('[data-panel]').forEach(b=>b.onclick=()=>select(b.dataset.panel));
$$('[data-result]').forEach(b=>b.onclick=()=>{resultTab=b.dataset.result;$$('[data-result]').forEach(x=>x.classList.toggle('active',x===b));renderResults()});
$('#view-2d').onclick=()=>mode('2d');$('#view-3d').onclick=()=>mode('3d');function mode(m){map?.setMode(m);$('.compass').innerHTML=m==='2d'?'Y<br>↑':'Z ↑';$('#view-2d').classList.toggle('active',m==='2d');$('#view-3d').classList.toggle('active',m==='3d');$('#map-help').textContent=m==='2d'?'拖动设备调整 X/Y · 滚轮缩放 · 高度在属性中设置':'拖动旋转 · 滚轮缩放 · 点击设备编辑';}
$('#reset-view').onclick=()=>map?.reset();$('#buildings').onchange=e=>map?.toggleBuildings(e.target.checked);$('#links').onchange=e=>{if(map)map.showLinks=e.target.checked;refresh(false);if(e.target.checked)toast('虚线仅表示布站配对，不是 RT 反射路径。')};$('#labels').onchange=e=>{if(map)map.showLabels=e.target.checked;refresh(false)};
$('#preset').onchange=()=>{config=clone(boot.presets.find(p=>p.id===$('#preset').value).config);select('global');attempt(()=>applyScene(config.propagation.scene,false));toast('已载入场景配置（地图网格已按该场景切换）。下方保留所选历史运行结果。')};
$('#add-tx').onclick=()=>{const d=clone(boot.defaults.emitter);d.id=uniqueId('tx_');config.emitters.push(d);select('emitters',config.emitters.length-1)};$('#add-rx').onclick=()=>{const d=clone(boot.defaults.receiver);d.id=uniqueId('rx_');d.antenna=clone(config.receivers[0].antenna);config.receivers.push(d);select('receivers',config.receivers.length-1)};
$('#validate').onclick=()=>attempt(async()=>{const r=await api('validate',config);$('#validation-result').textContent=`配置有效\n\n每通道 ${r.num_samples.toLocaleString()} 个采样\nIQ 输出 ${r.resources.iq_output_mb.toFixed(1)} MB\n预计 DSP 工作内存 ${r.resources.estimated_dsp_working_mb.toFixed(1)} MB\n\n${r.warnings.join('\n')}`;$('#validation-dialog').showModal()});$('#close-validation').onclick=()=>$('#validation-dialog').close();
async function saved(){const data=await api('scenarios');$('#saved-list').replaceChildren();for(const s of data)button($('#saved-list'),s.config.name,()=>{config=clone(s.config);select('global');map?.reset()});}
$('#save').onclick=()=>attempt(async()=>{await api('scenarios',config);await saved();toast('配置已保存到 user_scenarios/')});
$('#export').onclick=()=>{const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify(config,null,2)],{type:'application/json'}));a.href=url;a.download=(config.name||'scenario').replace(/[^\w\u4e00-\u9fff-]/g,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('#import').onclick=()=>$('#import-file').click();$('#import-file').onchange=e=>attempt(async()=>{const f=e.target.files[0];if(!f)return;const r=await api('validate',JSON.parse(await f.text()));config=r.config;select('global');map?.reset();toast('配置已导入并校验');e.target.value=''});
$('#run').onclick=()=>attempt(async()=>{$('#run').disabled=true;try{const r=await api('runs',config);currentRun=r.id;await loadRuns();toast('已启动独立仿真；结束后自动生成各站观测图。')}catch(e){$('#run').disabled=false;throw e}});
$('#run-select').onchange=()=>{currentRun=$('#run-select').value;attempt(loadRun)};
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(resultTab==='spectra')renderResults()},200)});
attempt(async()=>{boot=await api('bootstrap');if(boot.kinds)enums.kind=Object.fromEntries(boot.kinds.map(k=>[k.name,k.label]));$('#runtime').textContent=boot.runtime.wsl_rt?'● DSP + WSL Sionna RT 就绪':boot.runtime.native_rt?'● DSP + Sionna RT 就绪':'● DSP 就绪 / RT 未安装';
 if(boot.meshes?.length){const ms=$('#mesh-select');ms.innerHTML=boot.meshes.map(m=>`<option value="${esc(m.id)}">${esc(m.name)} · ${(m.triangle_count/1000).toFixed(1)}k 面</option>`).join('');ms.value=boot.meshes[0].id;ms.onchange=()=>attempt(()=>applyScene(sceneForMesh(ms.value)));}enums.scene=sceneOptions(boot.meshes);$('#preset').innerHTML=boot.presets.map(p=>`<option value="${esc(p.id)}">${esc(PRESET_LABELS[p.id]||p.label)}</option>`).join('');const initial=boot.presets.find(p=>p.id==='three_scenes/case01.json')||boot.presets[0];$('#preset').value=initial.id;config=clone(initial.config);refresh();attempt(()=>applyScene(config.propagation.scene,false));await Promise.all([loadRuns(),saved()]);});

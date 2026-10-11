// 零依赖自测：把 app.js 里的场景↔网格联动逻辑抠出来，在 Node 里用桩 DOM 跑一遍。
// 目的：证明「切换示例场景 → 地图网格跟着换」这条链路真的成立，而不是靠肉眼看网页。
// 用法：node webui/_check_scene_switch.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, 'assets', 'app.js'), 'utf8');

/** 从源码里抠出一个函数（按大括号配对），避免整份 app.js 的 DOM 依赖。 */
function grab(startMarker) {
  const i = src.indexOf(startMarker);
  if (i < 0) throw new Error(`找不到 ${startMarker}`);
  let depth = 0, j = src.indexOf('{', i);
  for (let k = j; k < src.length; k++) {
    if (src[k] === '{') depth++;
    else if (src[k] === '}') { depth--; if (depth === 0) return src.slice(i, k + 1); }
  }
  throw new Error(`${startMarker} 括号不配对`);
}
const cityScene = src.match(/const CITY_SCENE=.*/)[0];
const sceneForMesh = grab('function sceneForMesh(');
const meshForScene = grab('function meshForScene(');
const applyScene = grab('async function applyScene(');

function makeHarness() {
  const calls = { load: [], toast: [], title: [] };
  const meshSelect = { value: 'munich', options: [{ value: 'munich' }, { value: 'hangzhou' }] };
  const titleEl = { textContent: '' };
  const els = { '#mesh-select': meshSelect, '#scene-title': titleEl };
  const boot = { meshes: [{ id: 'munich', name: '慕尼黑' }, { id: 'hangzhou', name: '杭州 · 市中心' }] };
  const config = { propagation: { scene: 'munich' } };
  const state = { selected: { type: 'global' } };
  const map = { load: async (id) => { calls.load.push(id); if (id === 'boom') throw new Error('模拟加载失败'); } };
  const fn = new Function(
    'config', '$', 'map', 'boot', 'selected', 'toast', 'renderInspector', 'refresh',
    `let loadedMesh=null;\n${cityScene}\n${sceneForMesh}\n${meshForScene}\n${applyScene}\n
     return { applyScene, meshForScene, sceneForMesh, get loaded(){ return loadedMesh; } };`
  );
  const api = fn(
    config,
    (sel) => els[sel],
    map, boot, state.selected,
    (t) => calls.toast.push(t),
    () => {}, () => {}
  );
  return { api, calls, meshSelect, titleEl, config, state };
}

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  ok   ${name}`); }
  else { fail++; console.log(`  FAIL ${name} ${extra}`); }
}

console.log('=== 1) 场景字符串 ↔ 网格 id 映射 ===');
{
  const h = makeHarness();
  check('munich → munich', h.api.meshForScene('munich') === 'munich');
  check('priors/hangzhou_geometry/scene.xml → hangzhou',
    h.api.meshForScene('priors/hangzhou_geometry/scene.xml') === 'hangzhou');
  check('empty → null', h.api.meshForScene('empty') === null);
  check('空值 → null', h.api.meshForScene(undefined) === null);
  check('hangzhou → priors/hangzhou_geometry/scene.xml',
    h.api.sceneForMesh('hangzhou') === 'priors/hangzhou_geometry/scene.xml');
}

console.log('=== 2) 载入 D2（杭州）预设：网格必须跟着换 ===');
{
  const h = makeHarness();
  await h.api.applyScene('priors/hangzhou_geometry/scene.xml', false);
  check('map.load 收到 hangzhou', h.calls.load.join(',') === 'hangzhou', `实际 ${h.calls.load}`);
  check('下拉框切到 hangzhou', h.meshSelect.value === 'hangzhou');
  check('标题更新为杭州', h.titleEl.textContent === '杭州 · 市中心 · 城市电磁空间', `实际 ${h.titleEl.textContent}`);
  check('config.propagation.scene 同步', h.config.propagation.scene === 'priors/hangzhou_geometry/scene.xml');
}

console.log('=== 3) 重复应用同一场景：不重复下载网格 ===');
{
  const h = makeHarness();
  await h.api.applyScene('priors/hangzhou_geometry/scene.xml', false);
  await h.api.applyScene('priors/hangzhou_geometry/scene.xml', false);
  check('只加载一次', h.calls.load.length === 1, `实际 ${h.calls.load.length} 次`);
}

console.log('=== 4) 回到慕尼黑：网格换回去 ===');
{
  const h = makeHarness();
  await h.api.applyScene('priors/hangzhou_geometry/scene.xml', false);
  await h.api.applyScene('munich', false);
  check('load 顺序 = hangzhou, munich', h.calls.load.join(',') === 'hangzhou,munich', `实际 ${h.calls.load}`);
  check('标题回到慕尼黑', h.titleEl.textContent === '慕尼黑 · 城市电磁空间');
}

console.log('=== 5) 预设引用了本地没有的网格：必须出声，不能静默 ===');
{
  const h = makeHarness();
  await h.api.applyScene('priors/kunming_geometry/scene.xml', true);
  check('没有调用 map.load', h.calls.load.length === 0);
  check('给出警告 toast', h.calls.toast.some((t) => t.includes('kunming') && t.includes('没有')), `实际 ${JSON.stringify(h.calls.toast)}`);
}

console.log('=== 6) 网格文件损坏（load 抛错）：给出失败提示且不谎报成功 ===');
{
  const h = makeHarness();
  h.meshSelect.options.push({ value: 'boom' });
  await h.api.applyScene('priors/boom_geometry/scene.xml', true);
  check('提示加载失败', h.calls.toast.some((t) => t.includes('加载失败')), `实际 ${JSON.stringify(h.calls.toast)}`);
  check('未标记为已加载（下次会重试）', h.api.loaded !== 'boom');
}

console.log('=== 7) 空场景：不动网格，走原有提示 ===');
{
  const h = makeHarness();
  await h.api.applyScene('empty', true);
  check('没有调用 map.load', h.calls.load.length === 0);
  check('提示「空场景」', h.calls.toast.some((t) => t.includes('空场景')), `实际 ${JSON.stringify(h.calls.toast)}`);
}

console.log(`\n${fail === 0 ? '全部通过' : '有失败项'}：${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);

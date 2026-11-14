// 博物馆数字碎片配准工作台前端
// 三个 Three.js 视口：目标、源、叠合残差。所有配准计算在后端，
// 前端只做可视化与交互，不在浏览器里手工拖拽冒充算法结果。

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const state = {
  session: null,
  geometry: null,
  selectedScheme: null,
  activePair: null,
  pickMode: null, // 'target' | 'source' | null
  residualMax: 1,
};

function toast(msg, ms = 3200) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add('hidden'), ms);
}

async function api(path, method = 'GET', body) {
  const res = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---------- Three.js 视口封装 ----------
class Viewer {
  constructor(containerId, { picking = false } = {}) {
    const el = $(containerId);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x10141a);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    this.camera.position.set(260, 220, 320);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    el.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(200, 300, 180);
    this.scene.add(dir);
    this.grid = new THREE.GridHelper(400, 20, 0x3a4452, 0x232b36);
    this.scene.add(this.grid);
    this.cloudGroup = new THREE.Group();
    this.meshGroup = new THREE.Group();
    this.markerGroup = new THREE.Group();
    this.scene.add(this.cloudGroup, this.meshGroup, this.markerGroup);
    this.clipPlane = null;
    this.picking = picking;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.onPick = null;
    this.pickSurfaces = [];
    if (picking) this._bindPick();
    this._resize = () => this.resize();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    this._loop();
  }

  resize() {
    const el = this.renderer.domElement.parentElement;
    const w = el.clientWidth; const h = el.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  _bindPick() {
    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => { this._down = { x: e.clientX, y: e.clientY }; });
    dom.addEventListener('pointerup', (e) => {
      if (!this.onPick || !this._down) return;
      const moved = Math.hypot(e.clientX - this._down.x, e.clientY - this._down.y);
      this._down = null;
      if (moved > 5) return; // 拖拽旋转时不取点
      const rect = dom.getBoundingClientRect();
      this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hits = this.raycaster.intersectObjects(this.pickSurfaces, false);
      if (hits.length) this.onPick(hits[0].point);
    });
  }

  clearGroups() {
    for (const g of [this.cloudGroup, this.meshGroup, this.markerGroup]) {
      while (g.children.length) {
        const c = g.children.pop();
        c.geometry?.dispose?.();
        if (c.material) {
          c.material.map?.dispose?.();
          c.material.dispose?.();
        }
      }
    }
    this.pickSurfaces = [];
  }

  clearMarkers() {
    while (this.markerGroup.children.length) {
      const c = this.markerGroup.children.pop();
      c.geometry?.dispose?.();
      if (c.material) {
        c.material.map?.dispose?.();
        c.material.dispose?.();
      }
    }
  }

  setCloud(points, normals, color, { pickable = false } = {}) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3));
    if (normals) geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals.flat(), 3));
    const mat = new THREE.PointsMaterial({
      color, size: 1.6, sizeAttenuation: true,
      clippingPlanes: this.clipPlane ? [this.clipPlane] : [],
    });
    const pts = new THREE.Points(geo, mat);
    this.cloudGroup.add(pts);
    if (pickable) {
      // 用不可见的放大点云作为拾取面，提高点击命中率
      const hitMat = new THREE.PointsMaterial({ size: 6, sizeAttenuation: true, colorWrite: false });
      const hitPts = new THREE.Points(geo, hitMat);
      this.cloudGroup.add(hitPts);
      this.pickSurfaces.push(hitPts);
    }
    return pts;
  }

  setCloudVertexColors(points, colors, normals) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors.flat(), 3));
    if (normals) geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals.flat(), 3));
    const mat = new THREE.PointsMaterial({
      size: 1.8, sizeAttenuation: true, vertexColors: true,
      clippingPlanes: this.clipPlane ? [this.clipPlane] : [],
    });
    const pts = new THREE.Points(geo, mat);
    this.cloudGroup.add(pts);
  }

  setMesh(vertices, faces, color, opacity = 0.35) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flat(), 3));
    geo.setIndex(faces.flat());
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color, transparent: true, opacity, side: THREE.DoubleSide,
      clippingPlanes: this.clipPlane ? [this.clipPlane] : [],
    });
    const mesh = new THREE.Mesh(geo, mat);
    this.meshGroup.add(mesh);
    this.pickSurfaces.push(mesh);
    return mesh;
  }

  addMarker(point, color, label) {
    const geo = new THREE.SphereGeometry(3.2, 14, 14);
    const mat = new THREE.MeshBasicMaterial({ color });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.fromArray(point);
    this.markerGroup.add(sphere);
    if (label != null) {
      const canvas = document.createElement('canvas');
      canvas.width = 96; canvas.height = 48;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000a'; ctx.fillRect(0, 0, 96, 36);
      ctx.fillStyle = '#fff'; ctx.font = '22px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(String(label), 48, 27);
      const tex = new THREE.CanvasTexture(canvas);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
      sp.scale.set(18, 9, 1);
      sp.position.fromArray(point).add(new THREE.Vector3(0, 6, 0));
      this.markerGroup.add(sp);
    }
    return sphere;
  }

  addCorrespondenceLine(a, b) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...a), new THREE.Vector3(...b),
    ]);
    const line = new THREE.Line(geo,
      new THREE.LineBasicMaterial({ color: 0xf2e08a, clippingPlanes: this.clipPlane ? [this.clipPlane] : [] }));
    line.userData.isCorr = true;
    this.markerGroup.add(line);
  }

  frameContent() {
    const box = new THREE.Box3();
    this.cloudGroup.children.forEach((c) => box.setFromObject(c));
    this.meshGroup.children.forEach((c) => box.setFromObject(c));
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 50);
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(maxDim * 0.9, maxDim * 0.7, maxDim * 1.1));
    this.camera.near = maxDim / 100;
    this.camera.far = maxDim * 20;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.grid.position.y = box.min.y - 4;
  }

  setClip(axis, value) {
    if (axis === 'none' || value == null) {
      this.clipPlane = null;
    } else {
      const normal = new THREE.Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
      this.clipPlane = new THREE.Plane(normal, -value);
    }
    [...this.cloudGroup.children, ...this.meshGroup.children].forEach((obj) => {
      if (obj.material) obj.material.clippingPlanes = this.clipPlane ? [this.clipPlane] : [];
    });
  }

  _loop() {
    requestAnimationFrame(() => this._loop());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

let targetViewer, sourceViewer, overlapViewer;

// ---------- 数据载入 ----------
async function loadDataset() {
  const scenario = $('scenario').value;
  const seed = Number($('seed').value) || 42;
  state.session = await api('/api/datasets', 'POST', { scenario, seed });
  state.geometry = await api(`/api/sessions/${state.session.sessionId}/geometry`);
  state.selectedScheme = null;
  const m = state.session.meta;
  $('conventions').textContent = `单位 ${m.unit} · ${m.handedness === 'right' ? '右手系' : '左手系'} · 上轴 ${m.upAxis} · 场景 ${m.scenario}`;
  $('datasetInfo').innerHTML =
    `目标点 ${state.session.sizes.targetCloud} / 源点 ${state.session.sizes.sourceCloud}<br>` +
    `三角面 目标 ${state.session.sizes.targetFaces} / 源 ${state.session.sizes.sourceFaces}<br>` +
    `真值旋转角 ${m.groundTruth.angleDeg.toFixed(1)}°（仅用于验证，前端不参与计算）`;
  $('registerBtn').disabled = false;
  renderPairList();
  renderStaticViewers();
  renderOverlap(null);
  $('schemeList').textContent = '尚未配准';
  $('diagnostics').textContent = '选择方案后显示';
  $('outlierList').textContent = '无';
  $('confirmRecord').textContent = '未确认';
  ['icpBtn', 'acceptBtn', 'rejectBtn', 'exportBtn'].forEach((id) => $(id).disabled = true);
  toast(`已载入 ${scenario} 数据集（会话 ${state.session.sessionId.slice(0, 8)}）`);
}

function renderStaticViewers() {
  const g = state.geometry;
  targetViewer.clearGroups();
  sourceViewer.clearGroups();
  targetViewer.setCloud(g.target.cloud.points, g.target.cloud.normals, 0x6fb1ff, { pickable: true });
  sourceViewer.setCloud(g.source.cloud.points, g.source.cloud.normals, 0xff9d5c, { pickable: true });
  if ($('showMesh').checked) {
    targetViewer.setMesh(g.target.mesh.vertices, g.target.mesh.faces, 0x6fb1ff);
    sourceViewer.setMesh(g.source.mesh.vertices, g.source.mesh.faces, 0xff9d5c);
  }
  refreshMarkers();
  targetViewer.frameContent();
  sourceViewer.frameContent();
}

function refreshMarkers() {
  targetViewer.clearMarkers();
  sourceViewer.clearMarkers();
  state.session.correspondences.forEach((c, i) => {
    targetViewer.addMarker(c.targetPoint, 0xf2e08a, i);
    sourceViewer.addMarker(c.sourcePoint, 0xf2e08a, i);
  });
}

// ---------- 对应点管理 ----------
function renderPairList() {
  const list = $('pairList');
  list.innerHTML = '';
  state.session.correspondences.forEach((c, i) => {
    const item = document.createElement('div');
    item.className = 'pair-item' + (state.activePair === i ? ' active' : '');
    const corrupt = c.note && c.note.includes('污染');
    item.innerHTML = `
      <div class="pid">#${i} 对应点 ${corrupt ? '<span class="corrupt">（真值标记：污染）</span>' : ''}</div>
      <div class="coords">T [${c.targetPoint.map((x) => x.toFixed(1)).join(', ')}]</div>
      <div class="coords">S [${c.sourcePoint.map((x) => x.toFixed(1)).join(', ')}]</div>
      <div class="weights">
        <label>目标权重 <input type="number" min="0.01" step="0.1" value="${c.targetWeight}" data-i="${i}" data-w="t"></label>
        <label>源权重 <input type="number" min="0.01" step="0.1" value="${c.sourceWeight}" data-i="${i}" data-w="s"></label>
      </div>
      <button class="del ghost" data-del="${i}">删除该对</button>`;
    list.appendChild(item);
  });
  list.querySelectorAll('input[data-i]').forEach((inp) => {
    inp.addEventListener('change', async () => {
      const i = Number(inp.dataset.i);
      const c = state.session.correspondences[i];
      if (inp.dataset.w === 't') c.targetWeight = Number(inp.value);
      else c.sourceWeight = Number(inp.value);
      await saveCorrespondences();
    });
  });
  list.querySelectorAll('button[data-del]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      state.session.correspondences.splice(Number(btn.dataset.del), 1);
      state.activePair = null;
      await saveCorrespondences();
      renderPairList();
      refreshMarkers();
    });
  });
}

async function saveCorrespondences() {
  state.session = await api(`/api/sessions/${state.session.sessionId}/correspondences`,
    'PUT', { correspondences: state.session.correspondences });
}

function setupPicking() {
  targetViewer.onPick = (p) => handlePick('target', p);
  sourceViewer.onPick = (p) => handlePick('source', p);
}

async function handlePick(kind, point) {
  if (state.activePair == null) {
    toast('请先点击“添加对应点对”，再在视口取点');
    return;
  }
  const c = state.session.correspondences[state.activePair];
  const arr = [point.x, point.y, point.z];
  if (kind === 'target') c.targetPoint = arr;
  else c.sourcePoint = arr;
  await saveCorrespondences();
  renderPairList();
  refreshMarkers();
  toast(`已记录第 ${state.activePair} 对的${kind === 'target' ? '目标' : '源'}点`);
}

// ---------- 配准 ----------
async function runRegister() {
  const body = {};
  const it = Number($('inlierThreshold').value);
  const od = Number($('overlapDistance').value);
  if (Number.isFinite(it) && it > 0) body.inlierThreshold = it;
  if (Number.isFinite(od) && od > 0) body.overlapDistance = od;
  state.session = await api(`/api/sessions/${state.session.sessionId}/register`, 'POST', body);
  renderSchemes();
  const first = state.session.schemes[0];
  if (first) selectScheme(first.id);
  toast(`生成 ${state.session.schemes.length} 个候选方案，请结合警告人工甄别`);
}

function renderSchemes() {
  const el = $('schemeList');
  el.innerHTML = '';
  state.session.schemes.forEach((s) => {
    const div = document.createElement('div');
    div.className = 'scheme-item' + (state.selectedScheme === s.id ? ' active' : '');
    div.innerHTML = `
      <div class="head">
        <span>#${s.id} <span class="badge ${s.source}">${s.source}</span></span>
        <span>${(s.overlapFraction * 100).toFixed(0)}% 重叠</span>
      </div>
      <div class="metric">${s.label}</div>
      <div class="metric">对应点 RMS ${s.rms == null ? '—' : s.rms.toFixed(3)} mm ·
        点云 RMS ${s.cloudRms == null ? '—' : s.cloudRms.toFixed(3)} mm ·
        内点 ${(s.inlierIndices || []).length}</div>
      ${s.warnings.length ? s.warnings.map((w) => `<div class="warning">${w}</div>`).join('') : '<div class="ok-text small">无额外警告</div>'}`;
    div.addEventListener('click', () => selectScheme(s.id));
    el.appendChild(div);
  });
}

async function selectScheme(id) {
  state.selectedScheme = id;
  renderSchemes();
  const s = state.session.schemes.find((x) => x.id === id);
  renderOverlap(s);
  renderDiagnostics(s);
  renderOutliers(s);
  ['icpBtn', 'acceptBtn', 'rejectBtn', 'exportBtn'].forEach((bid) => $(bid).disabled = false);
}

// ---------- 叠合与残差可视化 ----------
function applyTransformArr(points, tf) {
  const R = tf.rotation; const t = tf.translation;
  return points.map((p) => [
    R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0],
    R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1],
    R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2],
  ]);
}

function residualColor(d, dmax) {
  const t = Math.min(1, d / Math.max(dmax, 1e-9));
  // 绿 -> 黄 -> 红
  const r = t < 0.5 ? 0.18 + t * 1.2 : 1;
  const g = t < 0.5 ? 0.78 : 0.78 - (t - 0.5) * 1.4;
  const b = 0.25 - t * 0.2;
  return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, b)];
}

function renderOverlap(scheme) {
  overlapViewer.clearGroups();
  const g = state.geometry;
  // 目标始终原样显示
  overlapViewer.setCloud(g.target.cloud.points, g.target.cloud.normals, 0x6fb1ff);
  if (!scheme) {
    overlapViewer.setCloud(g.source.cloud.points, g.source.cloud.normals, 0xff9d5c);
    applyClip();
    overlapViewer.frameContent();
    return;
  }
  const moved = applyTransformArr(g.source.cloud.points, scheme.transform);
  if ($('showResidual').checked && scheme.pointResiduals) {
    const valid = scheme.pointResiduals.filter((d) => d != null);
    state.residualMax = Math.max(...valid, 1e-9);
    const colors = scheme.pointResiduals.map((d) =>
      d == null ? [0.6, 0.6, 0.6] : residualColor(d, state.residualMax));
    overlapViewer.setCloudVertexColors(moved, colors, g.source.cloud.normals);
  } else {
    overlapViewer.setCloud(moved, g.source.cloud.normals, 0xff9d5c);
  }
  if ($('showMesh').checked) {
    const movedVerts = applyTransformArr(g.source.mesh.vertices, scheme.transform);
    overlapViewer.setMesh(movedVerts, g.source.mesh.faces, 0xff9d5c, 0.22);
    overlapViewer.setMesh(g.target.mesh.vertices, g.target.mesh.faces, 0x6fb1ff, 0.18);
  }
  // 对应点连线（仅当前方案内点）
  const inSet = new Set(scheme.inlierIndices || []);
  state.session.correspondences.forEach((c, i) => {
    if (!inSet.has(i)) return;
    const mp = applyTransformArr([c.sourcePoint], scheme.transform)[0];
    overlapViewer.addCorrespondenceLine(mp, c.targetPoint);
  });
  applyClip();
  overlapViewer.frameContent();
}

function renderDiagnostics(s) {
  const d = s.diagnostics || {};
  const deg = d.degeneracy || {};
  const rows = [
    ['变换行列式', d.determinant == null ? '—' : d.determinant.toFixed(6)],
    ['旋转角', d.rotationAngleDeg == null ? '—' : `${d.rotationAngleDeg.toFixed(2)}°`],
    ['条件数', d.conditionNumber == null ? '—' : (d.conditionNumber > 1e6 ? d.conditionNumber.toExponential(2) : d.conditionNumber.toFixed(2))],
    ['几何秩', deg.rank == null ? '—' : `${deg.rank}（${deg.rank < 3 ? '退化' : '满秩'}）`],
    ['共面比', deg.planarRatio == null ? '—' : deg.planarRatio.toExponential(2)],
    ['点云重叠率', `${(s.overlapFraction * 100).toFixed(1)}%`],
    ['未匹配源点', s.unmatchedSourceCount],
    ['点云 RMS', s.cloudRms == null ? '—' : `${s.cloudRms.toFixed(3)} mm`],
    ['点云中位残差', s.cloudMedian == null ? '—' : `${s.cloudMedian.toFixed(3)} mm`],
  ];
  $('diagnostics').innerHTML =
    rows.map(([k, v]) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`).join('') +
    (s.icp ? `<div class="kv"><span>ICP</span><span>${s.icp.converged ? '收敛' : '未收敛'} · ${s.icp.iterations} 轮 · 排除 ${s.icp.rejectedCount} 对</span></div>
      <div class="small muted">${s.icp.reason}</div>` : '');
}

function renderOutliers(s) {
  const el = $('outlierList');
  const report = s.outlierReport || [];
  const excluded = report.filter((r) => r.excluded);
  if (!excluded.length) { el.innerHTML = '<span class="ok-text">无被排除对应点</span>'; return; }
  el.innerHTML = excluded.map((r) =>
    `<div class="outlier-item"><strong>#${r.pairIndex}</strong> 残差 ${r.residual.toFixed(3)} mm<div class="muted">${r.reason}</div></div>`).join('');
}

// ---------- ICP / 确认 / 导出 ----------
async function runIcp() {
  if (state.selectedScheme == null) return;
  try {
    state.session = await api(
      `/api/sessions/${state.session.sessionId}/schemes/${state.selectedScheme}/icp`,
      'POST', {});
    renderSchemes();
    const newIcp = state.session.schemes.filter((s) => s.source === 'icp').pop();
    selectScheme(newIcp.id);
    toast('ICP 完成，结果已另存为新方案（原方案与原始模型保留）');
  } catch (e) { toast(e.message); }
}

async function confirm(decision) {
  const operator = $('operator').value.trim();
  if (!operator) { toast('请先填写操作员姓名'); return; }
  state.session = await api(
    `/api/sessions/${state.session.sessionId}/schemes/${state.selectedScheme}/confirm`,
    'POST', { operator, decision, note: '浏览器人工确认' });
  renderConfirmRecord();
  toast(`已记录${decision === 'accepted' ? '采纳' : '否决'}确认（绑定输入快照与参数）`);
}

function renderConfirmRecord() {
  const c = state.session.confirmation;
  if (!c) { $('confirmRecord').textContent = '未确认'; return; }
  $('confirmRecord').innerHTML =
    `<div class="kv"><span>方案</span><span>#${c.schemeId}</span></div>
     <div class="kv"><span>操作员</span><span>${c.operator}</span></div>
     <div class="kv"><span>决定</span><span class="${c.decision === 'accepted' ? 'ok-text' : 'danger-text'}">${c.decision}</span></div>
     <div class="kv"><span>输入指纹</span><span>${c.inputFingerprint}</span></div>
     <div class="kv"><span>时间</span><span>${new Date(c.timestamp).toLocaleString()}</span></div>
     <div class="muted">对应点快照 ${c.correspondenceSnapshot.length} 对；参数 ${JSON.stringify(c.params)}</div>`;
}

async function exportScheme() {
  const url = `/api/sessions/${state.session.sessionId}/schemes/${state.selectedScheme}/export`;
  const res = await fetch(url);
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `registration_${state.session.sessionId.slice(0, 8)}_scheme${state.selectedScheme}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
  toast('已导出含约定、变换、原始/变换点云与确认记录的可复算 JSON');
}

// ---------- 剖面 ----------
function applyClip() {
  const axis = $('clipAxis').value;
  if (axis === 'none') {
    overlapViewer.setClip('none');
    return;
  }
  // 用目标点云包围盒把滑块 -1..1 映射到实际坐标
  const pts = state.geometry.target.cloud.points;
  let mn = Infinity; let mx = -Infinity;
  const k = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
  for (const p of pts) { mn = Math.min(mn, p[k]); mx = Math.max(mx, p[k]); }
  const value = mn + (($('clipPos').valueAsNumber + 1) / 2) * (mx - mn);
  overlapViewer.setClip(axis, value);
}

// ---------- 初始化 ----------
function init() {
  targetViewer = new Viewer('targetView', { picking: true });
  sourceViewer = new Viewer('sourceView', { picking: true });
  overlapViewer = new Viewer('overlapView', { picking: false });
  setupPicking();
  $('loadBtn').addEventListener('click', () => loadDataset().catch((e) => toast(e.message)));
  $('addPairBtn').addEventListener('click', async () => {
    if (!state.session) { toast('请先载入数据集'); return; }
    const id = state.session.correspondences.length;
    state.session.correspondences.push({
      id, targetPoint: [0, 0, 0], sourcePoint: [0, 0, 0],
      targetWeight: 1, sourceWeight: 1, note: '用户新增对应点（请在视口取点）',
    });
    state.activePair = id;
    await saveCorrespondences();
    renderPairList();
    toast(`新增第 ${id} 对：先点目标视口，再点源视口`);
  });
  $('clearPairsBtn').addEventListener('click', async () => {
    if (!state.session) return;
    state.session.correspondences = [];
    state.activePair = null;
    await saveCorrespondences();
    renderPairList();
    refreshMarkers();
  });
  $('registerBtn').addEventListener('click', () => runRegister().catch((e) => toast(e.message)));
  $('icpBtn').addEventListener('click', () => runIcp().catch((e) => toast(e.message)));
  $('acceptBtn').addEventListener('click', () => confirm('accepted').catch((e) => toast(e.message)));
  $('rejectBtn').addEventListener('click', () => confirm('rejected').catch((e) => toast(e.message)));
  $('exportBtn').addEventListener('click', () => exportScheme().catch((e) => toast(e.message)));
  $('showMesh').addEventListener('change', () => { renderStaticViewers(); if (state.selectedScheme != null) selectScheme(state.selectedScheme); });
  $('showResidual').addEventListener('change', () => { if (state.selectedScheme != null) selectScheme(state.selectedScheme); });
  $('clipAxis').addEventListener('change', applyClip);
  $('clipPos').addEventListener('input', applyClip);
  // 默认载入一个完整场景，便于直接体验
  loadDataset().catch((e) => toast(e.message));
}

init();

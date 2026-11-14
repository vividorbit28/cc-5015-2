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
  // 拖动初始位姿：basePose 是内核候选方案的位姿，滑块只叠加内核允许的增量
  poseBase: null,       // { rotation, translation, schemeId, label }
  poseDeltas: { rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0 }, // 角度/毫米
  preview: null,        // 最近一次 /diagnose 返回
  previewTimer: null,
  previewLoading: false,
};

const DEG = Math.PI / 180;

// 绕 X/Y/Z 顺序叠加小角度旋转（内核输出仍为 rigid，det=+1）
function rotationX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [1, 0, 0, 0, c, -s, 0, s, c];
}
function rotationY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}
function rotationZ(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}
function matMul(a, b) {
  const r = new Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return r;
}
function matApply(m, p) {
  return [
    m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
    m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
    m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
  ];
}

function effectivePose() {
  if (!state.poseBase) return null;
  const Rz = rotationZ(state.poseDeltas.rz * DEG);
  const Ry = rotationY(state.poseDeltas.ry * DEG);
  const Rx = rotationX(state.poseDeltas.rx * DEG);
  // 增量作用在世界系：R = Rz · Ry · Rx · R_base
  const rotation = matMul(matMul(Rz, matMul(Ry, Rx)), state.poseBase.rotation);
  // 平移增量沿增量旋转后的基底；绕原点转动后再叠加纯平移，直观对应“拖动”
  const rotatedT = matApply(matMul(Rz, matMul(Ry, Rx)), state.poseBase.translation);
  const translation = [
    rotatedT[0] + state.poseDeltas.tx,
    rotatedT[1] + state.poseDeltas.ty,
    rotatedT[2] + state.poseDeltas.tz,
  ];
  return { rotation, translation };
}

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

  addCorrespondenceLine(a, b, color = 0xf2e08a) {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...a), new THREE.Vector3(...b),
    ]);
    const line = new THREE.Line(geo,
      new THREE.LineBasicMaterial({ color, clippingPlanes: this.clipPlane ? [this.clipPlane] : [] }));
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
  resetPoseControls();
  renderPreviews(null);
  $('evidencePanel').textContent = '对方案执行 ICP 后显示逐轮证据链';
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
  renderPreviews(null);
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
  // 内核候选位姿就绪：交由内核驱动拖动控件，而不是前端自造位姿
  syncPoseOptions();
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
  renderEvidence(s);
  ['icpBtn', 'acceptBtn', 'rejectBtn', 'exportBtn'].forEach((bid) => $(bid).disabled = false);
  syncPoseOptions(s);
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

function renderOverlap(scheme, overridePose = null) {
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
  const pose = overridePose || scheme.transform;
  const moved = applyTransformArr(g.source.cloud.points, pose);
  // 预览时残差来自 /api/diagnose（与源点云同索引）；正式方案来自方案记录
  const residuals = overridePose && state.preview ? state.preview.coverage.pointResiduals
    : scheme.pointResiduals;
  if ($('showResidual').checked && residuals) {
    const valid = residuals.filter((d) => d != null);
    state.residualMax = Math.max(...valid, 1e-9);
    const colors = residuals.map((d) =>
      d == null ? [0.55, 0.55, 0.55] : residualColor(d, state.residualMax));
    overlapViewer.setCloudVertexColors(moved, colors, g.source.cloud.normals);
  } else {
    overlapViewer.setCloud(moved, g.source.cloud.normals, 0xff9d5c);
  }
  if ($('showMesh').checked) {
    const movedVerts = applyTransformArr(g.source.mesh.vertices, pose);
    overlapViewer.setMesh(movedVerts, g.source.mesh.faces, 0xff9d5c, 0.22);
    overlapViewer.setMesh(g.target.mesh.vertices, g.target.mesh.faces, 0x6fb1ff, 0.18);
  }
  // 对应点连线：预览按逐对原因着色（内点=黄，超阈值=红）
  if (overridePose && state.preview) {
    const items = state.preview.correspondencePoints.items;
    state.session.correspondences.forEach((c, i) => {
      const item = items[i];
      const mp = applyTransformArr([c.sourcePoint], pose)[0];
      overlapViewer.addCorrespondenceLine(mp, c.targetPoint, item && item.inlier ? 0xf2e08a : 0xe5604f);
    });
  } else {
    const inSet = new Set(scheme.inlierIndices || []);
    state.session.correspondences.forEach((c, i) => {
      if (!inSet.has(i)) return;
      const mp = applyTransformArr([c.sourcePoint], pose)[0];
      overlapViewer.addCorrespondenceLine(mp, c.targetPoint);
    });
  }
  applyClip();
  // 拖动预览时不再自动重置相机，避免每次滑动镜头跳动
  if (!overridePose) overlapViewer.frameContent();
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
    syncPoseOptions(newIcp);
    if (newIcp.replay && newIcp.replay.stable === false) {
      toast('ICP 完成，但逐轮重放不一致：证据不可信，已在证据面板标红');
    } else if (newIcp.evidence && newIcp.evidence.cause && newIcp.evidence.cause.primary !== 'NONE') {
      toast(`ICP 完成（${newIcp.icp.stopCode}）：诊断主因 ${newIcp.evidence.cause.primary}`);
    } else {
      toast('ICP 完成，结果已另存为新方案（原方案与原始模型保留）');
    }
  } catch (e) { toast(`ICP 失败：${e.message}`); }
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

// ---------- 拖动初始位姿（内核驱动） ----------
// 控件选项完全来自内核产出的候选方案；没有候选时控件禁用，
// 不提供“前端凭空造位姿”的入口，避免结果与配准内核脱节。
function syncPoseOptions(selected = null) {
  const select = $('poseSchemeSelect');
  const schemes = state.session ? state.session.schemes : [];
  select.innerHTML = schemes.length
    ? schemes.map((s) => `<option value="${s.id}">#${s.id} ${s.label}</option>`).join('')
    : '<option value="">（先配准，生成候选位姿）</option>';
  const useId = selected ? selected.id : (state.selectedScheme != null ? state.selectedScheme : (schemes[0] && schemes[0].id));
  if (useId != null) select.value = String(useId);
  select.disabled = !schemes.length;
  setPoseBaseFromScheme();
}

function setPoseBaseFromScheme() {
  const id = Number($('poseSchemeSelect').value);
  const scheme = state.session && state.session.schemes.find((s) => s.id === id);
  if (!scheme) { resetPoseControls(); return; }
  state.poseBase = {
    rotation: scheme.transform.rotation.slice(),
    translation: scheme.transform.translation.slice(),
    schemeId: id,
    label: scheme.label,
  };
  state.poseDeltas = { rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0 };
  setPoseControlsEnabled(true);
  syncPoseSliderLabels();
  requestPreview();
}

function setPoseControlsEnabled(on) {
  ['poseRx', 'poseRy', 'poseRz', 'poseTx', 'poseTy', 'poseTz'].forEach((id) => $(id).disabled = !on);
  $('poseResetBtn').disabled = !on;
}

function resetPoseControls() {
  state.poseBase = null;
  state.poseDeltas = { rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0 };
  ['poseRx', 'poseRy', 'poseRz', 'poseTx', 'poseTy', 'poseTz'].forEach((id) => {
    $(id).value = 0;
    $(id).disabled = true;
  });
  $('poseResetBtn').disabled = true;
  syncPoseSliderLabels();
  state.preview = null;
  renderPreviewStats();
}

function syncPoseSliderLabels() {
  const d = state.poseDeltas;
  $('poseRxVal').textContent = `${d.rx}°`;
  $('poseRyVal').textContent = `${d.ry}°`;
  $('poseRzVal').textContent = `${d.rz}°`;
  $('poseTxVal').textContent = `${d.tx} mm`;
  $('poseTyVal').textContent = `${d.ty} mm`;
  $('poseTzVal').textContent = `${d.tz} mm`;
}

function onPoseSliderInput() {
  state.poseDeltas = {
    rx: Number($('poseRx').value),
    ry: Number($('poseRy').value),
    rz: Number($('poseRz').value),
    tx: Number($('poseTx').value),
    ty: Number($('poseTy').value),
    tz: Number($('poseTz').value),
  };
  syncPoseSliderLabels();
  // 拖动即时反映在视口；诊断请求做去抖，避免每帧都打后端
  const pose = effectivePose();
  const s = state.session.schemes.find((x) => x.id === state.selectedScheme);
  if (s && pose) renderOverlap(s, pose);
  requestPreview();
}

function resetPoseToScheme() {
  state.poseDeltas = { rx: 0, ry: 0, rz: 0, tx: 0, ty: 0, tz: 0 };
  ['poseRx', 'poseRy', 'poseRz', 'poseTx', 'poseTy', 'poseTz'].forEach((id) => { $(id).value = 0; });
  syncPoseSliderLabels();
  requestPreview();
}

function requestPreview() {
  clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(runPreview, 140);
}

async function runPreview() {
  if (!state.session || !state.poseBase) return;
  const pose = effectivePose();
  if (!pose) return;
  const body = { pose };
  const it = Number($('inlierThreshold').value);
  const od = Number($('overlapDistance').value);
  if (Number.isFinite(it) && it > 0) body.inlierThreshold = it;
  if (Number.isFinite(od) && od > 0) body.overlapDistance = od;
  state.previewLoading = true;
  try {
    state.preview = await api(`/api/sessions/${state.session.sessionId}/diagnose`, 'POST', body);
    state.previewError = null;
  } catch (e) {
    // 诊断失败必须回到配准面板，不得用默认值伪装成功
    state.preview = null;
    state.previewError = e.message;
    toast(`诊断预览失败：${e.message}`);
  } finally {
    state.previewLoading = false;
  }
  renderPreviewStats();
  const s = state.session.schemes.find((x) => x.id === state.selectedScheme);
  if (s && state.preview) renderOverlap(s, pose);
}

function renderPreviewStats() {
  const p = state.preview;
  const strip = $('previewStrip');
  strip.classList.toggle('hidden', false);
  if (state.previewError) {
    $('previewOverlap').textContent = '重合 —';
    $('previewResidual').textContent = '距离残差 —';
    $('previewCorr').textContent = '有效对应 —';
    $('previewCause').textContent = `诊断失败：${state.previewError}`;
    $('previewCause').className = 'pill bad';
    $('poseStatus').textContent = '诊断失败：未返回结果（不以默认值代替）';
    return;
  }
  if (!p) {
    $('previewOverlap').textContent = '重合 —';
    $('previewResidual').textContent = '距离残差 —';
    $('previewCorr').textContent = '有效对应 —';
    $('previewCause').textContent = '根因 —';
    ['previewOverlap', 'previewResidual', 'previewCorr', 'previewCause']
      .forEach((id) => { $(id).className = 'pill'; });
    $('poseStatus').textContent = '拖动滑块以试探配准，实时显示重合与残差';
    return;
  }
  const ov = p.coverage.overlapFraction;
  const matched = p.validCorrespondence.matchedCount;
  const total = matched + p.validCorrespondence.unmatchedCount;
  const res = p.validCorrespondence.residual;
  $('previewOverlap').textContent = `重合 ${(ov * 100).toFixed(1)}% · 目标覆盖 ${(p.coverage.targetCoverageFraction * 100).toFixed(1)}%`;
  $('previewOverlap').className = `pill ${ov >= 0.5 ? 'good' : ov > 0.2 ? 'warn' : 'bad'}`;
  $('previewResidual').textContent = res
    ? `距离残差 中位 ${res.median.toFixed(3)} / 均值 ${res.mean.toFixed(3)} mm`
    : '距离残差 —（无匹配点）';
  $('previewResidual').className = 'pill';
  $('previewCorr').textContent = `有效对应 ${matched}/${total}`;
  $('previewCorr').className = `pill ${p.validCorrespondence.satisfiesMinimum ? 'good' : 'bad'}`;
  const cause = p.cause;
  $('previewCause').textContent = `根因 ${cause.primary}` +
    (cause.secondary.length ? ` · ${cause.secondary.join(' / ')}` : '');
  $('previewCause').className = `pill ${cause.primary === 'NONE' ? 'good' : 'warn'}`;
  const outliers = p.correspondencePoints.outlierCount;
  $('poseStatus').textContent =
    `逐对原因：${p.correspondencePoints.inlierCount} 对在内点阈值内，${outliers} 对超出` +
    `（阈值 ${p.correspondencePoints.threshold.toFixed(3)} mm）`;
}

// ---------- ICP 证据面板 ----------
function renderEvidence(s) {
  const el = $('evidencePanel');
  if (!s || !s.evidence) {
    el.innerHTML = '<span class="muted">该方案尚无迭代证据；执行 ICP 局部优化后生成。</span>';
    return;
  }
  if (s.evidence.error) {
    el.innerHTML = `<div class="finding error">证据生成失败：${escapeHtml(s.evidence.error)}</div>`;
    return;
  }
  const ev = s.evidence;
  const cause = ev.cause || {};
  const stop = ev.stop || {};
  const budget = ev.roundBudget || {};
  const rows = [
    ['证据指纹', `<code>${escapeHtml(ev.fingerprint || '—')}</code>`],
    ['停止条件', `${stop.code || '—'}${stop.converged ? '（收敛）' : ''}`],
    ['迭代轮数', `${budget.totalRounds ?? '—'} 轮；保留 ${budget.retainedRounds ?? '—'}，省略 ${budget.omittedRounds ?? '—'}`],
    ['初始 → 最终重叠',
      `${(ev.coverage.initialOverlapFraction * 100).toFixed(1)}% → ${(ev.coverage.finalOverlapFraction * 100).toFixed(1)}%`],
    ['采样噪声地板', ev.noiseFloor && Number.isFinite(ev.noiseFloor.median)
      ? `${ev.noiseFloor.median.toFixed(3)} mm` : '—'],
    ['尺度距离比', ev.scaleRatios && ev.scaleRatios.available
      ? `${ev.scaleRatios.median.toFixed(3)}（IQR ${ev.scaleRatios.iqr.toFixed(3)}）` : '不可用'],
  ];
  let html = rows.map(([k, v]) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`).join('');
  html += `<h3>根因</h3><div class="cause-primary">${escapeHtml(cause.primary || '—')}` +
    (cause.secondary && cause.secondary.length ? ` + ${escapeHtml(cause.secondary.join(', '))}` : '') + '</div>';
  if (cause.evidence && cause.evidence.length) {
    html += cause.evidence.map((e) =>
      `<div class="finding ${e.code === cause.primary ? 'warning' : 'info'}">[${e.stage}/${e.code}] ${escapeHtml(e.message)}</div>`).join('');
  }
  html += '<h3>残差下降</h3>';
  html += sparkline((ev.residualTrend || []).map((t) => t.mean), '#5b9dff');
  html += (ev.residualTrend || []).map((t) =>
    `<div class="trend-row"><span>第 ${t.iteration} 轮</span><span>均值 ${fmtNum(t.mean)}</span>` +
    `<span>${t.drop == null ? '—' : (t.drop >= 0 ? `下降 ${fmtNum(t.drop)}` : `上升 ${fmtNum(-t.drop)}`)}</span></div>`).join('');
  html += '<h3>有效对应</h3>';
  html += (ev.validCorrespondenceTrend || []).map((t) =>
    `<div class="trend-row"><span>第 ${t.iteration} 轮</span><span>匹配 ${t.matched}</span>` +
    `<span>未匹配 ${t.unmatched}</span><span class="${t.satisfiesMinimum ? 'ok-text' : 'danger-text'}">` +
    `${t.satisfiesMinimum ? '满足下限' : '低于下限'}</span></div>`).join('');
  html += '<h3>阶段定位</h3>';
  html += (ev.stageFindings || []).map((f) =>
    `<div class="finding ${f.severity}">[${f.stage}/${f.severity}] ${escapeHtml(f.code)}：${escapeHtml(f.message)}</div>`).join('')
    || '<div class="muted">无异常阶段</div>';
  if (s.replay) {
    html += '<h3>单轮重放</h3>';
    html += `<div class="${s.replay.stable ? 'replay-ok' : 'replay-bad'}">` +
      `${s.replay.stable ? '逐轮重放指纹一致' : '存在重放不一致轮次，证据不可信'}</div>`;
    html += (s.replay.rounds || []).map((r) =>
      `<div class="trend-row"><span>第 ${r.iteration} 轮</span><span><code>${escapeHtml(r.fingerprint || '')}</code></span>` +
      `<span class="${r.match ? 'ok-text' : 'danger-text'}">${r.match ? '一致' : '不一致'}</span></div>`).join('');
  }
  el.innerHTML = html;
}

function sparkline(values, color) {
  const vals = values.filter((v) => Number.isFinite(v));
  if (!vals.length) return '';
  const w = 100; const h = 100; // 用 viewBox 归一化，宽度自适应
  const min = Math.min(...vals); const max = Math.max(...vals);
  const span = max - min || 1;
  const pts = vals.map((v, i) => {
    const x = vals.length === 1 ? 50 : (i / (vals.length - 1)) * w;
    const y = h - ((v - min) / span) * (h - 10) - 5;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
  return `<svg class="spark" viewBox="0 0 100 100" preserveAspectRatio="none">` +
    `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}" /></svg>`;
}

function fmtNum(x) {
  return Number.isFinite(x) ? x.toFixed(3) : '—';
}

function escapeHtml(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderPreviews(scheme) {
  renderPreviewStats();
  if (scheme) renderEvidence(scheme);
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
  // 拖动初始位姿：选项与基准位姿来自内核候选，滑块只做局部试探
  $('poseSchemeSelect').addEventListener('change', () => setPoseBaseFromScheme());
  ['poseRx', 'poseRy', 'poseRz', 'poseTx', 'poseTy', 'poseTz'].forEach((id) =>
    $(id).addEventListener('input', onPoseSliderInput));
  $('poseResetBtn').addEventListener('click', resetPoseToScheme);
  $('showMesh').addEventListener('change', () => { renderStaticViewers(); if (state.selectedScheme != null) selectScheme(state.selectedScheme); });
  $('showResidual').addEventListener('change', () => { if (state.selectedScheme != null) selectScheme(state.selectedScheme); });
  $('clipAxis').addEventListener('change', applyClip);
  $('clipPos').addEventListener('input', applyClip);
  // 默认载入一个完整场景，便于直接体验
  loadDataset().catch((e) => toast(e.message));
}

init();

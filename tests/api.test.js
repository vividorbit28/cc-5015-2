// API 端到端集成测试：模拟前端完整操作序列。
// 前置：node server/server.js 已在 PORT 监听。
// 运行：node tests/api.test.js

'use strict';

const BASE = process.env.BASE || 'http://localhost:8080';
let pass = 0;
let fail = 0;

async function call(path, method = 'GET', body) {
  const res = await fetch(BASE + path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}: ${data.error}`);
  return data;
}
function ok(name, cond, extra) {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.error(`  FAIL  ${name}${extra ? ' — ' + extra : ''}`); }
}

async function main() {
  console.log('== A. 数据集与几何接口 ==');
  const ds = await call('/api/datasets', 'POST', { scenario: 'wrongpairs', seed: 31 });
  ok('创建会话返回 sessionId', typeof ds.sessionId === 'string' && ds.sessionId.length >= 8);
  ok('会话携带单位/左右手/上轴约定', ds.meta.unit === 'mm' && ds.meta.handedness === 'right' && ds.meta.upAxis === 'Y');
  ok('预置 8 对对应点', ds.correspondences.length === 8);
  const sid = ds.sessionId;
  const geo = await call(`/api/sessions/${sid}/geometry`);
  ok('几何接口含目标/源点云与网格',
    Array.isArray(geo.target.cloud.points) && Array.isArray(geo.source.mesh.faces));
  ok('点云带法向量（点到面 ICP 用）', geo.target.cloud.normals.length === geo.target.cloud.points.length);

  console.log('== B. 对应点权重与校验 ==');
  const updated = await call(`/api/sessions/${sid}/correspondences`, 'PUT', {
    correspondences: ds.correspondences.map((c, i) => ({
      ...c, targetWeight: i === 0 ? 0.3 : 1, sourceWeight: 1,
    })),
  });
  ok('权重更新生效', updated.correspondences[0].targetWeight === 0.3);
  let rejected = false;
  try {
    await call(`/api/sessions/${sid}/correspondences`, 'PUT', {
      correspondences: [{ targetPoint: [0, 0], sourcePoint: [0, 0, 0], targetWeight: 1, sourceWeight: 1 }],
    });
  } catch { rejected = true; }
  ok('非三维坐标被拒绝', rejected);
  let badWeight = false;
  try {
    await call(`/api/sessions/${sid}/correspondences`, 'PUT', {
      correspondences: [{ targetPoint: [0, 0, 0], sourcePoint: [0, 0, 0], targetWeight: -1, sourceWeight: 1 }],
    });
  } catch { badWeight = true; }
  ok('非正权重被拒绝', badWeight);

  console.log('== C. 配准多候选与离群审计 ==');
  const reg = await call(`/api/sessions/${sid}/register`, 'POST', {});
  ok('返回基线+RANSAC+对称候选', reg.schemes.length >= 3);
  const best = reg.schemes.find((s) => s.source === 'ransac');
  ok('RANSAC 候选存在', !!best);
  ok('最佳候选 100% 重叠', best.overlapFraction > 0.99, `实际 ${best.overlapFraction}`);
  ok('污染对应点被排除并留理由', best.outlierReport.some((r) => r.excluded && /超过内点阈值/.test(r.reason)));
  ok('所有方案变换 det=+1（无反射/缩放）', reg.schemes.every((s) => {
    const R = s.transform.rotation;
    const mDet = (m) => m[0] * (m[4] * m[8] - m[5] * m[7]) -
      m[1] * (m[3] * m[8] - m[5] * m[6]) +
      m[2] * (m[3] * m[7] - m[4] * m[6]);
    return Math.abs(mDet(R) - 1) < 1e-8;
  }));
  ok('点残差数组与源点云等长', best.pointResiduals.length === geo.source.cloud.points.length);
  ok('退化诊断字段齐全', best.diagnostics && best.diagnostics.degeneracy && 'rank' in best.diagnostics.degeneracy);

  console.log('== D. ICP 局部优化不覆盖原方案 ==');
  const beforeCount = reg.schemes.length;
  const icpResp = await call(`/api/sessions/${sid}/schemes/${best.id}/icp`, 'POST', {});
  const icpScheme = icpResp.schemes.find((s) => s.source === 'icp');
  ok('ICP 产生独立新方案', icpResp.schemes.length === beforeCount + 1 && !!icpScheme);
  ok('ICP 记录收敛/迭代/排除信息', typeof icpScheme.icp.converged === 'boolean' &&
    Array.isArray(icpScheme.icp.history) && 'rejectedCount' in icpScheme.icp);
  ok('原 RANSAC 方案仍保留', icpResp.schemes.some((s) => s.id === best.id && s.source === 'ransac'));

  console.log('== D2. 迭代证据与逐轮重放 ==');
  ok('ICP 方案携带证据摘要与指纹', icpScheme.evidence && icpScheme.evidence.fingerprint &&
    typeof icpScheme.evidence.fingerprint === 'string');
  ok('证据含残差下降与有效对应趋势',
    Array.isArray(icpScheme.icp.residualTrend) && Array.isArray(icpScheme.icp.validCorrespondenceTrend));
  ok('证据含阶段定位与根因', Array.isArray(icpScheme.icp.stageFindings) && icpScheme.icp.cause &&
    typeof icpScheme.icp.cause.primary === 'string');
  ok('逐轮重放全部一致', icpScheme.replay && icpScheme.replay.stable === true);
  const evidenceResp = await call(`/api/sessions/${sid}/schemes/${icpScheme.id}/evidence`);
  ok('证据接口返回完整证据', evidenceResp.evidence.fingerprint === icpScheme.evidence.fingerprint);
  ok('证据接口复算重放稳定', evidenceResp.replayStable === true &&
    evidenceResp.replay.every(Boolean));
  ok('证据不含测试真值字段', !('groundTruth' in evidenceResp.evidence));

  console.log('== D3. 拖拽位姿诊断预览 ==');
  const geo2 = await call(`/api/sessions/${sid}/geometry`);
  const identityPose = {
    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    translation: [0, 0, 0],
  };
  const preview = await call(`/api/sessions/${sid}/diagnose`, 'POST', { pose: identityPose });
  ok('预览返回显式参数来源（不隐藏默认值）',
    preview.params && Number.isFinite(preview.params.inlierThreshold) &&
    Number.isFinite(preview.params.maxDistance));
  ok('预览回报参数推导来源', preview.paramsDerivedFrom &&
    typeof preview.paramsDerivedFrom.inlierThreshold === 'string' &&
    preview.paramsDerivedFrom.sourceTargetDiagonal.target > 0);
  ok('预览返回重叠、有效对应与逐对原因',
    Number.isFinite(preview.coverage.overlapFraction) &&
    Number.isFinite(preview.validCorrespondence.matchedCount) &&
    preview.correspondencePoints.items.length === ds.correspondences.length &&
    preview.correspondencePoints.items.every((it) => typeof it.reason === 'string'));
  ok('预览给出根因（错误初值下非 NONE）', preview.cause.primary !== 'NONE');
  ok('预览不修改会话（方案数不变）',
    (await call(`/api/sessions/${sid}`)).schemes.length === icpResp.schemes.length);
  let badPose = false;
  try {
    await call(`/api/sessions/${sid}/diagnose`, 'POST', { pose: { rotation: [1, 0, 0], translation: [0, 0, 0] } });
  } catch { badPose = true; }
  ok('非 3x3 位姿被拒绝', badPose);
  let badThreshold = false;
  try {
    await call(`/api/sessions/${sid}/diagnose`, 'POST', { pose: identityPose, inlierThreshold: -1 });
  } catch { badThreshold = true; }
  ok('非法阈值显式报错，不落回默认值', badThreshold);

  console.log('== D4. 诊断失败不遮错 ==');
  let previewNoSession = false;
  try { await call('/api/sessions/deadbeef/diagnose', 'POST', { pose: identityPose }); } catch { previewNoSession = true; }
  ok('失效会话诊断返回错误而非默认结果', previewNoSession);

  console.log('== E. 人工确认绑定输入与参数 ==');
  let noOperator = false;
  try { await call(`/api/sessions/${sid}/schemes/${icpScheme.id}/confirm`, 'POST', { decision: 'accepted' }); }
  catch { noOperator = true; }
  ok('缺少操作员的确认被拒绝', noOperator);
  const confirmed = await call(`/api/sessions/${sid}/schemes/${icpScheme.id}/confirm`, 'POST', {
    operator: '测试员甲', decision: 'undecided', note: '集成测试',
  });
  ok('确认记录含输入指纹与快照', confirmed.confirmation &&
    confirmed.confirmation.inputFingerprint.length === 16 &&
    confirmed.confirmation.correspondenceSnapshot.length === 8);
  ok('确认绑定变换快照', Array.isArray(confirmed.confirmation.transformSnapshot.rotation));

  console.log('== F. 导出可复算 ==');
  const expRes = await fetch(`${BASE}/api/sessions/${sid}/schemes/${icpScheme.id}/export`);
  ok('导出接口 200', expRes.ok);
  const exp = await expRes.json();
  ok('导出含约定声明', exp.conventions.unit === 'mm' && /无缩放/.test(exp.conventions.transformDirection));
  ok('导出同时含原始与变换后源点云',
    exp.originalSourceCloud.points.length === exp.transformedSourceCloud.points.length);
  ok('导出含真值（仅用于验证）与确认记录', exp.groundTruth && exp.confirmation);
  ok('导出升级为 v2 并携带证据与重放摘要',
    exp.exportVersion === 2 && exp.evidence && exp.evidence.fingerprint === icpScheme.evidence.fingerprint &&
    exp.replay && exp.replay.stable === true);
  ok('导出注明证据不含真值', /不含测试真值/.test(exp.conventions.evidenceScope || ''));

  console.log('== G. 镜像场景的警告链路 ==');
  const mir = await call('/api/datasets', 'POST', { scenario: 'mirrored', seed: 21 });
  const mirReg = await call(`/api/sessions/${mir.sessionId}/register`, 'POST', {});
  ok('镜像场景产生镜像警告', mirReg.schemes.some((s) => s.warnings.some((w) => /镜像|反射|左右手/.test(w))));
  const mirGeo = await call(`/api/sessions/${mir.sessionId}/geometry`);
  ok('镜像几何仍可正常返回', mirGeo.source.cloud.points.length > 0);

  console.log('== H. 错误处理 ==');
  let noSession = false;
  try { await call('/api/sessions/deadbeef/geometry'); } catch { noSession = true; }
  ok('失效会话返回 404', noSession);

  console.log(`\n${fail ? `失败 ${fail} 项` : `API 集成测试全部 ${pass} 项通过`}`);
  process.exit(fail ? 1 : 0);
}

main().catch((e) => { console.error('测试中断:', e); process.exit(1); });

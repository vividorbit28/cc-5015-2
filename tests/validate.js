// 数值与几何验证：
//  1) 已知变换：无噪声对应点应恢复真值（旋转角误差、平移误差、行列式）
//  2) 局部缺失：碎片部分缺失仍能配准，重叠率如实下降
//  3) 镜像候选：反射碎片被标记 det<0，不被当作合法刚性配准
//  4) 错误对应点：RANSAC 排除污染对，并留下逐点理由
//  5) 退化对应点：共线/共面配置给出秩不足诊断与多候选
//  运行：node tests/validate.js

'use strict';

const assert = require('assert');
const {
  weightedKabsch, applyTransform, mDet, mMul, mTranspose, vsub, vdist,
  rotationAxisAngle, symmetricEigen3, mApply,
} = require('../server/linalg');
const { ransacCandidates, icpRefine, cloudOverlapStats } = require('../server/registration');
const { generateDataset, axisAngleMatrix, mulberry32 } = require('../server/synthdata');

let pass = 0;
function check(name, fn) {
  try {
    fn();
    pass++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    console.error(`  FAIL  ${name}\n        ${e.message}`);
    process.exitCode = 1;
  }
}
const approx = (a, b, tol) => Math.abs(a - b) <= tol;

function rngSeed(s) { return mulberry32(s); }

console.log('== 1. 已知变换恢复 ==');
for (const scenario of ['bowl', 'plate', 'bigtwist']) {
  check(`${scenario}: Kabsch 恢复已知刚性变换（无缩放）`, () => {
    const ds = generateDataset(scenario, 42);
    const corr = ds.landmarks.filter((l) => !l.groundTruthCorrupt);
    assert.ok(corr.length >= 3);
    const fit = weightedKabsch(
      corr.map((c) => c.sourcePoint),
      corr.map((c) => c.targetPoint),
    );
    const gt = ds.meta.groundTruth;
    // source = R_gt p + t_gt，故 source->target 的正确变换为 R_gt^T 与 -R_gt^T t_gt
    const RgtInv = mTranspose(gt.rotation);
    const tgtInv = mApply(RgtInv, gt.translation).map((x) => -x);
    for (let i = 0; i < 9; i++)
      assert.ok(approx(fit.rotation[i], RgtInv[i], 1e-8),
        `旋转矩阵元素 ${i} 偏差 ${Math.abs(fit.rotation[i] - RgtInv[i])}`);
    for (let k = 0; k < 3; k++)
      assert.ok(approx(fit.translation[k], tgtInv[k], 1e-6),
        `平移分量 ${k} 偏差`);
    assert.ok(approx(mDet(fit.rotation), 1, 1e-10), 'det(R) 必须为 +1');
    assert.ok(fit.rms < 1e-6, `RMS 应近零，实际 ${fit.rms}`);
    assert.ok(fit.degeneracy.rank === 3, '非共面标志点应满秩');
  });
}

check('加噪对应点：权重融合与 RMS 量级合理', () => {
  const ds = generateDataset('bowl', 7);
  const rng = rngSeed(99);
  const noisy = ds.landmarks.map((l) => ({
    s: l.sourcePoint.map((x) => x + (rng() - 0.5) * 1.0),
    t: l.targetPoint.map((x) => x + (rng() - 0.5) * 1.0),
  }));
  const fit = weightedKabsch(noisy.map((x) => x.s), noisy.map((x) => x.t), {
    targetWeights: noisy.map((_, i) => (i === 0 ? 0.2 : 1)),
    sourceWeights: noisy.map(() => 1),
  });
  assert.ok(fit.rms < 1.2, `噪声 RMS 异常：${fit.rms}`);
  assert.ok(approx(mDet(fit.rotation), 1, 1e-10));
  const gt = ds.meta.groundTruth;
  const RgtInv = mTranspose(gt.rotation);
  let angErr = 0;
  for (let i = 0; i < 9; i++) angErr = Math.max(angErr, Math.abs(fit.rotation[i] - RgtInv[i]));
  assert.ok(angErr < 0.02, `加噪旋转偏差过大：${angErr}`);
});

console.log('== 2. 局部缺失 ==');
check('partial: 缺失区域下仍配准，重叠率如实反映且不引入缩放', () => {
  const ds = generateDataset('partial', 11);
  const corr = ds.landmarks;
  const fit = weightedKabsch(corr.map((c) => c.sourcePoint), corr.map((c) => c.targetPoint));
  const moved = applyTransform(ds.source.cloud.points, fit);
  const diag = Math.hypot(160, 120, 110);
  const stats = cloudOverlapStats(moved, ds.target.cloud.points, diag * 0.02);
  assert.ok(stats.cloudRms < 3, `点云 RMS 应在采样噪声级，实际 ${stats.cloudRms}`);
  assert.ok(stats.overlapFraction < 0.999, '缺失碎片重叠率不应为 1');
  assert.ok(stats.overlapFraction > 0.6, `重叠率过低：${stats.overlapFraction}`);
  // 无缩放检查：任意两点距离在变换前后不变
  const rng = rngSeed(3);
  for (let k = 0; k < 20; k++) {
    const i = Math.floor(rng() * moved.length);
    const j = Math.floor(rng() * moved.length);
    const d0 = vdist(ds.source.cloud.points[i], ds.source.cloud.points[j]);
    const d1 = vdist(moved[i], moved[j]);
    assert.ok(approx(d0, d1, 1e-9), '刚性变换改变了点间距离（疑似缩放）');
  }
});

check('ICP 从受扰初值收敛回真值附近', () => {
  const ds = generateDataset('bowl', 5);
  const gt = ds.meta.groundTruth;
  const RgtInv = mTranspose(gt.rotation);
  const tgtInv = mApply(RgtInv, gt.translation).map((x) => -x);
  const perturbR = axisAngleMatrix([0, 0, 1], 0.06);
  // 在正确的 source->target 逆变换附近施加扰动
  const init = {
    rotation: mMul(perturbR, RgtInv),
    translation: tgtInv.map((x) => x + 2.5),
  };
  const diag = Math.hypot(180, 140, 180);
  const out = icpRefine(ds.source.cloud, ds.target.cloud, init, {
    maxIterations: 60, maxDistance: diag * 0.04, huberK: diag * 0.02,
  });
  assert.ok(out.converged, `ICP 应收敛：${out.reason}`);
  let err = 0;
  for (let i = 0; i < 9; i++)
    err = Math.max(err, Math.abs(out.transform.rotation[i] - RgtInv[i]));
  // 两块点云独立面采样，噪声地板约 2mm，旋转误差容差与之相称
  assert.ok(err < 0.03, `ICP 旋转误差过大：${err}`);
  const terr = Math.hypot(...vsub(out.transform.translation, tgtInv));
  assert.ok(terr < 3.0, `ICP 平移误差过大：${terr}`);
  // 收敛后平均点距应显著小于初始残差
  const finalMean = out.history.at(-1).meanDistance;
  assert.ok(finalMean < 2.5, `ICP 末轮平均点距异常：${finalMean}`);
});

check('ICP 精确对应（同源点集）：收敛到误差近零的刚性解', () => {
  const rng = rngSeed(123);
  const base = Array.from({ length: 200 }, () => [
    (rng() - 0.5) * 100, (rng() - 0.5) * 100, (rng() - 0.5) * 100,
  ]);
  const R = axisAngleMatrix([0.2, 0.8, 0.5], 0.35);
  const t = [12, -7, 31];
  const moved = applyTransform(base, { rotation: R, translation: t });
  // source=moved -> target=base，真值为逆变换；在其附近施加扰动
  const Rinv = mTranspose(R);
  const tinv = mApply(Rinv, t).map((x) => -x);
  const perturbR = axisAngleMatrix([0, 0, 1], 0.05);
  const init = {
    rotation: mMul(perturbR, Rinv),
    translation: tinv.map((x) => x + 1.5),
  };
  const out = icpRefine(
    { points: moved, normals: moved.map(() => null) },
    { points: base, normals: base.map(() => null) },
    init,
    { maxIterations: 80, maxDistance: 200, huberK: null },
  );
  assert.ok(out.converged, `精确对应 ICP 应收敛：${out.reason}`);
  let err = 0;
  for (let i = 0; i < 9; i++) err = Math.max(err, Math.abs(out.transform.rotation[i] - Rinv[i]));
  assert.ok(err < 1e-6, `精确对应旋转误差应近零：${err}`);
  const terr = Math.hypot(...vsub(out.transform.translation, tinv));
  assert.ok(terr < 1e-5, `精确对应平移误差应近零：${terr}`);
});

console.log('== 3. 镜像候选 ==');
check('mirrored: 反射拟合被识别为 det=-1，刚性拟合残差如实变差', () => {
  const ds = generateDataset('mirrored', 21);
  assert.strictEqual(ds.meta.groundTruth.mirrored, true);
  assert.ok(mDet(ds.meta.groundTruth.rotation) < 0, '真值变换本身 det=-1');
  const fit = weightedKabsch(
    ds.landmarks.map((c) => c.sourcePoint),
    ds.landmarks.map((c) => c.targetPoint),
  );
  assert.ok(mDet(fit.rotation) > 0, '默认 Kabsch 输出必须 det=+1');
  assert.ok(mDet(fit.reflectionFit.rotation) < 0, '反射拟合 det 应为负');
  assert.ok(fit.mirrorSuspected === true, '镜像数据应触发 mirrorSuspected 标记');
  assert.ok(fit.reflectionFit.rms < 1e-5,
    '镜像在反射模型下残差很小——正说明不能仅凭残差小就断言同源');
  assert.ok(fit.rms > 1.0,
    `镜像在刚性模型下残差必须暴露问题，实际 ${fit.rms}`);
});

check('对称备选生成：180° 翻转候选与原候选并列保留', () => {
  const ds = generateDataset('bowl', 3);
  const fit = weightedKabsch(
    ds.landmarks.map((c) => c.sourcePoint),
    ds.landmarks.map((c) => c.targetPoint),
  );
  const { symmetryAlternatives } = require('../server/registration');
  const diag = 220;
  const alts = symmetryAlternatives(
    { transform: { rotation: fit.rotation, translation: fit.translation } },
    ds.source.cloud, ds.target.cloud,
    {
      overlapDistance: diag * 0.02,
      symmetryAxes: [{ axis: [0, 1, 0], label: '测试轴' }],
    },
  );
  assert.strictEqual(alts.length, 1);
  assert.ok(approx(mDet(alts[0].transform.rotation), 1, 1e-10));
  assert.ok(alts[0].overlapFraction >= 0);
});

console.log('== 4. 错误对应点 ==');
check('wrongpairs: RANSAC 排除污染对，正确方案内点不含损坏对', () => {
  const ds = generateDataset('wrongpairs', 31);
  const corr = ds.landmarks;
  const corrupt = new Set(corr.map((c, i) => (c.groundTruthCorrupt ? i : -1)).filter((i) => i >= 0));
  assert.ok(corrupt.size === 2);
  const cands = ransacCandidates(
    corr.map((c) => c.sourcePoint),
    corr.map((c) => c.targetPoint),
    { iterations: 500, inlierThreshold: 6.0, rng: rngSeed(55) },
  );
  assert.ok(cands.length >= 1, '应至少产生一个 RANSAC 候选');
  const best = cands[0];
  for (const bad of corrupt)
    assert.ok(!best.inlierIndices.includes(bad), `损坏对应点 ${bad} 不应进入内点集`);
  assert.ok(best.rms < 1e-5, `干净内点 RMS 应近零：${best.rms}`);
  const gt = ds.meta.groundTruth;
  const RgtInv = mTranspose(gt.rotation);
  let err = 0;
  for (let i = 0; i < 9; i++)
    err = Math.max(err, Math.abs(best.transform.rotation[i] - RgtInv[i]));
  assert.ok(err < 1e-6, '排除污染对后应恢复真值旋转');
});

check('全点 Kabsch 在错误对应点下显著变差（证明不能只看单一优化）', () => {
  const ds = generateDataset('wrongpairs', 31);
  const fitAll = weightedKabsch(
    ds.landmarks.map((c) => c.sourcePoint),
    ds.landmarks.map((c) => c.targetPoint),
  );
  const clean = ds.landmarks.filter((c) => !c.groundTruthCorrupt);
  const fitClean = weightedKabsch(
    clean.map((c) => c.sourcePoint),
    clean.map((c) => c.targetPoint),
  );
  assert.ok(fitAll.rms > fitClean.rms * 5,
    `污染应明显抬高全点 RMS（all=${fitAll.rms}, clean=${fitClean.rms}）`);
});

console.log('== 5. 退化对应点 ==');
check('共线对应点：诊断秩为 1，Kabsch 拒绝或标记欠定', () => {
  const rng = rngSeed(8);
  const sPts = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
  const R = axisAngleMatrix([0, 1, 0], 0.5);
  const t = [4, 5, 6];
  const tPts = sPts.map((p) => applyTransform([p], { rotation: R, translation: t })[0]);
  const fit = weightedKabsch(sPts, tPts);
  assert.strictEqual(fit.degeneracy.rank, 1);
  assert.ok(fit.polar.conditionNumber > 1e6 || fit.polar.conditionNumber === null,
    '共线配置条件数应病态');
  void rng;
});

check('共面对应点：planarRatio 接近 0，保留歧义警告依据', () => {
  const sPts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]];
  const fit = weightedKabsch(sPts, sPts.map((p) => [p[0], p[1], p[2] + 0.0001]));
  assert.ok(fit.degeneracy.planarRatio < 1e-3,
    `近共面 planarRatio=${fit.degeneracy.planarRatio}`);
});

check('Jacobi 特征分解：对角阵返回正确特征值', () => {
  const eig = symmetricEigen3([2, 0, 0, 0, 5, 0, 0, 0, 1]);
  assert.deepStrictEqual(eig.values.map((v) => Math.round(v)), [1, 2, 5]);
});

console.log(`\n${process.exitCode ? '存在失败项' : `全部 ${pass} 项检查通过`}`);

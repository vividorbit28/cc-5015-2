// 配准引擎：
//  1) 对用户对应点做加权 Kabsch + RANSAC，产生多个候选（含退化/对称备选）
//  2) 离群点按残差阈值排除并记录理由
//  3) ICP 点到点/点到面局部优化（仅旋转平移，不缩放）
//  4) 候选不确定性与重叠率评估
// 所有函数均不修改输入点云（原始模型不被变换结果覆盖）。

'use strict';

const {
  vsub, vadd, vcross, vnormalize, vdot,
  mApply, mMul, mTranspose, weightedKabsch, applyTransform,
} = require('./linalg');
const { KDTree } = require('./kdtree');
const diag = require('./diagnostics');

function diagError(code, message) {
  const e = new Error(message);
  e.code = code;
  return e;
}

function median(values) {
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

// 在索引集合中随机取 k 个不重复索引
function sampleIndices(n, k, rng) {
  const pool = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (n - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, k);
}

// 三点是否退化：面积过小 / 近乎共线
function tripletQuality(pts) {
  const e1 = vsub(pts[1], pts[0]);
  const e2 = vsub(pts[2], pts[0]);
  const area = 0.5 * Math.hypot(...vcross(e1, e2));
  const scale = Math.max(Math.hypot(...e1), Math.hypot(...e2), 1e-12);
  return { area, aspect: area / (scale * scale), degenerate: area / (scale * scale) < 1e-6 };
}

// 对一组 inlier 对应点拟合，返回候选对象（含诊断）
function fitCandidate(sourcePts, targetPts, inlierIdx, weights) {
  const s = inlierIdx.map((i) => sourcePts[i]);
  const t = inlierIdx.map((i) => targetPts[i]);
  const w = inlierIdx.map((i) => weights[i]);
  const fit = weightedKabsch(s, t, {
    targetWeights: w,
    sourceWeights: w,
  });
  return fit;
}

// 计算每对对应点在给定变换下的距离残差
function pairResiduals(sourcePts, targetPts, tf) {
  const moved = applyTransform(sourcePts, tf);
  return targetPts.map((tp, i) => {
    const d = vsub(moved[i], tp);
    return Math.hypot(d[0], d[1], d[2]);
  });
}

// RANSAC 主流程，输出按内点数/RMS 排序的候选列表
function ransacCandidates(sourcePts, targetPts, options = {}) {
  const n = sourcePts.length;
  const rng = options.rng || Math.random;
  const iterations = options.iterations || 300;
  const threshold = options.inlierThreshold;
  diag.requireFiniteNumber('inlierThreshold', threshold);
  const weights = options.weights || new Array(n).fill(1);
  const seen = new Map();
  const candidates = [];

  for (let it = 0; it < iterations; it++) {
    const idx3 = sampleIndices(n, 3, rng);
    const qS = tripletQuality(idx3.map((i) => sourcePts[i]));
    const qT = tripletQuality(idx3.map((i) => targetPts[i]));
    if (qS.degenerate || qT.degenerate) continue;

    let fit;
    try {
      fit = fitCandidate(sourcePts, targetPts, idx3, weights);
    } catch {
      continue;
    }
    const resid = pairResiduals(sourcePts, targetPts, fit);
    const inliers = [];
    const outliers = [];
    resid.forEach((d, i) => {
      if (d <= threshold) inliers.push(i);
      else outliers.push(i);
    });
    if (inliers.length < 3) continue;

    // 用全部内点重拟合；拒绝近奇异/非正交解，防止退化三元组产生伪候选
    const refined = fitCandidate(sourcePts, targetPts, inliers, weights);
    if (Math.abs(refined.polar.determinant - 1) > 1e-4 ||
        refined.polar.polarResidual > 1e-6 ||
        (refined.polar.conditionNumber != null && refined.polar.conditionNumber > 1e8)) {
      continue;
    }
    // 精配后变换已变化，按最终变换重算内点/外点，避免残留初拟合的陈旧划分
    const finalResidAll = pairResiduals(sourcePts, targetPts, refined);
    const finalInliers = [];
    const finalOutliers = [];
    finalResidAll.forEach((d, i) => {
      if (d <= threshold) finalInliers.push(i);
      else finalOutliers.push(i);
    });
    if (finalInliers.length < 3) continue;
    const key = refined.rotation.join(',').split(',').map((x) => Number(x).toFixed(5)).join(',')
      + '|' + refined.translation.map((x) => x.toFixed(5)).join(',');
    if (seen.has(key)) {
      seen.get(key).support = Math.max(seen.get(key).support, finalInliers.length);
      continue;
    }
    const inlierResid = finalInliers.map((i) => finalResidAll[i]);
    const cand = {
      id: candidates.length,
      transform: { rotation: refined.rotation, translation: refined.translation },
      inlierIndices: finalInliers,
      outlierIndices: finalOutliers,
      inlierCount: finalInliers.length,
      coverage: finalInliers.length / n,
      rms: Math.sqrt(inlierResid.reduce((a, d) => a + d * d, 0) / inlierResid.length),
      maxInlierResidual: Math.max(...inlierResid),
      medianResidual: median(finalResidAll),
      diagnostics: {
        conditionNumber: refined.polar.conditionNumber,
        determinant: refined.polar.determinant,
        degeneracy: refined.degeneracy,
        rotationAngleDeg: refined.rotationAngleDeg,
        rotationAxis: refined.rotationAxis,
        tripletAspect: Math.min(qS.aspect, qT.aspect),
      },
      source: 'ransac',
    };
    seen.set(key, cand);
    candidates.push(cand);
  }

  candidates.sort((a, b) =>
    b.inlierCount - a.inlierCount || a.rms - b.rms);
  candidates.forEach((c, i) => { c.id = i; });
  return candidates;
}

// 为对称形状生成歧义候选：围绕给定向量旋转 180° 的备选
// （当点云在某轴附近近似轴对称时，Kabsch 无法区分两种朝向）
function symmetryAlternatives(baseCand, sourceCloud, targetCloud, options = {}) {
  const alternatives = [];
  const axisCandidates = options.symmetryAxes || [];
  for (const axisRaw of axisCandidates) {
    const axis = vnormalize(axisRaw.axis);
    if (!axis) continue;
    // 180° 旋转 R2 = 2 a a^T - I
    const R2 = [
      2 * axis[0] * axis[0] - 1, 2 * axis[0] * axis[1], 2 * axis[0] * axis[2],
      2 * axis[1] * axis[0], 2 * axis[1] * axis[1] - 1, 2 * axis[1] * axis[2],
      2 * axis[2] * axis[0], 2 * axis[2] * axis[1], 2 * axis[2] * axis[2] - 1,
    ];
    const newR = mMul(R2, baseCand.transform.rotation);
    const tf = { rotation: newR, translation: baseCand.transform.translation };
    const stats = cloudOverlapStats(
      applyTransform(sourceCloud.points, tf),
      targetCloud.points,
      options.overlapDistance,
    );
    alternatives.push({
      id: -1,
      transform: tf,
      symmetryAxis: axis,
      label: axisRaw.label || '180° 对称备选',
      ...stats,
      source: 'symmetry',
    });
  }
  return alternatives;
}

// 点云层面的重叠/残差统计（变换后的源 vs 目标 KD 树）。
// 搜索半径必须由调用方给出：缺失时直接报错，而不是悄悄使用默认值。
function cloudOverlapStats(transformedSource, targetPoints, maxDistance) {
  diag.requireFiniteNumber('maxDistance', maxDistance);
  const targetTree = new KDTree(targetPoints);
  const distances = transformedSource.map((p) => {
    const nn = targetTree.nearest(p, maxDistance);
    return nn ? nn.distance : null;
  });
  const matched = distances.filter((d) => d !== null);
  const rms = matched.length
    ? Math.sqrt(matched.reduce((a, d) => a + d * d, 0) / matched.length)
    : null;
  // 反向覆盖：目标碎片中能被变换后源点云覆盖的比例，用于区分低重叠与错误位姿
  const sourceTree = new KDTree(transformedSource);
  let covered = 0;
  for (const p of targetPoints) {
    if (sourceTree.nearest(p, maxDistance)) covered++;
  }
  return {
    overlapFraction: matched.length / transformedSource.length,
    targetCoverageFraction: covered / targetPoints.length,
    unmatchedSourceCount: distances.length - matched.length,
    cloudRms: rms,
    cloudMedian: matched.length ? median(matched) : null,
    cloudMax: matched.length ? Math.max(...matched) : null,
    pointResiduals: distances, // null 表示超出搜索半径
  };
}

// ICP：目标带法向量时为点到面，否则点到点；仅允许刚性更新。
// 外层重匹配最近邻，内层对固定对应做高斯-牛顿迭代，配合 Huber 降权。
// 每轮记录：输入位姿、有效对应数、残差摘要、位姿更新是否被接受，
// 以及可独立重放的单轮快照（输入位姿 + 搜索参数 + 指纹）。
function icpRefine(sourceCloud, targetCloud, initialTf, options = {}) {
  const maxIter = options.maxIterations || 40;
  const tolerance = options.tolerance || 1e-9;
  const maxDist = options.maxDistance;
  if (!Number.isFinite(maxDist) || maxDist <= 0) {
    throw diagError('DIAG_INPUT_MISSING', 'icpRefine 需要正的最大搜索距离 maxDistance');
  }
  const robustK = options.huberK == null ? null : options.huberK;
  const tree = new KDTree(targetCloud.points);
  let tf = {
    rotation: initialTf.rotation.slice(),
    translation: initialTf.translation.slice(),
  };
  const history = [];
  const rounds = [];
  let rejected = [];
  let lastPairs = [];
  let stopCode = null;
  let stopReason = '';

  // 记录诊断快照指纹（不含真值、不含时间戳）。与离线重放共用同一字段集合。
  const roundFingerprint = (inputPose, pairs, residual) => diag.roundRecordFingerprint({
    inputPose,
    maxDistance: maxDist,
    huberK: robustK,
    matchedCount: pairs.length,
    unmatchedCount: rejected.length,
    residual,
    pairs,
  });

  const pairStats = (pairs) => diag.summarize(pairs.map((p) => p.distance));

  const normalAt = (idx) => {
    const n = targetCloud.normals ? targetCloud.normals[idx] : null;
    return n && Number.isFinite(n[0]) ? n : null;
  };
  // 评估候选变换：重新查询最近邻，避免用陈旧对应误判收敛
  const meanPairDistance = (candidate) => {
    const mv = applyTransform(sourceCloud.points, candidate);
    let sum = 0; let cnt = 0;
    for (let i = 0; i < mv.length; i++) {
      const nn = tree.nearest(mv[i], maxDist);
      if (nn) { sum += nn.distance; cnt++; }
    }
    return cnt ? sum / cnt : Infinity;
  };

  // 采样噪声地板：点云自身最近邻间距，低于此值不再有可分辨改进
  const noiseFloor = options.noiseFloor != null
    ? { median: options.noiseFloor, available: true }
    : diag.samplingNoiseFloor(sourceCloud.points);
  const plateauFactor = options.plateauFactor ?? diag.PLATEAU_NOISE_FACTOR;
  const plateauLimit = noiseFloor.median > 0 ? noiseFloor.median * plateauFactor : 0;

  for (let iter = 0; iter < maxIter; iter++) {
    const moved = applyTransform(sourceCloud.points, tf);
    const pairs = [];
    rejected = [];
    for (let i = 0; i < moved.length; i++) {
      const nn = tree.nearest(moved[i], maxDist);
      if (!nn) {
        rejected.push({ sourceIndex: i, reasonCode: 'NO_NEIGHBOR_IN_RADIUS', reason: `半径 ${maxDist.toExponential(2)} 内无目标最近邻（碎片缺失或未重叠）` });
        continue;
      }
      let weight = 1;
      if (robustK !== null && nn.distance > robustK) weight = robustK / nn.distance;
      pairs.push({ sourceIndex: i, targetIndex: nn.index, distance: nn.distance, weight });
    }
    lastPairs = pairs;
    const inputPose = { rotation: tf.rotation.slice(), translation: tf.translation.slice() };
    const inputResidual = pairStats(pairs);

    if (pairs.length < diag.MIN_PAIRS) {
      stopCode = diag.STOP.INSUFFICIENT;
      stopReason = `有效 ICP 对应仅 ${pairs.length} 对，少于 ${diag.MIN_PAIRS}，停止优化`;
      rounds.push({
        iteration: iter,
        inputPose,
        matchedCount: pairs.length,
        unmatchedCount: rejected.length,
        residual: inputResidual,
        step: { accepted: false, meanBefore: inputResidual ? inputResidual.mean : null, meanAfter: null, stepNorm: null, reasonCode: 'INSUFFICIENT_CORRESPONDENCES' },
        inputFingerprint: roundFingerprint(inputPose, pairs, inputResidual),
      });
      history.push({ iteration: iter, pairCount: pairs.length, meanDistance: inputResidual ? inputResidual.mean : null, stepNorm: null });
      break;
    }

    // 高斯-牛顿内迭代：每步重新匹配最近邻，避免固定对应在大步长下失配
    let innerTf = tf;
    let innerMean = pairs.reduce((a, p) => a + p.distance * p.weight, 0) /
      pairs.reduce((a, p) => a + p.weight, 0);
    let lastStep = 0;
    let innerPairs = pairs;
    let poseStep = { accepted: false, meanBefore: innerMean, meanAfter: innerMean, stepNorm: 0, reasonCode: 'NO_INNER_IMPROVEMENT' };
    for (let gn = 0; gn < 15; gn++) {
      const mv = applyTransform(sourceCloud.points, innerTf);
      const Ai = [];
      const bi = [];
      const fresh = [];
      for (let i = 0; i < mv.length; i++) {
        const nn = tree.nearest(mv[i], maxDist);
        if (!nn) continue;
        let weight = 1;
        if (robustK !== null && nn.distance > robustK) weight = robustK / nn.distance;
        fresh.push({ sourceIndex: i, targetIndex: nn.index, distance: nn.distance, weight });
        const q = targetCloud.points[nn.index];
        const p = mv[i];
        const residualVec = vsub(q, p);
        const nrm = normalAt(nn.index);
        if (nrm) {
          // 点到面：残差 r = q - (R p + t)，左乘 Rodrigues 增量下旋转雅可比 n·[p]_×
          const cp = vcross(p, nrm);
          Ai.push([cp[0] * weight, cp[1] * weight, cp[2] * weight,
            nrm[0] * weight, nrm[1] * weight, nrm[2] * weight]);
          bi.push(vdot(nrm, residualVec) * weight);
        } else {
          // 点到点：旋转雅可比 [p]_×，平移雅可比 I
          const basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
          basis.forEach((e, k) => {
            const c = vcross(p, e);
            Ai.push([c[0] * weight, c[1] * weight, c[2] * weight,
              e[0] * weight, e[1] * weight, e[2] * weight]);
            bi.push(residualVec[k] * weight);
          });
        }
      }
      innerPairs = fresh;
      if (fresh.length < diag.MIN_PAIRS) break;
      const deltaI = solveLeastSquares6(Ai, bi);
      if (!deltaI || deltaI.some((x) => !Number.isFinite(x))) break;
      const [al, be, ga, tx2, ty2, tz2] = deltaI;
      // 精确 Rodrigues 增量旋转，避免小角度一阶近似的收敛地板
      const wv = [al, be, ga];
      const angle = Math.hypot(...wv);
      let Rd;
      if (angle < 1e-12) {
        Rd = [1, 0, 0, 0, 1, 0, 0, 0, 1];
      } else {
        const ax = wv.map((x) => x / angle);
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const C = 1 - c;
        const [x, y, z] = ax;
        Rd = [
          c + x * x * C, x * y * C - z * s, x * z * C + y * s,
          y * x * C + z * s, c + y * y * C, y * z * C - x * s,
          z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ];
      }
      const cand = {
        rotation: mMul(Rd, innerTf.rotation),
        translation: vadd(mApply(Rd, innerTf.translation), [tx2, ty2, tz2]),
      };
      const candMean = meanPairDistance(cand);
      poseStep = {
        accepted: candMean < innerMean,
        meanBefore: innerMean,
        meanAfter: candMean,
        stepNorm: Math.hypot(...deltaI),
        reasonCode: candMean < innerMean ? 'ACCEPTED_RESIDUAL_DROP' : 'REJECTED_NO_RESIDUAL_DROP',
      };
      if (!(candMean < innerMean)) break;
      innerTf = cand;
      innerMean = candMean;
      lastStep = Math.hypot(...deltaI);
      if (lastStep < tolerance) break;
    }

    tf = innerTf;
    const outputPairs = innerPairs;
    const outputResidual = pairStats(outputPairs);
    const round = {
      iteration: iter,
      inputPose,
      matchedCount: pairs.length,
      unmatchedCount: rejected.length,
      residual: inputResidual,
      outputResidual,
      step: poseStep,
      inputFingerprint: roundFingerprint(inputPose, pairs, inputResidual),
    };
    rounds.push(round);
    history.push({ iteration: iter, pairCount: innerPairs.length, meanDistance: innerMean, stepNorm: lastStep });

    if (!Number.isFinite(lastStep)) {
      stopCode = diag.STOP.NUMERIC;
      stopReason = '迭代步长出现非数值（法方程病态），ICP 停止';
      break;
    }
    // 位姿更新未降低平均点距：若残差已到采样噪声地板则为正常平台收敛，
    // 否则如实判定收敛停滞，不再把“第一轮没改善”误报为收敛。
    if (!poseStep.accepted) {
      if (innerMean <= plateauLimit) {
        stopCode = diag.STOP.PLATEAU;
        stopReason = `位姿更新不再降低残差，平台 ${innerMean.toFixed(4)} 处于采样噪声地板 ${noiseFloor.median.toFixed(4)} 以内`;
      } else {
        stopCode = diag.STOP.STAGNATED;
        stopReason = `位姿更新不再降低残差，平台 ${innerMean.toFixed(4)} 高于采样噪声地板 ${noiseFloor.median.toFixed(4)}`;
      }
      break;
    }
    if (lastStep < tolerance) {
      stopCode = diag.STOP.TOLERANCE;
      stopReason = '步长低于容差';
      break;
    }
  }
  if (!stopCode) {
    stopCode = diag.STOP.MAX_ITERATIONS;
    stopReason = `达到最大迭代次数 ${maxIter} 仍未收敛`;
  }

  const converged = stopCode === diag.STOP.TOLERANCE ||
    stopCode === diag.STOP.PLATEAU ||
    (stopCode === diag.STOP.MAX_ITERATIONS && history.length > 0);
  return {
    transform: tf,
    converged,
    stopCode,
    reason: stopReason,
    iterations: history.length,
    history,
    rounds,
    noiseFloor,
    plateauLimit,
    rejected,
    pairs: lastPairs,
  };
}

// 6 参数正规方程 (A^T A + λD) x = A^T b，高斯消元带主元；
// D 取 A^T A 对角元，λ 为 Levenberg-Marquardt 阻尼
function solveLeastSquares6(A, b, damping = 1e-12) {
  const N = 6;
  const ata = Array.from({ length: N }, () => new Array(N).fill(0));
  const atb = new Array(N).fill(0);
  for (let r = 0; r < A.length; r++) {
    for (let i = 0; i < N; i++) {
      atb[i] += A[r][i] * b[r];
      for (let j = 0; j < N; j++) ata[i][j] += A[r][i] * A[r][j];
    }
  }
  for (let i = 0; i < N; i++) ata[i][i] += damping * Math.max(ata[i][i], 1e-12);
  const scale = Math.max(...ata.map((row, i) => Math.abs(row[i])), 1e-12);
  const m = ata.map((row, i) => row.concat(atb[i]));
  for (let col = 0; col < N; col++) {
    let piv = col;
    for (let r = col + 1; r < N; r++)
      if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < scale * 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    for (let r = 0; r < N; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let k = col; k <= N; k++) m[r][k] -= f * m[col][k];
    }
  }
  const x = new Array(N);
  for (let i = 0; i < N; i++) x[i] = m[i][N] / m[i][i];
  return x.every(Number.isFinite) ? x : null;
}

// 离群点报告：为每个被排除的对应点给出可审计理由
function explainOutliers(sourcePts, targetPts, tf, inlierIdx, threshold) {
  const resid = pairResiduals(sourcePts, targetPts, tf);
  const inlierSet = new Set(inlierIdx);
  return resid.map((d, i) => ({
    pairIndex: i,
    residual: d,
    excluded: !inlierSet.has(i),
    reason: !inlierSet.has(i)
      ? `残差 ${d.toFixed(4)} 超过内点阈值 ${threshold.toFixed(4)}（模型单位）`
      : null,
  }));
}

// 综合评估候选：点云残差 + 退化/对称不确定性标签
function evaluateCandidate(cand, sourceCloud, targetCloud, overlapDistance) {
  const moved = applyTransform(sourceCloud.points, cand.transform);
  const stats = cloudOverlapStats(moved, targetCloud.points, overlapDistance);
  const warnings = [];
  const d = cand.diagnostics || {};
  if (d.degeneracy) {
    if (d.degeneracy.rank < 3)
      warnings.push(`对应点几何秩为 ${d.degeneracy.rank}：旋转分量欠定，结果存在绕轴歧义`);
    else if (d.degeneracy.planarRatio < 1e-3)
      warnings.push('对应点近共面，平面法向方向的平移与旋转不确定');
  }
  if (d.conditionNumber && d.conditionNumber > 1e6)
    warnings.push(`Kabsch 矩阵条件数 ${d.conditionNumber.toExponential(1)}，数值病态`);
  if (stats.overlapFraction < 0.5)
    warnings.push(`点云重叠率仅 ${(stats.overlapFraction * 100).toFixed(1)}%，重叠不足`);
  return { ...cand, ...stats, warnings };
}

module.exports = {
  ransacCandidates,
  symmetryAlternatives,
  icpRefine,
  cloudOverlapStats,
  pairResiduals,
  explainOutliers,
  evaluateCandidate,
  median,
};

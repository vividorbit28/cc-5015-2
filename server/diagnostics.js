// 配准诊断：只依赖点云、对应点、权重与内核迭代记录。
// 明确禁止读取 data.meta.groundTruth —— 生产求解与根因分类都不能借助测试真值。
// 所有阈值都以可观测统计（采样间距、对应距离比、内点率、重叠率）为依据。

'use strict';

const crypto = require('crypto');
const { vdist, weightedKabsch } = require('./linalg');
const { KDTree } = require('./kdtree');

const STAGE = Object.freeze({
  INPUT: 'input',
  CORRESPONDENCE: 'correspondence',
  WEIGHTING: 'weighting',
  POSE_UPDATE: 'pose-update',
  STOP: 'stop',
});

const CAUSE = Object.freeze({
  NONE: 'NONE',
  SCALE_UNIT_MISMATCH: 'SCALE_UNIT_MISMATCH',
  CORRESPONDENCE_ERROR: 'CORRESPONDENCE_ERROR',
  LOW_OVERLAP: 'LOW_OVERLAP',
  WRONG_INITIAL_POSE: 'WRONG_INITIAL_POSE',
  LOCAL_MINIMUM: 'LOCAL_MINIMUM',
  SYMMETRY_AMBIGUITY: 'SYMMETRY_AMBIGUITY',
  DEGENERATE_GEOMETRY: 'DEGENERATE_GEOMETRY',
  NUMERIC_FAILURE: 'NUMERIC_FAILURE',
});

const STOP = Object.freeze({
  TOLERANCE: 'CONVERGED_TOLERANCE',
  PLATEAU: 'CONVERGED_PLATEAU',
  STAGNATED: 'STAGNATED_NO_IMPROVEMENT',
  INSUFFICIENT: 'INSUFFICIENT_CORRESPONDENCES',
  MAX_ITERATIONS: 'MAX_ITERATIONS',
  NUMERIC: 'NUMERIC_FAILURE',
  CLOSED_FORM: 'CLOSED_FORM_SOLVED',
  RANSAC: 'RANSAC_COMPLETE',
});

const MIN_PAIRS = 6;
const PLATEAU_NOISE_FACTOR = 2.0;
const SCALE_TOLERANCE = 0.05;

function diagError(code, message) {
  const e = new Error(message);
  e.code = code;
  return e;
}

// 诊断拒绝缺省值冒充观测：必需输入缺失或非有限值时直接报错。
function requireFiniteNumber(name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw diagError('DIAG_INPUT_MISSING', `${name} 必须是有限数值，实际为 ${String(value)}`);
  }
  return value;
}

function requirePoints(name, points) {
  if (!Array.isArray(points) || points.length === 0) {
    throw diagError('DIAG_INPUT_MISSING', `${name} 必须是非空点数组`);
  }
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (!Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite)) {
      throw diagError('DIAG_INPUT_MISSING', `${name}[${i}] 不是有限三维点`);
    }
  }
  return points;
}

function median(values) {
  if (!values.length) return null;
  const s = values.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

function quantile(values, q) {
  if (!values.length) return null;
  const s = values.slice().sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))));
  return s[idx];
}

function summarize(values) {
  const finite = values.filter((v) => v != null && Number.isFinite(v));
  if (!finite.length) return null;
  const mean = finite.reduce((a, v) => a + v, 0) / finite.length;
  const squares = finite.reduce((a, v) => a + v * v, 0) / finite.length;
  return {
    count: finite.length,
    mean,
    rms: Math.sqrt(squares),
    median: median(finite),
    p90: quantile(finite, 0.9),
    max: Math.max(...finite),
    min: Math.min(...finite),
  };
}

// 采样点间距（self nearest-neighbour），作为“无法再低于”的噪声地板。
// 缓存到点数组对象上：同一会话的多个方案只计算一次。
const noiseFloorCache = new WeakMap();
function samplingNoiseFloor(points, sampleCount = 256) {
  requirePoints('points', points);
  if (points.length < 2) {
    return { median: 0, sampleCount: 0, stride: 1, available: false };
  }
  const cached = noiseFloorCache.get(points);
  if (cached && cached.sampleCount >= Math.min(sampleCount, points.length)) return cached;
  const n = points.length;
  const targetSamples = Math.min(sampleCount, n);
  const stride = Math.max(1, Math.floor(n / targetSamples));
  const ds = [];
  for (let i = 0; i < n; i += stride) {
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = vdist(points[i], points[j]);
      if (d < best) best = d;
    }
    if (Number.isFinite(best)) ds.push(best);
  }
  const result = {
    median: median(ds) ?? 0,
    sampleCount: ds.length,
    stride,
    available: ds.length > 0,
  };
  noiseFloorCache.set(points, result);
  return result;
}

function canonicalize(value) {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(canonicalize);
  const out = {};
  for (const key of Object.keys(value).sort()) out[key] = canonicalize(value[key]);
  return out;
}

function stableStringify(value) {
  return JSON.stringify(canonicalize(value));
}

function fingerprint(value, length = 16) {
  return crypto.createHash('sha256').update(stableStringify(value)).digest('hex').slice(0, length);
}

// 单轮记录的规范指纹：内核写入与离线重放必须使用同一字段集合，
// 否则“复跑一致”会因序列化差异产生假阴性。
function roundRecordFingerprint({ inputPose, maxDistance, huberK, matchedCount, unmatchedCount, residual, pairs }) {
  if (!inputPose) throw diagError('DIAG_INPUT_MISSING', 'roundRecordFingerprint 缺少 inputPose');
  requireFiniteNumber('maxDistance', maxDistance);
  if (huberK != null) requireFiniteNumber('huberK', huberK);
  return fingerprint({
    inputPose,
    maxDistance,
    huberK,
    matchedCount,
    unmatchedCount,
    residual,
    pairs,
  });
}

// 对应点两两距离比：稳定尺度偏差会表现为所有比值一致地偏离 1。
function correspondenceDistanceRatios(sourcePoints, targetPoints) {
  requirePoints('sourcePoints', sourcePoints);
  requirePoints('targetPoints', targetPoints);
  if (sourcePoints.length !== targetPoints.length) {
    throw diagError('DIAG_INPUT_MISSING', '对应点数量不一致');
  }
  const ratios = [];
  for (let i = 0; i < sourcePoints.length; i++) {
    for (let j = i + 1; j < sourcePoints.length; j++) {
      const ds = vdist(sourcePoints[i], sourcePoints[j]);
      const dt = vdist(targetPoints[i], targetPoints[j]);
      if (ds > 1e-9 && dt > 1e-9) ratios.push(dt / ds);
    }
  }
  if (!ratios.length) {
    return { available: false, samples: 0 };
  }
  const med = median(ratios);
  const q1 = quantile(ratios, 0.25);
  const q3 = quantile(ratios, 0.75);
  const iqr = q3 - q1;
  // 一致尺度：比值离散度小；若同时明显偏离 1，则不是噪声而是单位/尺度问题
  const uniform = iqr <= Math.max(0.02, Math.abs(med) * 0.1);
  return {
    available: true,
    samples: ratios.length,
    median: med,
    min: Math.min(...ratios),
    max: Math.max(...ratios),
    iqr,
    uniform,
    deviatesFromUnit: Math.abs(med - 1) > SCALE_TOLERANCE,
  };
}

// 点云双向覆盖：源→目标（重叠率）与目标→源（目标被覆盖比例）。
// 输入点数组必须由调用方提前变换；本函数不改写输入（KD 树按索引引用点）。
function coverageStats(transformedSource, targetPoints, maxDistance) {
  requirePoints('transformedSource', transformedSource);
  requirePoints('targetPoints', targetPoints);
  requireFiniteNumber('maxDistance', maxDistance);
  const targetTree = new KDTree(targetPoints);
  const distances = transformedSource.map((p) => {
    const nn = targetTree.nearest(p, maxDistance);
    return nn ? nn.distance : null;
  });
  const matched = distances.filter((d) => d != null);
  const sourceTree = new KDTree(transformedSource);
  let covered = 0;
  for (const p of targetPoints) {
    if (sourceTree.nearest(p, maxDistance)) covered++;
  }
  return {
    sourceCount: transformedSource.length,
    targetCount: targetPoints.length,
    overlapFraction: matched.length / transformedSource.length,
    targetCoverageFraction: covered / targetPoints.length,
    unmatchedSourceCount: distances.length - matched.length,
    cloudRms: matched.length ? Math.sqrt(matched.reduce((a, d) => a + d * d, 0) / matched.length) : null,
    cloudMedian: median(matched),
    cloudMax: matched.length ? Math.max(...matched) : null,
    pointResiduals: distances,
  };
}

// 最近的 ICP 有效对应与逐源点残差（预览与证据共用的观测）。
function nearestCorrespondenceStats(transformedSource, targetPoints, maxDistance, huberK) {
  requireFiniteNumber('maxDistance', maxDistance);
  const tree = new KDTree(targetPoints);
  const pairs = [];
  const rejected = [];
  for (let i = 0; i < transformedSource.length; i++) {
    const nn = tree.nearest(transformedSource[i], maxDistance);
    if (!nn) {
      rejected.push({ sourceIndex: i, reasonCode: 'NO_NEIGHBOR_IN_RADIUS' });
      continue;
    }
    const weight = huberK != null && nn.distance > huberK ? huberK / nn.distance : 1;
    pairs.push({ sourceIndex: i, targetIndex: nn.index, distance: nn.distance, weight });
  }
  const stats = summarize(pairs.map((p) => p.distance));
  const weightSum = pairs.reduce((a, p) => a + p.weight, 0);
  return {
    pairs,
    rejected,
    matchedCount: pairs.length,
    unmatchedCount: rejected.length,
    weightSum,
    residual: stats,
    minRequired: MIN_PAIRS,
    satisfiesMinimum: pairs.length >= MIN_PAIRS,
  };
}

// 用给定位姿生成对应点逐项报告（是否内点、原因）。
function correspondenceReport(sourcePoints, targetPoints, weights, transform, threshold) {
  requirePoints('sourcePoints', sourcePoints);
  requirePoints('targetPoints', targetPoints);
  requireFiniteNumber('threshold', threshold);
  if (sourcePoints.length !== targetPoints.length) {
    throw diagError('DIAG_INPUT_MISSING', '对应点数量不一致');
  }
  const w = weights || new Array(sourcePoints.length).fill(1);
  if (w.length !== sourcePoints.length) {
    throw diagError('DIAG_INPUT_MISSING', '权重数量与对应点不一致');
  }
  if (!transform || !Array.isArray(transform.rotation) || !Array.isArray(transform.translation)) {
    throw diagError('DIAG_INPUT_MISSING', 'correspondenceReport 需要完整位姿');
  }
  const R = transform.rotation;
  const t = transform.translation;
  const items = [];
  for (let i = 0; i < sourcePoints.length; i++) {
    const s = sourcePoints[i];
    const moved = [
      R[0] * s[0] + R[1] * s[1] + R[2] * s[2] + t[0],
      R[3] * s[0] + R[4] * s[1] + R[5] * s[2] + t[1],
      R[6] * s[0] + R[7] * s[1] + R[8] * s[2] + t[2],
    ];
    const residual = vdist(moved, targetPoints[i]);
    const inlier = residual <= threshold;
    items.push({
      pairIndex: i,
      residual,
      weight: w[i],
      inlier,
      reasonCode: inlier ? 'WITHIN_INLIER_THRESHOLD' : 'RESIDUAL_OVER_INLIER_THRESHOLD',
      reason: inlier
        ? `残差 ${residual.toFixed(4)} ≤ 内点阈值 ${threshold.toFixed(4)}（模型单位）`
        : `残差 ${residual.toFixed(4)} 超过内点阈值 ${threshold.toFixed(4)}（模型单位）`,
    });
  }
  const inliers = items.filter((x) => x.inlier);
  const residuals = items.map((x) => x.residual).sort((a, b) => a - b);
  // 双峰检查：把干净内点与错误对应分开的最大间隙
  let maxGap = 0;
  let gapAfter = -1;
  for (let i = 1; i < residuals.length; i++) {
    const gap = residuals[i] - residuals[i - 1];
    if (gap > maxGap) { maxGap = gap; gapAfter = i; }
  }
  // 与初始位姿无关的刚性一致性：该位姿能否让多数对应点同时贴合。
  const bestFit = {
    inlierCount: inliers.length,
    inlierRatio: items.length ? inliers.length / items.length : 0,
  };
  return {
    items,
    threshold,
    inlierCount: inliers.length,
    outlierCount: items.length - inliers.length,
    inlierRatio: bestFit.inlierRatio,
    bestFit,
    residual: summarize(items.map((x) => x.residual)),
    maxGap,
    gapAfter,
    bimodal: residuals.length >= 4 && maxGap > Math.max(threshold, 3 * (median(residuals) ?? 0)),
  };
}

// 与初始位姿无关的刚性一致性检验：
// 枚举所有三点组合做加权 Kabsch，取“能被同一刚性变换贴合的最多对应点数”。
// 若多数点可同时贴合，则配对本身成立，失败来自输入位姿/重叠；
// 若怎么都凑不出多数，则是配对错误。阈值仍由调用方显式给出。
function rigidConsistencyCheck(sourcePoints, targetPoints, weights, threshold) {
  requirePoints('sourcePoints', sourcePoints);
  requirePoints('targetPoints', targetPoints);
  requireFiniteNumber('threshold', threshold);
  const n = sourcePoints.length;
  if (n < 3) {
    return {
      available: true,
      sampleCount: 0,
      bestInlierCount: 0,
      bestInlierRatio: 0,
      bestInlierIndices: [],
      bestRms: null,
      note: '对应点少于 3 对，无法检验刚性一致性',
    };
  }
  const w = weights || new Array(n).fill(1);
  const inlierCountFor = (tf, tol = threshold) => {
    const R = tf.rotation;
    const t = tf.translation;
    const indices = [];
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const s = sourcePoints[i];
      const moved = [
        R[0] * s[0] + R[1] * s[1] + R[2] * s[2] + t[0],
        R[3] * s[0] + R[4] * s[1] + R[5] * s[2] + t[1],
        R[6] * s[0] + R[7] * s[1] + R[8] * s[2] + t[2],
      ];
      const d = vdist(moved, targetPoints[i]);
      if (d <= tol) { indices.push(i); sq += d * d; }
    }
    return { indices, rms: indices.length ? Math.sqrt(sq / indices.length) : null };
  };
  // 用三点解播种，再用内点重拟合迭代到稳定。三点拟合本身只能覆盖三点，
  // 若直接以内点数比较会低估；重拟合后才能反映“最多多少对能被同一刚性变换贴合”。
  // allowReflection 时允许重拟合落到反射极因子（仅用于镜像判别，不作为刚性输出）。
  const consensusFor = (seedTf, allowReflection = false) => {
    let tf = seedTf;
    let inliers = inlierCountFor(tf);
    let bestInliers = inliers;
    for (let round = 0; round < 4; round++) {
      if (inliers.indices.length < 3) break;
      let refined;
      try {
        refined = weightedKabsch(
          inliers.indices.map((i) => sourcePoints[i]),
          inliers.indices.map((i) => targetPoints[i]),
          {
            targetWeights: inliers.indices.map((i) => w[i]),
            sourceWeights: inliers.indices.map((i) => w[i]),
          },
        );
      } catch { break; }
      const nextTf = allowReflection && refined.reflectionFit
        ? { rotation: refined.reflectionFit.rotation, translation: refined.reflectionFit.translation }
        : { rotation: refined.rotation, translation: refined.translation };
      const nextInliers = inlierCountFor(nextTf);
      if (nextInliers.indices.length > bestInliers.indices.length ||
          (nextInliers.indices.length === bestInliers.indices.length &&
            nextInliers.rms != null && bestInliers.rms != null && nextInliers.rms < bestInliers.rms)) {
        bestInliers = nextInliers;
        tf = nextTf;
      }
      if (nextInliers.indices.length === inliers.indices.length &&
          nextInliers.rms != null && inliers.rms != null &&
          Math.abs(nextInliers.rms - inliers.rms) < 1e-9) break;
      inliers = nextInliers;
    }
    return { tf, inliers: bestInliers };
  };

  let best = { count: -1, indices: [], rms: null, transform: null };
  let bestReflection = { count: -1, indices: [], rms: null };
  // n 通常很小（8–12）；三点组合数量有限，穷举可保证确定性。
  // 对应点很多时（用户手工加点）不做全量 O(n^3)，改为确定性等距抽样三元组，
  // 保证拖动预览的日志/计算成本有上限，且结果可复现。
  const MAX_TRIPLETS = 400;
  const totalTriplets = n * (n - 1) * (n - 2) / 6;
  let tripletKeys;
  if (totalTriplets <= MAX_TRIPLETS) {
    tripletKeys = [];
    for (let a = 0; a < n; a++)
      for (let b = a + 1; b < n; b++)
        for (let c = b + 1; c < n; c++) tripletKeys.push([a, b, c]);
  } else {
    tripletKeys = [];
    for (let k = 0; k < MAX_TRIPLETS; k++) {
      const base = Math.floor((k * totalTriplets) / MAX_TRIPLETS);
      // 由序号还原确定的三元组下标组合
      let rem = base;
      let a = 0;
      while (rem >= (n - a - 1) * (n - a - 2) / 2) {
        rem -= (n - a - 1) * (n - a - 2) / 2;
        a++;
      }
      let b = a + 1;
      while (rem >= n - b - 1) { rem -= n - b - 1; b++; }
      const c = b + 1 + rem;
      if (c < n) tripletKeys.push([a, b, c]);
    }
  }
  for (const idx of tripletKeys) {
    let fit;
    try {
      fit = weightedKabsch(
        idx.map((i) => sourcePoints[i]),
        idx.map((i) => targetPoints[i]),
        { targetWeights: idx.map((i) => w[i]), sourceWeights: idx.map((i) => w[i]) },
      );
    } catch {
      continue;
    }
    // 三点必然共面（PCA 秩至多为 2），这里只排除真正退化的近共线三点：
    // 共线三元组无法确定唯一旋转，会产出伪刚性解并高估一致性。
    if (fit.degeneracy.rank < 2) continue;
    const proper = consensusFor({ rotation: fit.rotation, translation: fit.translation });
    if (proper.inliers.indices.length > best.count ||
        (proper.inliers.indices.length === best.count && best.rms != null &&
          proper.inliers.indices.length > 0 &&
          proper.inliers.rms != null && proper.inliers.rms < best.rms)) {
      best = {
        count: proper.inliers.indices.length,
        indices: proper.inliers.indices,
        rms: proper.inliers.rms,
        transform: proper.tf,
      };
    }
    // 反射模型：仅用于区分“镜像碎片”与“配对错误”，不替代刚性输出
    if (fit.reflectionFit) {
      const refl = consensusFor(
        { rotation: fit.reflectionFit.rotation, translation: fit.reflectionFit.translation },
        true,
      );
      if (refl.inliers.indices.length > bestReflection.count ||
          (refl.inliers.indices.length === bestReflection.count && bestReflection.rms != null &&
            refl.inliers.indices.length > 0 &&
            refl.inliers.rms != null && refl.inliers.rms < bestReflection.rms)) {
        bestReflection = {
          count: refl.inliers.indices.length,
          indices: refl.inliers.indices,
          rms: refl.inliers.rms,
        };
      }
    }
  }
  const ratio = best.count < 0 ? 0 : best.count / n;
  const reflectionRatio = bestReflection.count < 0 ? 0 : bestReflection.count / n;
  const outlierIndices = Array.from({ length: n }, (_, i) => i)
    .filter((i) => !(best.indices || []).includes(i));
  return {
    available: true,
    sampleCount: n,
    threshold,
    bestInlierCount: Math.max(0, best.count),
    bestInlierRatio: ratio,
    bestInlierIndices: best.indices || [],
    outlierIndices,
    bestRms: best.rms,
    bestTransform: best.transform,
    mutuallyConsistent: ratio >= 0.5,
    reflectionInlierCount: Math.max(0, bestReflection.count),
    reflectionInlierRatio: reflectionRatio,
    reflectionRms: bestReflection.rms,
    reflectionDominant: reflectionRatio >= 0.5 && reflectionRatio > ratio + 1e-9,
  };
}
function retainRounds(rounds, options = {}) {
  if (!Array.isArray(rounds)) throw diagError('DIAG_INPUT_MISSING', 'rounds 必须是数组');
  const maxRecords = options.maxRecords ?? 24;
  if (rounds.length <= maxRecords) {
    return { rounds: rounds.map((r) => r), omitted: 0, retained: rounds.length };
  }
  const head = options.head ?? 4;
  const tail = options.tail ?? 8;
  const kept = new Set();
  for (let i = 0; i < Math.min(head, rounds.length); i++) kept.add(i);
  for (let i = Math.max(0, rounds.length - tail); i < rounds.length; i++) kept.add(i);
  const middleSlots = Math.max(0, maxRecords - kept.size);
  if (middleSlots > 0) {
    const start = head;
    const end = rounds.length - tail;
    const span = Math.max(1, end - start);
    for (let k = 0; k < middleSlots; k++) {
      const idx = start + Math.floor((k + 1) * span / (middleSlots + 1));
      if (idx >= 0 && idx < rounds.length) kept.add(idx);
    }
  }
  const indices = Array.from(kept).sort((a, b) => a - b).slice(0, maxRecords);
  return {
    rounds: indices.map((i) => rounds[i]),
    omitted: rounds.length - indices.length,
    retained: indices.length,
    retainedIndices: indices,
  };
}

function residualTrendFromRounds(rounds) {
  const trend = [];
  let prev = null;
  for (const r of rounds) {
    const mean = r.residual && r.residual.mean;
    if (mean == null || !Number.isFinite(mean)) continue;
    const drop = prev == null ? null : prev - mean;
    trend.push({
      iteration: r.iteration,
      mean,
      drop,
      increased: drop != null && drop < 0,
    });
    prev = mean;
  }
  return trend;
}

function validCorrespondenceTrendFromRounds(rounds) {
  return rounds.map((r) => ({
    iteration: r.iteration,
    matched: r.matchedCount,
    unmatched: r.unmatchedCount,
    satisfiesMinimum: r.matchedCount >= MIN_PAIRS,
    belowMinimum: r.matchedCount < MIN_PAIRS,
  }));
}

function stageFinding(stage, code, severity, message, metrics, iteration) {
  return { stage, code, severity, message, metrics, iteration: iteration ?? null };
}

// 把异常定位到具体阶段：对应搜索 / 位姿更新 / 停止条件。
function buildStageFindings({ rounds, correspondence, stopCode, initialOverlap }) {
  const findings = [];
  if (correspondence) {
    if (correspondence.outlierCount > 0) {
      const tightInliers = correspondence.inlierCount >= 3 &&
        correspondence.residual &&
        correspondence.residual.median != null &&
        correspondence.residual.median <= correspondence.threshold;
      findings.push(stageFinding(
        STAGE.CORRESPONDENCE,
        tightInliers ? 'CORRESPONDENCE_OUTLIERS_FILTERED' : 'CORRESPONDENCE_MAJORITY_OUTSIDE_THRESHOLD',
        tightInliers ? 'info' : 'error',
        tightInliers
          ? `已筛除 ${correspondence.outlierCount} 对残差超阈值对应点，剩余 ${correspondence.inlierCount} 对紧致`
          : `仅 ${correspondence.inlierCount}/${correspondence.items.length} 对对应点位于阈值内，刚性模型与当前配对不一致`,
        {
          inlierCount: correspondence.inlierCount,
          outlierCount: correspondence.outlierCount,
          inlierRatio: correspondence.inlierRatio,
          threshold: correspondence.threshold,
          bimodal: correspondence.bimodal,
          maxGap: correspondence.maxGap,
        },
      ));
    }
    if (correspondence.bimodal) {
      findings.push(stageFinding(
        STAGE.CORRESPONDENCE, 'CORRESPONDENCE_BIMODAL_RESIDUAL', 'warning',
        `对应残差呈双峰（最大间隙 ${correspondence.maxGap.toFixed(4)}），存在错误配对而非统一噪声`,
        { maxGap: correspondence.maxGap, gapAfter: correspondence.gapAfter },
      ));
    }
  }

  if (initialOverlap != null && initialOverlap < 0.2) {
    findings.push(stageFinding(
      STAGE.CORRESPONDENCE, 'INITIAL_OVERLAP_LOW', 'error',
      `初始位姿下源点云仅有 ${(initialOverlap * 100).toFixed(1)}% 落在目标搜索半径内`,
      { initialOverlap },
    ));
  }

  for (let i = 0; i < rounds.length; i++) {
    const r = rounds[i];
    if (r.matchedCount < MIN_PAIRS) {
      findings.push(stageFinding(
        STAGE.CORRESPONDENCE, 'VALID_CORRESPONDENCE_BELOW_MIN', 'error',
        `第 ${r.iteration} 轮有效对应仅 ${r.matchedCount} 对（最少 ${MIN_PAIRS} 对）`,
        { matched: r.matchedCount, unmatched: r.unmatchedCount, minimum: MIN_PAIRS },
        r.iteration,
      ));
    } else if (i > 0 && rounds[i - 1].matchedCount > 0) {
      const ratio = r.matchedCount / rounds[i - 1].matchedCount;
      if (ratio < 0.5) {
        findings.push(stageFinding(
          STAGE.CORRESPONDENCE, 'VALID_CORRESPONDENCE_DROPPED', 'warning',
          `第 ${r.iteration} 轮有效对应从 ${rounds[i - 1].matchedCount} 降至 ${r.matchedCount}`,
          { previous: rounds[i - 1].matchedCount, current: r.matchedCount },
          r.iteration,
        ));
      }
    }
    if (r.step && i === 0 && !r.step.accepted && r.matchedCount >= MIN_PAIRS) {
      // 首轮有效对应充足却仍无法降低残差：位姿停在远离目标的平台上。
      // 这里只标注停止阶段现象，根因由 classifyCause 依据重叠/对应证据区分。
      findings.push(stageFinding(
        STAGE.STOP, 'POSE_UPDATE_REJECTED_FIRST_ROUND', 'warning',
        `首轮位姿更新未降低平均点距（${fmt(r.step.meanBefore)} → ${fmt(r.step.meanAfter)}），尚未进入可下降区域`,
        { meanBefore: r.step.meanBefore, meanAfter: r.step.meanAfter, stepNorm: r.step.stepNorm },
        r.iteration,
      ));
    } else if (r.step && !r.step.accepted) {
      findings.push(stageFinding(
        STAGE.POSE_UPDATE, 'POSE_STEP_REJECTED', 'warning',
        `第 ${r.iteration} 轮位姿更新未降低平均点距（${fmt(r.step.meanBefore)} → ${fmt(r.step.meanAfter)}）`,
        { meanBefore: r.step.meanBefore, meanAfter: r.step.meanAfter, stepNorm: r.step.stepNorm },
        r.iteration,
      ));
    }
    if (i > 0) {
      const prev = rounds[i - 1].residual;
      const cur = r.residual;
      if (prev && cur && cur.mean != null && prev.mean != null) {
        const drop = prev.mean - cur.mean;
        if (drop < 0) {
          findings.push(stageFinding(
            STAGE.POSE_UPDATE, 'RESIDUAL_INCREASED', 'error',
            `第 ${r.iteration} 轮平均距离残差从 ${prev.mean.toFixed(4)} 升至 ${cur.mean.toFixed(4)}`,
            { previous: prev.mean, current: cur.mean, drop },
            r.iteration,
          ));
        } else if (drop === 0 && cur.mean > 0) {
          findings.push(stageFinding(
            STAGE.POSE_UPDATE, 'RESIDUAL_NOT_DECREASING', 'warning',
            `第 ${r.iteration} 轮平均距离残差无下降（平台 ${cur.mean.toFixed(4)}）`,
            { previous: prev.mean, current: cur.mean, drop },
            r.iteration,
          ));
        }
      }
    }
  }

  if (stopCode) {
    findings.push(stageFinding(
      STAGE.STOP, stopCode, stopSeverity(stopCode),
      stopMessage(stopCode),
      { stopCode },
    ));
  }
  return findings;
}

function stopSeverity(code) {
  if (code === STOP.TOLERANCE || code === STOP.PLATEAU || code === STOP.CLOSED_FORM || code === STOP.RANSAC) return 'info';
  if (code === STOP.MAX_ITERATIONS) return 'warning';
  return 'error';
}

function stopMessage(code) {
  const map = {
    [STOP.TOLERANCE]: '步长低于容差，正常收敛',
    [STOP.PLATEAU]: '残差已处于采样噪声地板，判定收敛',
    [STOP.STAGNATED]: '位姿更新不再降低残差，且残差平台高于采样噪声地板：收敛停滞',
    [STOP.INSUFFICIENT]: '有效对应不足，优化未启动',
    [STOP.MAX_ITERATIONS]: '达到最大迭代次数仍未满足容差',
    [STOP.NUMERIC]: '法方程病态或步长非数值',
    [STOP.CLOSED_FORM]: 'Kabsch 闭式解完成',
    [STOP.RANSAC]: 'RANSAC 候选枚举完成',
  };
  return map[code] || code;
}

function fmt(x) {
  return Number.isFinite(x) ? x.toFixed(4) : String(x);
}

function pointCloudDiagonal(points) {
  requirePoints('points', points);
  const mn = [Infinity, Infinity, Infinity];
  const mx = [-Infinity, -Infinity, -Infinity];
  for (const p of points)
    for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
  return Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
}

const SCENE_DEFAULTS = Object.freeze({
  inlierFactor: 0.02,
  overlapFactor: 0.02,
  maxDistanceFactor: 2,
  maxIterations: 40,
  tolerance: 1e-8,
});

// 服务器端点与命令行工具共用同一推导规则：同一输入必然得到同一阈值，
// 避免两条路径各自隐式取默认值导致证据不可比。调用方显式传值优先。
function deriveSceneConfig({ targetPoints, correspondencePoints, overrides = {} }) {
  const targetDiag = pointCloudDiagonal(targetPoints);
  const corrDiag = pointCloudDiagonal(correspondencePoints);
  // 显式给出但非正/非有限的值必须报错；只有“未提供”（null/undefined）才走推导。
  const explicit = (name, v) => {
    if (v == null) return null;
    const n = Number(v);
    if (!Number.isFinite(n) || n <= 0) {
      throw diagError('DIAG_INPUT_MISSING', `${name} 必须是正有限数值，实际为 ${String(v)}`);
    }
    return n;
  };
  const exInlier = explicit('inlierThreshold', overrides.inlierThreshold);
  const exOverlap = explicit('overlapDistance', overrides.overlapDistance);
  const exMaxDist = explicit('maxDistance', overrides.maxDistance);
  const exHuberK = explicit('huberK', overrides.huberK);
  const exMaxIter = explicit('maxIterations', overrides.maxIterations);
  const exTol = explicit('tolerance', overrides.tolerance);
  const inlierThreshold = exInlier != null ? exInlier : corrDiag * SCENE_DEFAULTS.inlierFactor;
  const overlapDistance = exOverlap != null ? exOverlap : targetDiag * SCENE_DEFAULTS.overlapFactor;
  const maxDistance = exMaxDist != null ? exMaxDist : overlapDistance * SCENE_DEFAULTS.maxDistanceFactor;
  const huberK = overrides.huber === false ? null : (exHuberK != null ? exHuberK : overlapDistance);
  const maxIterations = exMaxIter != null ? Math.floor(exMaxIter) : SCENE_DEFAULTS.maxIterations;
  const tolerance = exTol != null ? exTol : SCENE_DEFAULTS.tolerance;
  return {
    inlierThreshold,
    overlapDistance,
    maxDistance,
    huberK,
    maxIterations,
    tolerance,
    derivedFrom: {
      inlierThreshold: exInlier != null ? 'explicit' : `correspondenceDiagonal*${SCENE_DEFAULTS.inlierFactor}`,
      overlapDistance: exOverlap != null ? 'explicit' : `targetDiagonal*${SCENE_DEFAULTS.overlapFactor}`,
      maxDistance: exMaxDist != null ? 'explicit' : `overlapDistance*${SCENE_DEFAULTS.maxDistanceFactor}`,
      huberK: overrides.huber === false ? 'disabled' : (exHuberK != null ? 'explicit' : 'overlapDistance'),
      sourceTargetDiagonal: { target: targetDiag, correspondence: corrDiag },
    },
  };
}

function pctText(x) {
  return x == null || !Number.isFinite(x) ? '未知' : `${(x * 100).toFixed(1)}%`;
}

// 综合分类：产出主因、次因及逐条观测证据。
// 优先级：尺度/单位 > 退化 > 对应错误 > 错误初始位姿 > 低重叠 > 局部极小值 > 对称歧义。
function classifyCause(input) {
  const {
    correspondence, rounds = [], stopCode, noiseFloor,
    initialOverlap, finalOverlap, targetCoverage,
    scaleRatios, degeneracy, symmetryAlternatives,
    rigidConsistency, inputPoseInliers, reachableOverlap,
  } = input;
  const evidence = [];
  const causes = [];
  const add = (code, message, metrics, stage) => {
    causes.push(code);
    evidence.push({ code, message, metrics, stage });
  };

  if (scaleRatios && scaleRatios.available && scaleRatios.uniform && scaleRatios.deviatesFromUnit) {
    add(CAUSE.SCALE_UNIT_MISMATCH,
      `对应点距离比中位数 ${scaleRatios.median.toFixed(4)}，稳定偏离 1（单位或尺度不一致）`,
      { median: scaleRatios.median, iqr: scaleRatios.iqr, samples: scaleRatios.samples },
      STAGE.INPUT);
  }

  if (degeneracy && (degeneracy.rank < 3 || (degeneracy.planarRatio != null && degeneracy.planarRatio < 1e-3))) {
    add(CAUSE.DEGENERATE_GEOMETRY,
      `对应点几何秩 ${degeneracy.rank}，存在欠定方向`,
      { rank: degeneracy.rank, planarRatio: degeneracy.planarRatio },
      STAGE.CORRESPONDENCE);
  }

  // 配对错误的判定与“初始位姿好坏”无关：
  // 只有刚性一致性检验（存在能让多数点同时贴合的刚性变换吗）不过关时，
  // 才是配对本身错误；若多数点可贴合，则对应点成立，失败归因于输入位姿/重叠。
  const consistent = rigidConsistency ? rigidConsistency.mutuallyConsistent : null;
  if (rigidConsistency && rigidConsistency.reflectionDominant) {
    // 反射模型明显优于刚性模型：左手系/镜像碎片，而非配对错误
    add(CAUSE.SYMMETRY_AMBIGUITY,
      `反射变换可贴合 ${rigidConsistency.reflectionInlierCount}/${rigidConsistency.sampleCount} 对，多于刚性拟合的 ${rigidConsistency.bestInlierCount} 对：疑似镜像/左右手翻转`,
      {
        reflectionInlierCount: rigidConsistency.reflectionInlierCount,
        rigidInlierCount: rigidConsistency.bestInlierCount,
        sampleCount: rigidConsistency.sampleCount,
        reflectionRms: rigidConsistency.reflectionRms,
      },
      STAGE.CORRESPONDENCE);
  } else if (rigidConsistency && consistent === false) {
    add(CAUSE.CORRESPONDENCE_ERROR,
      `不存在能让多数对应点同时贴合的刚性变换（最佳 ${rigidConsistency.bestInlierCount}/${rigidConsistency.sampleCount}），配对自身不一致`,
      {
        bestInlierCount: rigidConsistency.bestInlierCount,
        bestInlierRatio: rigidConsistency.bestInlierRatio,
        bestRms: rigidConsistency.bestRms,
        threshold: rigidConsistency.threshold,
        outlierIndices: rigidConsistency.outlierIndices,
      },
      STAGE.CORRESPONDENCE);
  } else if (rigidConsistency && consistent === true && rigidConsistency.outlierIndices.length > 0) {
    // 多数点一致、少数点离群：错误对应会被筛除，如实记录但不升级为主因
    add(CAUSE.CORRESPONDENCE_ERROR,
      `多数对应点可由同一刚性变换贴合，另有 ${rigidConsistency.outlierIndices.length} 对为离群错误配对：${rigidConsistency.outlierIndices.join(', ')}`,
      {
        bestInlierCount: rigidConsistency.bestInlierCount,
        sampleCount: rigidConsistency.sampleCount,
        outlierIndices: rigidConsistency.outlierIndices,
      },
      STAGE.CORRESPONDENCE);
  } else if (correspondence && correspondence.bimodal && correspondence.outlierCount > 0) {
    const inputInliers = inputPoseInliers == null ? correspondence.inlierCount : inputPoseInliers;
    add(CAUSE.CORRESPONDENCE_ERROR,
      `对应残差双峰，${correspondence.outlierCount} 对为错误配对（最大间隙 ${correspondence.maxGap.toFixed(4)}）`,
      { inputPoseInliers, outlierCount: correspondence.outlierCount, maxGap: correspondence.maxGap },
      STAGE.CORRESPONDENCE);
  }

  // 输入阶段：用“配对是否刚性一致”区分错误初值与低重叠。
  //  - 配对一致、且存在可达位姿带来高重叠，而初始位姿重叠低 → 错误初始位姿
  //  - 配对一致、但可达位姿也无法达到高重叠 → 碎片真实低重叠
  const pairingConsistent = rigidConsistency ? rigidConsistency.mutuallyConsistent : true;
  const ceiling = reachableOverlap != null ? reachableOverlap : null;
  const poseClearlyWorse = ceiling != null && ceiling >= 0.5 &&
    initialOverlap != null && initialOverlap < 0.5 &&
    initialOverlap < ceiling * 0.5;
  // 点云整体重叠可能仍然很高（形状近似对称/扁平），但当前位姿下对应点残差大面积超阈值时，
  // 对应点本身已足以证明位姿不对：只要配对可刚性贴合（最佳内点明显多于当前内点），
  // 仍归因错误初始位姿，而不是“无异常”。
  const corrMostlyOutliers = correspondence && correspondence.items &&
    correspondence.items.length >= 3 &&
    correspondence.inlierCount / correspondence.items.length < 0.5;
  const pairingFitsBetter = rigidConsistency && rigidConsistency.mutuallyConsistent &&
    rigidConsistency.bestInlierCount != null && inputPoseInliers != null &&
    rigidConsistency.bestInlierCount >= inputPoseInliers + 3;
  const poseRejectedByCorrespondence = !poseClearlyWorse && corrMostlyOutliers && pairingFitsBetter;
  if (!causes.includes(CAUSE.CORRESPONDENCE_ERROR) &&
      !causes.includes(CAUSE.SYMMETRY_AMBIGUITY) && pairingConsistent &&
      (poseClearlyWorse || poseRejectedByCorrespondence)) {
    // 存在可达位姿而初始位姿明显远离：无论 ICP 是否部分恢复，输入位姿都已被观测判为错误。
    add(CAUSE.WRONG_INITIAL_POSE,
      `配对可刚性贴合，存在可达位姿重叠 ${pctText(ceiling)}；初始位姿仅 ${pctText(initialOverlap)}，` +
      `不足可达值的 ${(initialOverlap / ceiling * 100).toFixed(0)}%`,
      { initialOverlap, reachableOverlap: ceiling },
      STAGE.CORRESPONDENCE);
  } else if (!causes.includes(CAUSE.CORRESPONDENCE_ERROR) &&
             !causes.includes(CAUSE.SYMMETRY_AMBIGUITY) && pairingConsistent &&
             ceiling != null && ceiling < 0.5 &&
             ((initialOverlap != null && initialOverlap < 0.5) ||
              (finalOverlap != null && finalOverlap < 0.5))) {
    // 任何由一致对应点导出的位姿都无法达到高重叠：碎片自身重叠不足。
    add(CAUSE.LOW_OVERLAP,
      `配对可刚性贴合，但最佳可达重叠仅 ${pctText(ceiling)}（初始 ${pctText(initialOverlap)}、最终 ${pctText(finalOverlap)}）：碎片真实重叠不足`,
      { initialOverlap, finalOverlap, reachableOverlap: ceiling },
      STAGE.CORRESPONDENCE);
  }
  const overlapLow = (finalOverlap != null && finalOverlap < 0.5) &&
    (targetCoverage == null || targetCoverage < 0.5);
  if (overlapLow && !causes.includes(CAUSE.WRONG_INITIAL_POSE) &&
      !causes.includes(CAUSE.LOW_OVERLAP) &&
      !causes.includes(CAUSE.CORRESPONDENCE_ERROR)) {
    add(CAUSE.LOW_OVERLAP,
      `最终重叠 ${pctText(finalOverlap)}，目标覆盖 ${pctText(targetCoverage)}`,
      { finalOverlap, targetCoverage },
      STAGE.CORRESPONDENCE);
  }

  // 残差平台明显高于采样噪声地板、位姿更新停滞、配对自身一致且存在可达高重叠，
  // 但位姿没有明显被初值甩开（初值重叠不低）：这是停在错误盆地的局部极小值。
  // 周期/近对称几何（如花口、齿状断面）会让 ICP 停在错位但表面仍贴合的位置。
  const lastRound = rounds[rounds.length - 1];
  const plateau = lastRound && lastRound.residual ? lastRound.residual.mean : null;
  const floor = noiseFloor && noiseFloor.median;
  const plateauFactor = plateau != null && floor > 0 ? plateau / floor : null;
  const noInputCause = !causes.includes(CAUSE.CORRESPONDENCE_ERROR) &&
    !causes.includes(CAUSE.SYMMETRY_AMBIGUITY) &&
    !causes.includes(CAUSE.WRONG_INITIAL_POSE) &&
    !causes.includes(CAUSE.LOW_OVERLAP);
  const reachableHigh = ceiling != null && ceiling >= 0.5;
  if (stopCode === STOP.STAGNATED && plateauFactor != null &&
      plateauFactor > PLATEAU_NOISE_FACTOR && reachableHigh &&
      pairingConsistent && noInputCause) {
    add(CAUSE.LOCAL_MINIMUM,
      `存在可达高重叠位姿 ${pctText(ceiling)}，但 ICP 停在同一盆地内：` +
      `残差平台 ${plateau.toFixed(3)} 为采样噪声地板 ${floor.toFixed(3)} 的 ${plateauFactor.toFixed(2)} 倍`,
      {
        plateau,
        noiseFloor: floor,
        plateauFactor,
        thresholdFactor: PLATEAU_NOISE_FACTOR,
        initialOverlap,
        finalOverlap,
        reachableOverlap: ceiling,
      },
      STAGE.STOP);
  }

  if (symmetryAlternatives && symmetryAlternatives.length > 0) {
    const best = symmetryAlternatives.reduce((a, b) =>
      (a.cloudRms ?? Infinity) <= (b.cloudRms ?? Infinity) ? a : b);
    const spread = symmetryAlternatives
      .map((a) => a.cloudRms)
      .filter((x) => x != null && Number.isFinite(x));
    if (spread.length > 1) {
      const bestRms = Math.min(...spread);
      const worstRms = Math.max(...spread);
      const nearEqual = worstRms <= bestRms * 1.05 + 1e-9;
      if (nearEqual) {
        add(CAUSE.SYMMETRY_AMBIGUITY,
          `对称备选与主候选残差近似相等（${bestRms.toFixed(3)} / ${worstRms.toFixed(3)}），朝向不可由残差区分`,
          { bestRms, worstRms, alternatives: symmetryAlternatives.length },
          STAGE.POSE_UPDATE);
      }
    }
    void best;
  }

  if (stopCode === STOP.NUMERIC) {
    add(CAUSE.NUMERIC_FAILURE, '位姿更新出现非数值步长，法方程病态',
      { stopCode }, STAGE.POSE_UPDATE);
  }

  // 去重但保持优先级顺序
  const unique = [];
  for (const c of causes) if (!unique.includes(c)) unique.push(c);
  const uniqueEvidence = [];
  const seen = new Set();
  for (const e of evidence) {
    const key = `${e.code}|${e.message}`;
    if (seen.has(key)) continue;
    seen.add(key);
    uniqueEvidence.push(e);
  }
  return {
    primary: unique[0] || CAUSE.NONE,
    secondary: unique.slice(1),
    evidence: uniqueEvidence,
    plateauFactor,
  };
}

// 复跑一致性指纹：只覆盖确定性的观测字段，不含时间戳。
function evidenceFingerprint(evidence) {
  if (!evidence || typeof evidence !== 'object') {
    throw diagError('DIAG_INPUT_MISSING', '缺少证据对象');
  }
  return fingerprint({
    version: evidence.version,
    config: evidence.config,
    input: evidence.input,
    rounds: evidence.rounds,
    residualTrend: evidence.residualTrend,
    validCorrespondenceTrend: evidence.validCorrespondenceTrend,
    stageFindings: evidence.stageFindings,
    cause: evidence.cause,
  });
}

// 重放单轮：使用保存的输入位姿与搜索参数，复算对应和残差。
// 返回的 fingerprint 必须与快照一致，才能证明这一轮可独立重放。
function replayRound(sourcePoints, targetPoints, snapshot) {
  requirePoints('sourcePoints', sourcePoints);
  requirePoints('targetPoints', targetPoints);
  if (!snapshot || !snapshot.inputPose) {
    throw diagError('DIAG_INPUT_MISSING', '重放需要 inputPose 快照');
  }
  const pose = snapshot.inputPose;
  if (!Array.isArray(pose.rotation) || pose.rotation.length !== 9 ||
      !Array.isArray(pose.translation) || pose.translation.length !== 3) {
    throw diagError('DIAG_INPUT_MISSING', 'inputPose 必须是 3x3 旋转与三维平移');
  }
  const maxDistance = requireFiniteNumber('maxDistance', snapshot.maxDistance);
  const huberK = snapshot.huberK == null ? null : requireFiniteNumber('huberK', snapshot.huberK);
  const moved = sourcePoints.map((p) => [
    pose.rotation[0] * p[0] + pose.rotation[1] * p[1] + pose.rotation[2] * p[2] + pose.translation[0],
    pose.rotation[3] * p[0] + pose.rotation[4] * p[1] + pose.rotation[5] * p[2] + pose.translation[1],
    pose.rotation[6] * p[0] + pose.rotation[7] * p[1] + pose.rotation[8] * p[2] + pose.translation[2],
  ]);
  const stats = nearestCorrespondenceStats(moved, targetPoints, maxDistance, huberK);
  const replay = {
    inputPose: { rotation: pose.rotation.slice(), translation: pose.translation.slice() },
    maxDistance,
    huberK,
    matchedCount: stats.matchedCount,
    unmatchedCount: stats.unmatchedCount,
    weightSum: stats.weightSum,
    residual: stats.residual,
    pairs: stats.pairs,
  };
  const replayFingerprint = roundRecordFingerprint(replay);
  return {
    ...replay,
    replayFingerprint,
    originalFingerprint: snapshot.fingerprint || null,
    match: replayFingerprint === snapshot.fingerprint,
  };
}

// 从迭代记录生成趋势 + 阶段定位 + 分类，供内核与导出共用。
function buildIcpDiagnostics({ rounds, stopCode, noiseFloor, initialOverlap,
  finalOverlap, targetCoverage, scaleRatios, degeneracy, symmetryAlternatives,
  correspondence, rigidConsistency, inputPoseInliers, reachableOverlap }) {
  const limited = retainRounds(rounds, { maxRecords: 24, head: 4, tail: 8 });
  const retained = limited.rounds;
  const residualTrend = residualTrendFromRounds(retained);
  const validCorrespondenceTrend = validCorrespondenceTrendFromRounds(retained);
  const stageFindings = buildStageFindings({
    rounds: retained,
    correspondence,
    stopCode,
    initialOverlap,
  });
  const cause = classifyCause({
    correspondence,
    rounds: retained,
    stopCode,
    noiseFloor,
    initialOverlap,
    finalOverlap,
    targetCoverage,
    scaleRatios,
    degeneracy,
    symmetryAlternatives,
    rigidConsistency,
    inputPoseInliers,
    reachableOverlap,
  });
  return {
    budget: {
      maxRounds: 24,
      totalRounds: rounds.length,
      retainedRounds: limited.retained,
      omittedRounds: limited.omitted,
      pairSampleCap: 32,
    },
    rounds: retained,
    residualTrend,
    validCorrespondenceTrend,
    stageFindings,
    cause,
  };
}

// 收集一次配准的完整迭代证据。服务器 API 与命令行工具共用同一实现，
// 保证“同一输入 → 同一证据 → 同一指纹”，不因调用路径不同而变化。
// icpResult 由内核（registration.icpRefine）产出；本函数不调用内核、不读真值。
function buildIcpEvidence({ sourceCloud, targetCloud, correspondence, initialPose,
  icpResult, inlierThreshold, overlapDistance, maxIterations, huberK, method }) {
  requirePoints('sourceCloud.points', sourceCloud.points);
  requirePoints('targetCloud.points', targetCloud.points);
  requireFiniteNumber('inlierThreshold', inlierThreshold);
  requireFiniteNumber('overlapDistance', overlapDistance);
  if (!icpResult || !Array.isArray(icpResult.rounds)) {
    throw diagError('DIAG_INPUT_MISSING', 'buildIcpEvidence 需要内核迭代记录 rounds');
  }
  if (!correspondence || !correspondence.length) {
    throw diagError('DIAG_INPUT_MISSING', '缺少对应点，无法建立证据链');
  }

  const sourcePts = correspondence.map((c) => c.sourcePoint);
  const targetPts = correspondence.map((c) => c.targetPoint);
  const weights = correspondence.map((c) => {
    const denom = c.targetWeight + c.sourceWeight;
    return denom > 1e-15 ? (c.targetWeight * c.sourceWeight) / denom : 0;
  });

  // 初始位姿下的观测：有效对应、逐对报告中位数、重叠率
  const initialMoved = applyPose(sourceCloud.points, initialPose);
  const initialCoverage = coverageStats(initialMoved, targetCloud.points, overlapDistance);
  const initialCorrespondence = correspondenceReport(
    sourcePts, targetPts, weights, initialPose, inlierThreshold,
  );

  // 最终位姿下的对应与覆盖
  const finalMoved = applyPose(sourceCloud.points, icpResult.transform);
  const finalCoverage = coverageStats(finalMoved, targetCloud.points, overlapDistance);
  const finalCorrespondence = correspondenceReport(
    sourcePts, targetPts, weights, icpResult.transform, inlierThreshold,
  );

  const noiseFloor = icpResult.noiseFloor || samplingNoiseFloor(sourceCloud.points);
  const scaleRatios = correspondenceDistanceRatios(sourcePts, targetPts);
  // 与初始位姿无关的配对一致性检验：区分配对错误与输入位姿错误。
  const rigidConsistency = rigidConsistencyCheck(sourcePts, targetPts, weights, inlierThreshold);
  // 由对应点导出的“可达最重叠加”：若存在某个刚性位姿（由一致内点拟合）能让点云高度重叠，
  // 而当前初始位姿不能，则失败来自初始位姿，而非碎片真的重叠不足。
  let reachableOverlap = null;
  if (rigidConsistency.bestTransform) {
    const reachMoved = applyPose(sourceCloud.points, rigidConsistency.bestTransform);
    reachableOverlap = coverageStats(reachMoved, targetCloud.points, overlapDistance);
  }

  // 根因分类以“输入状态”为准：初始位姿下的对应/重叠才区分错误初值、错误对应与低重叠。
  // 已更新到的最终位姿不能反过来抹掉这些输入阶段的异常。
  const diagnostics = buildIcpDiagnostics({
    rounds: icpResult.rounds,
    stopCode: icpResult.stopCode,
    noiseFloor,
    initialOverlap: initialCoverage.overlapFraction,
    finalOverlap: finalCoverage.overlapFraction,
    targetCoverage: finalCoverage.targetCoverageFraction,
    scaleRatios,
    degeneracy: icpResult.degeneracy || null,
    symmetryAlternatives: icpResult.symmetryAlternatives || [],
    correspondence: initialCorrespondence,
    rigidConsistency,
    inputPoseInliers: initialCorrespondence.inlierCount,
    reachableOverlap: reachableOverlap ? reachableOverlap.overlapFraction : null,
  });

  const evidence = {
    version: 1,
    method,
    config: { inlierThreshold, overlapDistance, maxIterations, huberK },
    input: {
      sourcePointCount: sourceCloud.points.length,
      targetPointCount: targetCloud.points.length,
      correspondence: correspondence.map((c) => ({
        id: c.id ?? null,
        targetPoint: c.targetPoint,
        sourcePoint: c.sourcePoint,
        targetWeight: c.targetWeight,
        sourceWeight: c.sourceWeight,
        fusedWeight: c.targetWeight + c.sourceWeight > 1e-15
          ? (c.targetWeight * c.sourceWeight) / (c.targetWeight + c.sourceWeight) : 0,
      })),
    },
    initialPose: {
      rotation: initialPose.rotation.slice(),
      translation: initialPose.translation.slice(),
    },
    finalPose: {
      rotation: icpResult.transform.rotation.slice(),
      translation: icpResult.transform.translation.slice(),
    },
    correspondenceInitial: summarizeCorrespondence(initialCorrespondence),
    correspondenceFinal: summarizeCorrespondence(finalCorrespondence),
    coverage: {
      initialOverlapFraction: initialCoverage.overlapFraction,
      initialTargetCoverageFraction: initialCoverage.targetCoverageFraction,
      finalOverlapFraction: finalCoverage.overlapFraction,
      finalTargetCoverageFraction: finalCoverage.targetCoverageFraction,
      finalCloudRms: finalCoverage.cloudRms,
      unmatchedSourceCount: finalCoverage.unmatchedSourceCount,
    },
    noiseFloor,
    scaleRatios,
    rigidConsistency,
    stop: {
      code: icpResult.stopCode,
      reason: icpResult.reason,
      converged: icpResult.converged,
      iterations: icpResult.iterations,
    },
    rounds: diagnostics.rounds,
    roundBudget: diagnostics.budget,
    residualTrend: diagnostics.residualTrend,
    validCorrespondenceTrend: diagnostics.validCorrespondenceTrend,
    stageFindings: diagnostics.stageFindings,
    cause: diagnostics.cause,
  };
  evidence.fingerprint = evidenceFingerprint(evidence);
  return evidence;
}

function applyPose(points, pose) {
  const R = pose.rotation;
  const t = pose.translation;
  return points.map((s) => [
    R[0] * s[0] + R[1] * s[1] + R[2] * s[2] + t[0],
    R[3] * s[0] + R[4] * s[1] + R[5] * s[2] + t[1],
    R[6] * s[0] + R[7] * s[1] + R[8] * s[2] + t[2],
  ]);
}

function summarizeCorrespondence(report) {
  return {
    threshold: report.threshold,
    inlierCount: report.inlierCount,
    outlierCount: report.outlierCount,
    inlierRatio: report.inlierRatio,
    residual: report.residual,
    bimodal: report.bimodal,
    maxGap: report.maxGap,
    items: report.items.map((x) => ({
      pairIndex: x.pairIndex,
      residual: x.residual,
      weight: x.weight,
      inlier: x.inlier,
      reasonCode: x.reasonCode,
      reason: x.reason,
    })),
  };
}

module.exports = {
  STAGE,
  CAUSE,
  STOP,
  MIN_PAIRS,
  PLATEAU_NOISE_FACTOR,
  requireFiniteNumber,
  requirePoints,
  median,
  quantile,
  summarize,
  samplingNoiseFloor,
  correspondenceDistanceRatios,
  coverageStats,
  nearestCorrespondenceStats,
  correspondenceReport,
  rigidConsistencyCheck,
  retainRounds,
  residualTrendFromRounds,
  validCorrespondenceTrendFromRounds,
  buildStageFindings,
  classifyCause,
  buildIcpDiagnostics,
  buildIcpEvidence,
  pointCloudDiagonal,
  SCENE_DEFAULTS,
  deriveSceneConfig,
  roundRecordFingerprint,
  evidenceFingerprint,
  replayRound,
  canonicalize,
  stableStringify,
  fingerprint,
};

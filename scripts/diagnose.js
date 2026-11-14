#!/usr/bin/env node
// 配准诊断工具（命令行）：不启动服务器即可完整复跑一轮配准并输出执行证据链。
//
// 用法：
//   node scripts/diagnose.js --scenario partial --seed 11 \\
//     --perturb-angle 35 --perturb-axis 0,1,0 --perturb-translation 40,0,0
//   node scripts/diagnose.js --scenario mirrored --show-rounds
//   node scripts/diagnose.js --evidence exports/evidence.json --replay
//
// 约束：
//  - 诊断代码绝不读取 dataset.meta.groundTruth；--check-truth 仅作为独立校验器，
//    在校验时才惰性读取真值，且不参与求解或根因分类。
//  - 缺少参数时不使用隐藏默认值：所有阈值在输出的 config 中显式列出。

'use strict';

const fs = require('fs');
const path = require('path');
const {
  generateDataset, axisAngleMatrix, mulberry32,
} = require('../server/synthdata');
const {
  weightedKabsch, applyTransform, mMul, mTranspose, mApply,
} = require('../server/linalg');
const reg = require('../server/registration');
const diag = require('../server/diagnostics');

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    if (key === 'help') { out.help = true; continue; }
    const hasValue = i + 1 < argv.length && !argv[i + 1].startsWith('--');
    if (hasValue && !['replay', 'show-rounds', 'check-truth', 'json', 'strict'].includes(key)) {
      out[key] = argv[++i];
    } else {
      out[key] = true;
    }
  }
  return out;
}

function parseVec(text, name) {
  if (text == null) return null;
  const parts = String(text).split(',').map((x) => Number(x.trim()));
  if (parts.length !== 3 || parts.some((x) => !Number.isFinite(x))) {
    throw new Error(`${name} 需要 3 个有限数值，实际为 "${text}"`);
  }
  return parts;
}

const DEFAULT_SCENE_PARAMS = diag.SCENE_DEFAULTS;

// 与服务器 ICP/诊断端点完全相同的推导规则（同一函数，避免两处漂移）
function deriveConfig(dataset, args) {
  if (args['max-iterations'] != null && !Number.isFinite(Number(args['max-iterations']))) {
    throw new Error('--max-iterations 必须是数值');
  }
  const config = diag.deriveSceneConfig({
    targetPoints: dataset.target.cloud.points,
    correspondencePoints: dataset.landmarks.flatMap((l) => [l.targetPoint, l.sourcePoint]),
    overrides: {
      inlierThreshold: args['inlier-threshold'],
      overlapDistance: args['overlap-distance'],
      maxDistance: args['max-distance'],
      maxIterations: args['max-iterations'],
      huber: args['no-huber'] ? false : undefined,
    },
  });
  return {
    ...config,
    derivedFrom: {
      inlierThreshold: config.derivedFrom.inlierThreshold,
      overlapDistance: config.derivedFrom.overlapDistance,
      maxDistance: config.derivedFrom.maxDistance,
      huberK: args['no-huber'] ? 'disabled' : 'overlapDistance',
    },
  };
}

function requireNumber(name, v) {
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`${name} 必须是有限数值，实际为 ${v}`);
  return n;
}

function correspondenceFromDataset(dataset) {
  return dataset.landmarks.map((l) => ({
    id: l.id,
    targetPoint: l.targetPoint,
    sourcePoint: l.sourcePoint,
    targetWeight: l.targetWeight,
    sourceWeight: l.sourceWeight,
  }));
}

function initialPoseFromDataset(dataset, corr, args) {
  const fit = weightedKabsch(
    corr.map((c) => c.sourcePoint),
    corr.map((c) => c.targetPoint),
    {
      targetWeights: corr.map((c) => c.targetWeight),
      sourceWeights: corr.map((c) => c.sourceWeight),
    },
  );
  return { rotation: fit.rotation, translation: fit.translation };
}

function perturbPose(pose, args) {
  const angleDeg = args['perturb-angle'] != null
    ? requireNumber('--perturb-angle', args['perturb-angle']) : 0;
  const axis = parseVec(args['perturb-axis'] || '0,1,0', '--perturb-axis');
  const offset = parseVec(args['perturb-translation'] || '0,0,0', '--perturb-translation');
  if (angleDeg === 0 && offset.every((x) => x === 0)) return pose;
  const Rp = axisAngleMatrix(axis, (angleDeg * Math.PI) / 180);
  return {
    rotation: mMul(Rp, pose.rotation),
    translation: pose.translation.map((x, i) => x + offset[i]),
  };
}

function renderText(report) {
  const lines = [];
  const push = (s = '') => lines.push(s);
  push(`# 配准诊断：${report.scenario} / seed ${report.seed}`);
  push(`诊断版本 ${report.evidence.version} · 方法 ${report.evidence.method} · 指纹 ${report.evidence.fingerprint}`);
  push('');
  push('## 输入');
  push(`  源点 ${report.evidence.input.sourcePointCount} · 目标点 ${report.evidence.input.targetPointCount} · 对应点 ${report.evidence.input.correspondence.length}`);
  push(`  单位/坐标约定: ${report.datasetConventions.unit} / ${report.datasetConventions.handedness} / 上轴 ${report.datasetConventions.upAxis}`);
  push(`  阈值: 内点 ${report.evidence.config.inlierThreshold.toFixed(4)} · 重叠半径 ${report.evidence.config.overlapDistance.toFixed(4)} · 搜索半径 ${report.config.maxDistance.toFixed(4)} · Huber ${report.evidence.config.huberK == null ? '关闭' : report.evidence.config.huberK.toFixed(4)}`);
  push(`  参数来源: ${JSON.stringify(report.config.derivedFrom)}`);
  push('');
  push('## 初始状态观测');
  push(`  初始重叠 ${pct(report.evidence.coverage.initialOverlapFraction)} · 目标覆盖 ${pct(report.evidence.coverage.initialTargetCoverageFraction)}`);
  const ic = report.evidence.correspondenceInitial;
  push(`  初始对应内点 ${ic.inlierCount}/${ic.items.length} · 残差中位 ${num(ic.residual && ic.residual.median)} · 双峰 ${ic.bimodal}`);
  push(`  距离比中位 ${report.evidence.scaleRatios.available ? report.evidence.scaleRatios.median.toFixed(4) : '不可用'} · 稳定偏离 1: ${report.evidence.scaleRatios.available ? report.evidence.scaleRatios.deviatesFromUnit : '-'}`);
  push(`  采样噪声地板 ${report.evidence.noiseFloor.median.toFixed(4)}（${report.evidence.noiseFloor.sampleCount} 采样）`);
  push('');
  push('## 迭代执行证据（受预算裁剪）');
  push(`  总轮数 ${report.evidence.roundBudget.totalRounds} · 保留 ${report.evidence.roundBudget.retainedRounds} · 省略 ${report.evidence.roundBudget.omittedRounds} · 每轮对应样本上限 ${report.evidence.roundBudget.pairSampleCap}`);
  for (const r of report.evidence.rounds) {
    const res = r.residual ? r.residual.mean : null;
    const step = r.step || {};
    push(`  第 ${r.iteration} 轮: 输入位姿 ${r.inputFingerprint} · 有效对应 ${r.matchedCount}/${r.matchedCount + r.unmatchedCount} · 平均距离 ${num(res)} · 步长 ${num(step.stepNorm)} · 更新${step.accepted ? '接受' : '拒绝'}`);
  }
  push('');
  push('## 残差下降');
  for (const t of report.evidence.residualTrend) {
    push(`  第 ${t.iteration} 轮: 平均距离 ${num(t.mean)} · 下降 ${num(t.drop)}${t.increased ? '（上升）' : ''}`);
  }
  push('');
  push('## 有效对应趋势');
  for (const t of report.evidence.validCorrespondenceTrend) {
    push(`  第 ${t.iteration} 轮: 匹配 ${t.matched} · 未匹配 ${t.unmatched} · 满足最少 ${t.satisfiesMinimum ? '是' : '否'}`);
  }
  push('');
  push('## 阶段定位（异常落到阶段）');
  for (const f of report.evidence.stageFindings) {
    push(`  [${f.stage}/${f.severity}] ${f.code}: ${f.message}`);
  }
  push('');
  push('## 根因');
  push(`  主因 ${report.evidence.cause.primary}`);
  if (report.evidence.cause.secondary.length)
    push(`  次因 ${report.evidence.cause.secondary.join(', ')}`);
  for (const e of report.evidence.cause.evidence) {
    push(`  - [${e.stage}] ${e.code}: ${e.message}`);
  }
  push('');
  push('## 停止条件');
  push(`  ${report.evidence.stop.code}（收敛=${report.evidence.stop.converged}）: ${report.evidence.stop.reason}`);
  push('');
  push('## 单轮重放');
  push(`  全部一致: ${report.replay.stable ? '是' : '否'}`);
  for (const r of report.replay.rounds) {
    push(`  第 ${r.iteration} 轮: 记录 ${r.fingerprint} · 重放 ${r.replayFingerprint} · ${r.match ? '一致' : '不一致'}`);
  }
  if (report.truthCheck) {
    push('');
    push('## 独立真值校验（不参与求解）');
    push(`  ${JSON.stringify(report.truthCheck)}`);
  }
  return lines.join('\n');
}

function pct(x) { return x == null ? '未知' : `${(x * 100).toFixed(1)}%`; }
function num(x) { return x == null || !Number.isFinite(x) ? '—' : x.toFixed(4); }

function runDiagnosis(args, io = console) {
  const scenario = args.scenario || 'bowl';
  const seed = args.seed != null ? requireNumber('--seed', args.seed) : 42;
  const dataset = generateDataset(scenario, seed);
  const corr = correspondenceFromDataset(dataset);
  if (corr.length < 3) throw new Error('对应点不足 3 对，无法配准');
  // --source-scale 用于复现“单位/尺度不一致”（如 mm 与 inch 混用），
  // 同时作用于源点云与对应点源端，制造一致的距离比偏差。
  const sourceScale = args['source-scale'] != null
    ? requireNumber('--source-scale', args['source-scale']) : 1;
  if (sourceScale <= 0) throw new Error('--source-scale 必须为正数');
  if (sourceScale !== 1) {
    dataset.source.cloud = {
      points: dataset.source.cloud.points.map((p) => p.map((x) => x * sourceScale)),
      normals: dataset.source.cloud.normals.slice(),
    };
    for (const c of corr) c.sourcePoint = c.sourcePoint.map((x) => x * sourceScale);
  }
  const config = deriveConfig(dataset, args);
  const basePose = initialPoseFromDataset(dataset, corr, args);
  const pose = perturbPose(basePose, args);

  const result = reg.icpRefine(dataset.source.cloud, dataset.target.cloud, pose, {
    maxIterations: config.maxIterations,
    tolerance: config.tolerance,
    maxDistance: config.maxDistance,
    huberK: config.huberK,
  });
  const evidence = diag.buildIcpEvidence({
    sourceCloud: dataset.source.cloud,
    targetCloud: dataset.target.cloud,
    correspondence: corr,
    initialPose: pose,
    icpResult: result,
    inlierThreshold: config.inlierThreshold,
    overlapDistance: config.overlapDistance,
    maxIterations: config.maxIterations,
    huberK: config.huberK,
    method: 'icp-cli',
  });
  const replayRounds = evidence.rounds.map((r) => {
    const snap = {
      inputPose: r.inputPose,
      maxDistance: config.maxDistance,
      huberK: config.huberK,
      fingerprint: r.inputFingerprint,
    };
    const check = diag.replayRound(dataset.source.cloud.points, dataset.target.cloud.points, snap);
    return {
      iteration: r.iteration,
      fingerprint: snap.fingerprint,
      replayFingerprint: check.replayFingerprint,
      match: check.match,
      matchedCount: check.matchedCount,
      residual: check.residual,
    };
  });
  const replay = {
    stable: replayRounds.every((r) => r.match),
    rounds: replayRounds,
  };

  // --check-truth 是独立校验器：只在此时读取真值，且不回流到 evidence/cause。
  let truthCheck = null;
  if (args['check-truth']) {
    const gt = dataset.meta.groundTruth;
    const Rinv = mTranspose(gt.rotation);
    const tinv = mApply(Rinv, gt.translation).map((x) => -x);
    let rotErr = 0;
    for (let i = 0; i < 9; i++)
      rotErr = Math.max(rotErr, Math.abs(result.transform.rotation[i] - Rinv[i]));
    truthCheck = {
      rotationMaxAbsError: rotErr,
      translationError: Math.hypot(...result.transform.translation.map((x, i) => x - tinv[i])),
      note: '真值仅用于离线校验，未参与求解或分类',
    };
  }

  return {
    scenario,
    seed,
    datasetConventions: {
      unit: dataset.meta.unit,
      handedness: dataset.meta.handedness,
      upAxis: dataset.meta.upAxis,
    },
    config,
    initialPerturbation: {
      angleDeg: args['perturb-angle'] != null ? requireNumber('--perturb-angle', args['perturb-angle']) : 0,
      axis: parseVec(args['perturb-axis'] || '0,1,0', '--perturb-axis'),
      translation: parseVec(args['perturb-translation'] || '0,0,0', '--perturb-translation'),
    },
    evidence,
    replay,
    truthCheck,
  };
}

// --evidence <file> --replay：对导出 JSON 做离线重放核对，证明证据自洽可重放。
function replayEvidenceFile(file, io = console) {
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  const evidence = raw.evidence || raw;
  if (!evidence || !Array.isArray(evidence.rounds)) throw new Error('文件不含迭代证据 rounds');
  const scenario = raw.scenario || evidence.scenario;
  if (!scenario) throw new Error('重放需要导出文件中的 scenario 字段以重建点云');
  const seed = raw.seed != null ? raw.seed : (raw.meta && raw.meta.seed);
  if (seed == null) throw new Error('重放需要 seed 以重建同一程序化点云');
  const dataset = generateDataset(scenario, seed);
  const rounds = evidence.rounds.map((r) => {
    const check = diag.replayRound(
      dataset.source.cloud.points,
      dataset.target.cloud.points,
      {
        inputPose: r.inputPose,
        maxDistance: evidence.config.overlapDistance * 2,
        huberK: evidence.config.huberK,
        fingerprint: r.inputFingerprint,
      },
    );
    return {
      iteration: r.iteration,
      fingerprint: r.inputFingerprint,
      replayFingerprint: check.replayFingerprint,
      match: check.match,
      matchedCount: check.matchedCount,
      residual: check.residual,
    };
  });
  return {
    file,
    stable: rounds.every((r) => r.match),
    rounds,
    recomputedEvidenceFingerprint: diag.evidenceFingerprint(evidence),
    storedEvidenceFingerprint: evidence.fingerprint || null,
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log('用法: node scripts/diagnose.js --scenario <bowl|plate|partial|mirrored|wrongpairs|bigtwist> [--seed N]');
    console.log('  --perturb-angle DEG --perturb-axis X,Y,Z --perturb-translation X,Y,Z');
    console.log('  --inlier-threshold MM --overlap-distance MM --max-distance MM --max-iterations N --no-huber');
    console.log('  --show-rounds --json --strict --check-truth');
    console.log('  --evidence FILE --replay');
    return;
  }
  if (args.evidence) {
    if (!args.replay) throw new Error('--evidence 需要配合 --replay');
    const out = replayEvidenceFile(args.evidence);
    console.log(JSON.stringify(out, null, 2));
    if (!out.stable && args.strict) process.exitCode = 2;
    return;
  }
  const report = runDiagnosis(args);
  if (args.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(renderText(report));
  }
  if (args.strict) {
    const ok = report.evidence.fingerprint && report.replay.stable;
    if (!ok) process.exitCode = 2;
  }
}

if (require.main === module) {
  try {
    main();
  } catch (e) {
    console.error(`诊断失败：${e.message}`);
    process.exitCode = 1;
  }
}

module.exports = {
  parseArgs,
  deriveConfig,
  runDiagnosis,
  replayEvidenceFile,
  renderText,
  initialPoseFromDataset,
  perturbPose,
  DEFAULT_SCENE_PARAMS,
};

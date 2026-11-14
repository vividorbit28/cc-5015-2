// 配准诊断单元测试：证据链、复跑一致性、根因分类、日志预算与“不读真值”约束。
// 运行：node tests/diagnostics.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const diag = require('../server/diagnostics');
const reg = require('../server/registration');
const cli = require('../scripts/diagnose');
const { generateDataset } = require('../server/synthdata');
const { weightedKabsch, mMul, applyTransform } = require('../server/linalg');
const { axisAngleMatrix } = require('../server/synthdata');

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

function run(scenario, extra = {}) {
  return cli.runDiagnosis({ scenario, seed: 42, ...extra });
}

// 构造一份“内核轮次 + 输入观测”，不依赖真值，用于分类分支的单测。
function classify(overrides = {}) {
  return diag.classifyCause({
    correspondence: {
      threshold: 4, inlierCount: 8, outlierCount: 0, inlierRatio: 1,
      items: new Array(8).fill(null), residual: { median: 0.2 },
      bimodal: false, maxGap: 0.1, gapAfter: 0,
    },
    rounds: [{
      iteration: 0,
      inputPose: { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] },
      matchedCount: 8, unmatchedCount: 0,
      residual: { mean: 6 },
      step: { accepted: false, meanBefore: 6, meanAfter: 6.002 },
    }],
    stopCode: diag.STOP.STAGNATED,
    noiseFloor: { median: 1.5, available: true },
    initialOverlap: 0.9,
    finalOverlap: 0.9,
    targetCoverage: 0.9,
    scaleRatios: { available: true, uniform: true, median: 1, iqr: 0.01, deviatesFromUnit: false },
    degeneracy: { rank: 3, planarRatio: 0.2 },
    symmetryAlternatives: [],
    rigidConsistency: {
      mutuallyConsistent: true, bestInlierCount: 8, sampleCount: 8,
      outlierIndices: [], reflectionDominant: false,
    },
    inputPoseInliers: 8,
    reachableOverlap: 1,
    ...overrides,
  });
}

console.log('== 1. 输入校验：不允许默认值遮错 ==');
check('缺少 maxDistance 时 icpRefine 直接报错而非静默取默认', () => {
  const ds = generateDataset('bowl', 42);
  let err = null;
  try {
    reg.icpRefine(ds.source.cloud, ds.target.cloud,
      { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] }, {});
  } catch (e) { err = e; }
  assert.ok(err, '应抛出错误');
  assert.strictEqual(err.code, 'DIAG_INPUT_MISSING');
});

check('replayRound 缺 inputPose 时报 DIAG_INPUT_MISSING', () => {
  let err = null;
  try {
    diag.replayRound([[0, 0, 0]], [[0, 0, 0]], { maxDistance: 1 });
  } catch (e) { err = e; }
  assert.ok(err && err.code === 'DIAG_INPUT_MISSING');
});

check('requireFiniteNumber 拒绝 NaN/字符串', () => {
  assert.throws(() => diag.requireFiniteNumber('x', NaN), /有限数值/);
  assert.throws(() => diag.requireFiniteNumber('x', '3'), /有限数值/);
  assert.strictEqual(diag.requireFiniteNumber('x', 3), 3);
});

console.log('== 2. 迭代证据链与复跑一致性 ==');
check('同一输入两次诊断：证据指纹相同、逐轮重放一致', () => {
  const a = run('partial', { 'perturb-angle': 20, 'perturb-axis': '0,1,0', 'perturb-translation': '12,0,0' });
  const b = run('partial', { 'perturb-angle': 20, 'perturb-axis': '0,1,0', 'perturb-translation': '12,0,0' });
  assert.strictEqual(a.evidence.fingerprint, b.evidence.fingerprint);
  assert.ok(a.replay.stable);
  for (const r of a.replay.rounds) assert.strictEqual(r.fingerprint, r.replayFingerprint);
});

check('证据含逐轮输入位姿、有效对应、残差与位姿更新结论', () => {
  const r = run('bowl', { 'perturb-angle': 8 });
  assert.ok(r.evidence.rounds.length >= 1);
  for (const round of r.evidence.rounds) {
    assert.ok(Array.isArray(round.inputPose.rotation) && round.inputPose.rotation.length === 9);
    assert.ok(Number.isInteger(round.matchedCount) && Number.isInteger(round.unmatchedCount));
    assert.ok(round.residual && Number.isFinite(round.residual.mean));
    assert.ok(typeof round.inputFingerprint === 'string' && round.inputFingerprint.length === 16);
  }
});

check('残差下降与有效对应趋势可对应到具体轮次', () => {
  const r = run('bowl', { 'perturb-angle': 8 });
  assert.ok(Array.isArray(r.evidence.residualTrend));
  assert.ok(Array.isArray(r.evidence.validCorrespondenceTrend));
  for (const t of r.evidence.residualTrend) assert.ok(Number.isFinite(t.mean));
  for (const t of r.evidence.validCorrespondenceTrend) assert.ok('satisfiesMinimum' in t);
});

check('阶段定位把异常落到 input/correspondence/pose-update/stop 之一', () => {
  const allowed = new Set(Object.values(diag.STAGE));
  for (const scenario of ['bowl', 'mirrored', 'wrongpairs']) {
    const r = run(scenario);
    for (const f of r.evidence.stageFindings) assert.ok(allowed.has(f.stage), `未知阶段 ${f.stage}`);
  }
});

console.log('== 3. 根因分类 ==');
check('干净数据：无根因（NONE）', () => {
  for (const s of ['bowl', 'plate', 'partial']) {
    assert.strictEqual(run(s).evidence.cause.primary, diag.CAUSE.NONE, `${s} 误报`);
  }
});

check('错误初始位姿：配对一致但初值重叠远低于可达重叠', () => {
  const r = run('bowl', { 'perturb-angle': 35, 'perturb-translation': '30,0,0' });
  assert.strictEqual(r.evidence.cause.primary, diag.CAUSE.WRONG_INITIAL_POSE);
  const ev = r.evidence.cause.evidence.find((e) => e.code === diag.CAUSE.WRONG_INITIAL_POSE);
  assert.ok(ev.metrics.reachableOverlap >= 0.5 && ev.metrics.initialOverlap < ev.metrics.reachableOverlap * 0.5);
});

check('对应错误：刚性一致性不足，主因为 CORRESPONDENCE_ERROR', () => {
  const r = run('wrongpairs');
  assert.strictEqual(r.evidence.cause.primary, diag.CAUSE.CORRESPONDENCE_ERROR);
  assert.ok(r.evidence.rigidConsistency.outlierIndices.length > 0);
});

check('镜像/左右手：反射一致性高于刚性一致性 → SYMMETRY_AMBIGUITY', () => {
  const r = run('mirrored');
  assert.strictEqual(r.evidence.cause.primary, diag.CAUSE.SYMMETRY_AMBIGUITY);
  assert.ok(r.evidence.rigidConsistency.reflectionDominant);
});

check('尺度/单位不一致：距离比稳定偏离 1 → SCALE_UNIT_MISMATCH', () => {
  for (const scale of [1.08, 0.6]) {
    const r = run('bowl', { 'source-scale': scale });
    assert.strictEqual(r.evidence.cause.primary, diag.CAUSE.SCALE_UNIT_MISMATCH, `scale=${scale}`);
    assert.ok(Math.abs(r.evidence.scaleRatios.median - 1 / scale) < 0.02);
  }
});

check('局部极小值：高重叠但残差平台高于噪声地板且配对一致 → LOCAL_MINIMUM', () => {
  const cause = classify();
  assert.strictEqual(cause.primary, diag.CAUSE.LOCAL_MINIMUM);
  assert.ok(cause.plateauFactor > diag.PLATEAU_NOISE_FACTOR);
});

check('平台贴近噪声地板时不误判局部极小值', () => {
  const cause = classify({
    rounds: [{
      iteration: 0,
      inputPose: { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] },
      matchedCount: 8, unmatchedCount: 0,
      residual: { mean: 1.6 },
      step: { accepted: false, meanBefore: 1.6, meanAfter: 1.61 },
    }],
  });
  assert.notStrictEqual(cause.primary, diag.CAUSE.LOCAL_MINIMUM);
});

check('局部极小值不叠加在错误初值/对应错误之上', () => {
  const wrongPose = classify({ initialOverlap: 0.1, finalOverlap: 0.1, reachableOverlap: 1 });
  assert.strictEqual(wrongPose.primary, diag.CAUSE.WRONG_INITIAL_POSE);
  assert.ok(!wrongPose.secondary.includes(diag.CAUSE.LOCAL_MINIMUM));
});

console.log('== 4. 版本与日志预算 ==');
check('日志预算：长轮次序列裁剪到 24 并保留首尾', () => {
  const rounds = Array.from({ length: 150 }, (_, i) => ({
    iteration: i,
    inputPose: { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] },
    matchedCount: 8 + (i % 2), unmatchedCount: i % 2,
    residual: { mean: 10 - i * 0.05 },
    step: { accepted: true, meanBefore: 10, meanAfter: 9.9 },
  }));
  const kept = diag.retainRounds(rounds, { maxRecords: 24, head: 4, tail: 8 });
  assert.strictEqual(kept.rounds.length, 24);
  assert.strictEqual(kept.omitted, 126);
  assert.strictEqual(kept.rounds[0].iteration, 0);
  assert.strictEqual(kept.rounds.at(-1).iteration, 149);
  assert.strictEqual(kept.rounds[3].iteration, 3);
});

check('证据指纹不含时间戳：重复计算不变', () => {
  const r = run('plate');
  const again = diag.evidenceFingerprint(r.evidence);
  assert.strictEqual(again, r.evidence.fingerprint);
});

console.log('== 5. 不读取测试真值 ==');
check('生产路径（诊断实现与内核）不读取 groundTruth', () => {
  for (const file of ['../server/diagnostics.js', '../server/registration.js']) {
    const code = fs.readFileSync(path.join(__dirname, file), 'utf8')
      .split('\n')
      .map((line) => line.replace(/\/\/.*$/, ''))
      .join('\n');
    assert.ok(!/\.groundTruth/.test(code), `${file} 在代码路径读取了 groundTruth`);
  }
});

check('命令行工具仅在 --check-truth 分支内访问真值', () => {
  const lines = fs.readFileSync(path.join(__dirname, '../scripts/diagnose.js'), 'utf8').split('\n');
  const idx = lines.findIndex((l) => /\.groundTruth/.test(l) && !/^\s*\/\//.test(l));
  assert.ok(idx >= 0, '未找到离线校验代码');
  const before = lines.slice(0, idx).join('\n');
  assert.ok(/if \(args\['check-truth'\]\)/.test(before.slice(-400)), '真值访问必须位于 check-truth 分支');
});

check('--check-truth 仅在离线校验字段出现，且不改变证据/根因', () => {
  const without = run('partial', { 'perturb-angle': 20 });
  const withTruth = run('partial', { 'perturb-angle': 20, 'check-truth': true });
  assert.strictEqual(without.evidence.fingerprint, withTruth.evidence.fingerprint);
  assert.deepStrictEqual(without.evidence.cause, withTruth.evidence.cause);
  assert.ok(withTruth.truthCheck && withTruth.truthCheck.note.includes('未参与'));
  assert.strictEqual(without.truthCheck, null);
});

console.log(`\n${process.exitCode ? '存在失败项' : `全部 ${pass} 项检查通过`}`);

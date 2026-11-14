// 零依赖 HTTP 服务：静态前端 + 配准工作台 JSON API。
// 会话（session）保存原始模型、用户对应点与全部方案，
// 任何变换结果另存，绝不覆盖原始点云/网格。

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {
  weightedKabsch, applyTransform, mApply, vsub, mDet,
} = require('./linalg');
const reg = require('./registration');
const diag = require('./diagnostics');
const { generateDataset } = require('./synthdata');

const PORT = process.env.PORT || 8080;
const ROOT = path.resolve(__dirname, '..');
const sessions = new Map();

function json(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > 64 * 1024 * 1024) reject(new Error('请求体过大'));
      else chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); }
      catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

function serveStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let rel = decodeURIComponent(url.pathname);
  if (rel === '/') rel = '/index.html';
  // 前端应用资源位于 public/，第三方运行时位于 vendor/
  const baseDir = rel.startsWith('/vendor/') ? ROOT : path.join(ROOT, 'public');
  const filePath = path.normalize(path.join(baseDir, rel));
  if (!filePath.startsWith(baseDir)) { res.writeHead(403); return res.end('forbidden'); }
  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(data);
  });
}

function newSessionId() {
  return crypto.randomBytes(9).toString('hex');
}

function publicSession(s) {
  return {
    sessionId: s.id,
    scenario: s.dataset.meta.scenario,
    meta: s.dataset.meta,
    sizes: {
      targetCloud: s.dataset.target.cloud.points.length,
      sourceCloud: s.dataset.source.cloud.points.length,
      targetFaces: s.dataset.target.mesh.faces.length,
      sourceFaces: s.dataset.source.mesh.faces.length,
    },
    correspondences: s.correspondences,
    schemes: s.schemes.map(publicScheme),
    confirmedSchemeId: s.confirmedSchemeId,
    confirmation: s.confirmation || null,
  };
}

function publicScheme(sc) {
  return {
    id: sc.id,
    label: sc.label,
    source: sc.source,
    transform: sc.transform,
    inlierIndices: sc.inlierIndices,
    outlierIndices: sc.outlierIndices,
    outlierReport: sc.outlierReport,
    rms: sc.rms,
    cloudRms: sc.cloudRms,
    cloudMedian: sc.cloudMedian,
    overlapFraction: sc.overlapFraction,
    targetCoverageFraction: sc.targetCoverageFraction,
    unmatchedSourceCount: sc.unmatchedSourceCount,
    diagnostics: sc.diagnostics,
    warnings: sc.warnings || [],
    icp: sc.icpSummary || null,
    evidence: sc.evidenceSummary || null,
    replay: sc.replaySummary || null,
    symmetry: sc.symmetry || null,
    pointResiduals: sc.pointResiduals,
    createdAt: sc.createdAt,
  };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const route = `${req.method} ${url.pathname}`;
  try {
    if (!url.pathname.startsWith('/api/')) return serveStatic(req, res);

    if (route === 'POST /api/datasets') {
      const body = await readBody(req);
      const scenario = ['bowl', 'plate', 'partial', 'mirrored', 'wrongpairs', 'bigtwist']
        .includes(body.scenario) ? body.scenario : 'bowl';
      const seed = Number.isInteger(body.seed) ? body.seed : Date.now() % 100000;
      const dataset = generateDataset(scenario, seed);
      const id = newSessionId();
      const session = {
        id,
        dataset,
        correspondences: dataset.landmarks.map((lm) => ({
          id: lm.id,
          targetPoint: lm.targetPoint,
          sourcePoint: lm.sourcePoint,
          targetWeight: lm.targetWeight,
          sourceWeight: lm.sourceWeight,
          note: '程序生成的候选对应点（可在界面对调/删除/调权）',
        })),
        schemes: [],
        confirmedSchemeId: null,
        confirmation: null,
      };
      sessions.set(id, session);
      return json(res, 200, publicSession(session));
    }

    const m = url.pathname.match(/^\/api\/sessions\/([a-f0-9]+)(\/.*)?$/);
    if (!m) return json(res, 404, { error: '未知接口' });
    const session = sessions.get(m[1]);
    if (!session) return json(res, 404, { error: '会话不存在或已过期' });
    const sub = m[2] || '';

    if (route.startsWith('GET /api/sessions/') && sub === '') {
      return json(res, 200, publicSession(session));
    }

    // 几何数据（点云/网格/变换后预览），数据量较大，单独接口
    if (route.startsWith('GET /api/sessions/') && sub === '/geometry') {
      const d = session.dataset;
      return json(res, 200, {
        unit: d.meta.unit, handedness: d.meta.handedness, upAxis: d.meta.upAxis,
        target: d.target,
        source: d.source,
      });
    }

    if (route.startsWith('PUT /api/sessions/') && sub === '/correspondences') {
      const body = await readBody(req);
      if (!Array.isArray(body.correspondences))
        return json(res, 400, { error: 'correspondences 必须为数组' });
      for (const c of body.correspondences) {
        if (!Array.isArray(c.targetPoint) || !Array.isArray(c.sourcePoint) ||
            c.targetPoint.length !== 3 || c.sourcePoint.length !== 3)
          return json(res, 400, { error: '对应点坐标必须是三维数组' });
        const tw = Number(c.targetWeight), sw = Number(c.sourceWeight);
        if (!(tw > 0) || !(sw > 0))
          return json(res, 400, { error: '误差权重必须为正数' });
      }
      session.correspondences = body.correspondences.map((c, i) => ({
        id: c.id ?? i,
        targetPoint: c.targetPoint.map(Number),
        sourcePoint: c.sourcePoint.map(Number),
        targetWeight: Number(c.targetWeight),
        sourceWeight: Number(c.sourceWeight),
        note: c.note || '用户对应点',
      }));
      // 输入变化后旧方案保留但标记失效，不自动删除
      return json(res, 200, publicSession(session));
    }

    if (route.startsWith('POST /api/sessions/') && sub === '/register') {
      const body = await readBody(req);
      const corr = session.correspondences;
      if (corr.length < 3)
        return json(res, 400, { error: `对应点仅 ${corr.length} 对，至少需要 3 对` });
      const sPts = corr.map((c) => c.sourcePoint);
      const tPts = corr.map((c) => c.targetPoint);
      const tW = corr.map((c) => c.targetWeight);
      const sW = corr.map((c) => c.sourceWeight);
      const cfg = diag.deriveSceneConfig({
        targetPoints: session.dataset.target.cloud.points,
        correspondencePoints: tPts.concat(sPts),
        overrides: {
          inlierThreshold: body.inlierThreshold,
          overlapDistance: body.overlapDistance,
        },
      });
      const inlierThreshold = cfg.inlierThreshold;
      const overlapDistance = cfg.overlapDistance;

      // 1) 全部点的加权 Kabsch（基线，含退化诊断）
      let baseline;
      try {
        const fit = weightedKabsch(sPts, tPts, { targetWeights: tW, sourceWeights: sW });
        baseline = {
          id: session.schemes.length,
          label: '全点加权 Kabsch 基线',
          source: 'kabsch-all',
          transform: { rotation: fit.rotation, translation: fit.translation },
          inlierIndices: corr.map((_, i) => i),
          outlierIndices: [],
          outlierReport: reg.explainOutliers(sPts, tPts,
            { rotation: fit.rotation, translation: fit.translation },
            corr.map((_, i) => i), inlierThreshold),
          rms: fit.rms,
          diagnostics: {
            conditionNumber: fit.polar.conditionNumber,
            determinant: fit.polar.determinant,
            degeneracy: fit.degeneracy,
            rotationAngleDeg: fit.rotationAngleDeg,
            rotationAxis: fit.rotationAxis,
          },
          warnings: fit.mirrorSuspected
            ? ['反射（镜像）拟合残差显著更小：数据疑似左右手翻转，刚性配准结论不可靠，需人工核查']
            : [],
          mirrorSuspected: fit.mirrorSuspected,
          reflectionFitRms: fit.reflectionFit ? fit.reflectionFit.rms : null,
          createdAt: new Date().toISOString(),
        };
      } catch (e) {
        return json(res, 400, { error: `Kabsch 求解失败：${e.message}` });
      }

      // 2) RANSAC 多候选
      const rng = mulberryFromSeed(body.seed || 1234);
      const ransac = reg.ransacCandidates(sPts, tPts, {
        iterations: body.iterations || 400,
        inlierThreshold,
        weights: corr.map((c) => (c.targetWeight * c.sourceWeight) /
          (c.targetWeight + c.sourceWeight)),
        rng,
      });

      const schemes = [baseline];
      // 先在点云层面评估全部 RANSAC 候选，再按“内点数 + 重叠率”综合排序，
      // 避免仅因对应点内点多就把几何上不重叠的退化解排在前面
      const evaluatedRansac = ransac.map((c) => {
        const ev = reg.evaluateCandidate(c,
          session.dataset.source.cloud,
          session.dataset.target.cloud, overlapDistance);
        return Object.assign(c, {
          cloudRms: ev.cloudRms,
          cloudMedian: ev.cloudMedian,
          overlapFraction: ev.overlapFraction,
          targetCoverageFraction: ev.targetCoverageFraction,
          unmatchedSourceCount: ev.unmatchedSourceCount,
          pointResiduals: ev.pointResiduals,
          warnings: ev.warnings,
        });
      });
      evaluatedRansac.sort((a, b) =>
        (b.inlierCount * 2 + b.overlapFraction * 10) -
        (a.inlierCount * 2 + a.overlapFraction * 10));
      for (const c of evaluatedRansac.slice(0, 4)) {
        c.label = `RANSAC 候选 ${schemes.length}（内点 ${c.inlierCount}/${corr.length}，重叠 ${(c.overlapFraction * 100).toFixed(0)}%）`;
        c.outlierReport = reg.explainOutliers(sPts, tPts, c.transform,
          c.inlierIndices, inlierThreshold);
        c.createdAt = new Date().toISOString();
        schemes.push(c);
      }

      // 3) 对称备选：以最佳 RANSAC 候选为底，绕 PCA 主轴翻转 180°
      if (evaluatedRansac.length) {
        const best = evaluatedRansac[0];
        const axes = symmetryAxes(session.dataset.target.cloud.points);
        const alts = reg.symmetryAlternatives(best,
          session.dataset.source.cloud,
          session.dataset.target.cloud,
          { symmetryAxes: axes, overlapDistance });
        for (const a of alts) {
          a.id = schemes.length;
          a.inlierIndices = best.inlierIndices;
          a.outlierIndices = best.outlierIndices;
          a.outlierReport = best.outlierReport;
          a.diagnostics = best.diagnostics;
          a.rms = best.rms;
          a.warnings = ['对称/近轴对称形状产生的等残差候选，需人工依据纹饰/断面选择', ...(a.warnings || [])];
          a.createdAt = new Date().toISOString();
          schemes.push(a);
        }
      }

      // 4) 点云层面评估基线与对称候选（RANSAC 候选已评估）
      for (const sc of schemes) {
        if (sc.pointResiduals) continue;
        const ev = reg.evaluateCandidate(sc,
          session.dataset.source.cloud,
          session.dataset.target.cloud, overlapDistance);
        Object.assign(sc, {
          cloudRms: ev.cloudRms,
          cloudMedian: ev.cloudMedian,
          overlapFraction: ev.overlapFraction,
          targetCoverageFraction: ev.targetCoverageFraction,
          unmatchedSourceCount: ev.unmatchedSourceCount,
          pointResiduals: ev.pointResiduals,
          warnings: Array.from(new Set([...(sc.warnings || []), ...ev.warnings])),
        });
        if (mDet(sc.transform.rotation) <= 0)
          sc.warnings.push('变换行列式非 +1：包含反射，不是合法刚性变换（可能为镜像碎片）');
      }
      schemes.forEach((sc, i) => { sc.id = i; });
      session.schemes = schemes;
      session.lastParams = { inlierThreshold, overlapDistance };
      return json(res, 200, {
        ...publicSession(session),
        params: session.lastParams,
      });
    }

    if (route.startsWith('POST /api/sessions/') && sub.startsWith('/schemes/') && sub.endsWith('/icp')) {
      const sid = Number(sub.split('/')[2]);
      const scheme = session.schemes[sid];
      if (!scheme) return json(res, 404, { error: '方案不存在' });
      const body = await readBody(req);
      let cfg;
      try {
        cfg = diag.deriveSceneConfig({
          targetPoints: session.dataset.target.cloud.points,
          correspondencePoints: session.correspondences
            .flatMap((c) => [c.targetPoint, c.sourcePoint]),
          overrides: {
            inlierThreshold: body.inlierThreshold ?? (session.lastParams && session.lastParams.inlierThreshold),
            overlapDistance: body.overlapDistance ?? (session.lastParams && session.lastParams.overlapDistance),
            maxDistance: body.maxDistance,
            maxIterations: body.maxIterations,
            huber: body.huber,
            huberK: body.huberK,
          },
        });
      } catch (e) {
        return json(res, 400, { error: `配准参数无效：${e.message}` });
      }
      const { overlapDistance, inlierThreshold, maxDistance, huberK, maxIterations } = cfg;
      const result = reg.icpRefine(
        session.dataset.source.cloud,
        session.dataset.target.cloud,
        scheme.transform,
        { maxIterations, tolerance: cfg.tolerance, maxDistance, huberK },
      );
      // ICP 结果另存为新方案，不覆盖原方案与原始模型
      const moved = applyTransform(session.dataset.source.cloud.points, result.transform);
      const stats = reg.cloudOverlapStats(moved,
        session.dataset.target.cloud.points,
        overlapDistance * 2);

      // 迭代证据：输入/对应/权重/每轮位姿更新与停止条件，可与命令行复跑对账
      let evidence = null;
      let evidenceError = null;
      try {
        evidence = diag.buildIcpEvidence({
          sourceCloud: session.dataset.source.cloud,
          targetCloud: session.dataset.target.cloud,
          correspondence: session.correspondences,
          initialPose: scheme.transform,
          icpResult: result,
          inlierThreshold,
          overlapDistance,
          maxIterations,
          huberK,
          method: 'icp',
        });
      } catch (e) {
        evidenceError = e.message;
      }
      const replayRecords = evidence ? evidence.rounds.map((r) => {
        const snap = {
          iteration: r.iteration,
          inputPose: r.inputPose,
          maxDistance,
          huberK,
          fingerprint: r.inputFingerprint,
        };
        const check = diag.replayRound(
          session.dataset.source.cloud.points,
          session.dataset.target.cloud.points,
          snap,
        );
        return {
          iteration: r.iteration,
          fingerprint: snap.fingerprint,
          replayFingerprint: check.replayFingerprint,
          match: check.match,
          matchedCount: check.matchedCount,
          residual: check.residual,
        };
      }) : [];
      const replayStable = replayRecords.every((r) => r.match);

      const refined = {
        id: session.schemes.length,
        label: `${scheme.label} → ICP 局部优化`,
        source: 'icp',
        transform: result.transform,
        inlierIndices: scheme.inlierIndices,
        outlierIndices: scheme.outlierIndices,
        outlierReport: scheme.outlierReport,
        rms: scheme.rms,
        diagnostics: scheme.diagnostics,
        icpSummary: {
          converged: result.converged,
          stopCode: result.stopCode,
          iterations: result.iterations,
          reason: result.reason || '收敛',
          history: result.history,
          rejectedPairs: result.rejected.slice(0, 20),
          rejectedCount: result.rejected.length,
          pairCount: result.pairs.length || result.history.at(-1)?.pairCount || 0,
          noiseFloor: result.noiseFloor,
          cause: evidence ? evidence.cause : null,
          stageFindings: evidence ? evidence.stageFindings : null,
          residualTrend: evidence ? evidence.residualTrend : null,
          validCorrespondenceTrend: evidence ? evidence.validCorrespondenceTrend : null,
        },
        evidenceSummary: evidence ? {
          version: evidence.version,
          fingerprint: evidence.fingerprint,
          stop: evidence.stop,
          cause: evidence.cause,
          coverage: evidence.coverage,
          correspondenceInitial: evidence.correspondenceInitial,
          correspondenceFinal: evidence.correspondenceFinal,
          noiseFloor: evidence.noiseFloor,
          scaleRatios: evidence.scaleRatios,
          roundBudget: evidence.roundBudget,
          stageFindings: evidence.stageFindings,
        } : { error: evidenceError },
        replaySummary: {
          stable: replayStable,
          rounds: replayRecords,
          note: replayStable
            ? '每轮输入位姿的独立重放指纹与本轮记录一致'
            : '存在重放指纹不一致的轮次，证据不可信',
        },
        evidenceFull: evidence,
        warnings: result.converged ? [] : [`ICP 未收敛：${result.reason}`],
        createdAt: new Date().toISOString(),
        ...stats,
      };
      if (mDet(refined.transform.rotation) <= 0)
        refined.warnings.push('ICP 结果含反射分量，已标记为非法刚性变换');
      if (evidence && evidence.cause && evidence.cause.primary !== diag.CAUSE.NONE) {
        refined.warnings.push(`诊断主因：${evidence.cause.primary}`);
      }
      if (evidenceError) refined.warnings.push(`诊断证据生成失败：${evidenceError}`);
      if (!replayStable) refined.warnings.push('迭代证据的逐轮重放不一致');
      session.schemes.push(refined);
      return json(res, 200, publicSession(session));
    }

    if (route.startsWith('POST /api/sessions/') && sub.startsWith('/schemes/') && sub.endsWith('/confirm')) {
      const sid = Number(sub.split('/')[2]);
      const scheme = session.schemes[sid];
      if (!scheme) return json(res, 404, { error: '方案不存在' });
      const body = await readBody(req);
      if (!body.operator || !body.decision)
        return json(res, 400, { error: '确认需提供 operator 与 decision' });
      // 绑定实际输入快照与参数，防止事后无法复算
      session.confirmedSchemeId = sid;
      session.confirmation = {
        schemeId: sid,
        operator: body.operator,
        decision: body.decision, // accepted | rejected | undecided
        note: body.note || '',
        correspondenceSnapshot: JSON.parse(JSON.stringify(session.correspondences)),
        params: session.lastParams || null,
        transformSnapshot: JSON.parse(JSON.stringify(scheme.transform)),
        inputFingerprint: fingerprint(session),
        timestamp: new Date().toISOString(),
      };
      return json(res, 200, publicSession(session));
    }

    // 诊断预览：给定任意初始位姿（界面拖动滑块或命令行），返回
    // 有效对应、对应原因、距离残差、覆盖与阶段定位。不修改会话状态、不生成方案，
    // 确保“拖动试配准”与“正式配准”共用同一内核，且预览失败能如实反馈。
    if (route.startsWith('POST /api/sessions/') && sub === '/diagnose') {
      const body = await readBody(req);
      const pose = body.pose;
      if (!pose || !Array.isArray(pose.rotation) || pose.rotation.length !== 9 ||
          !Array.isArray(pose.translation) || pose.translation.length !== 3 ||
          !pose.rotation.every(Number.isFinite) || !pose.translation.every(Number.isFinite)) {
        return json(res, 400, { error: 'pose 必须是 3x3 旋转与三维平移（有限数值）' });
      }
      if (session.correspondences.length === 0)
        return json(res, 400, { error: '没有对应点，无法给出对应原因' });
      // 预览与正式配准共用同一推导函数；返回参数来源，避免隐式默认值遮错
      let cfg;
      try {
        cfg = diag.deriveSceneConfig({
          targetPoints: session.dataset.target.cloud.points,
          correspondencePoints: session.correspondences
            .flatMap((c) => [c.targetPoint, c.sourcePoint]),
          overrides: {
            inlierThreshold: body.inlierThreshold,
            overlapDistance: body.overlapDistance,
            maxDistance: body.maxDistance,
            huber: body.huber,
            huberK: body.huberK,
          },
        });
      } catch (e) {
        return json(res, 400, { error: `诊断参数无效：${e.message}` });
      }
      const { inlierThreshold, overlapDistance, maxDistance, huberK } = cfg;
      const scaleRatios = diag.correspondenceDistanceRatios(
        session.correspondences.map((c) => c.sourcePoint),
        session.correspondences.map((c) => c.targetPoint),
      );
      const moved = applyTransform(session.dataset.source.cloud.points, pose);
      const coverage = diag.coverageStats(moved,
        session.dataset.target.cloud.points, overlapDistance);
      const pairReport = diag.nearestCorrespondenceStats(moved,
        session.dataset.target.cloud.points, maxDistance, huberK);
      const corrReport = diag.correspondenceReport(
        session.correspondences.map((c) => c.sourcePoint),
        session.correspondences.map((c) => c.targetPoint),
        session.correspondences.map((c) => {
          const denom = c.targetWeight + c.sourceWeight;
          return denom > 1e-15 ? (c.targetWeight * c.sourceWeight) / denom : 0;
        }),
        pose, inlierThreshold,
      );
      const degeneracy = body.degeneracy || null;
      // 拖动预览也必须给出与位姿无关的配对一致性，否则会把“初值错”误报成 NONE/低重叠。
      const corrWeights = session.correspondences.map((c) => {
        const denom = c.targetWeight + c.sourceWeight;
        return denom > 1e-15 ? (c.targetWeight * c.sourceWeight) / denom : 0;
      });
      const rigidConsistency = diag.rigidConsistencyCheck(
        session.correspondences.map((c) => c.sourcePoint),
        session.correspondences.map((c) => c.targetPoint),
        corrWeights,
        inlierThreshold,
      );
      let reachableOverlap = null;
      if (rigidConsistency.bestTransform) {
        const reachMoved = applyTransform(session.dataset.source.cloud.points,
          rigidConsistency.bestTransform);
        reachableOverlap = diag.coverageStats(reachMoved,
          session.dataset.target.cloud.points, overlapDistance).overlapFraction;
      }
      const cause = diag.classifyCause({
        correspondence: corrReport,
        rounds: [],
        stopCode: pairReport.satisfiesMinimum ? null : diag.STOP.INSUFFICIENT,
        noiseFloor: diag.samplingNoiseFloor(session.dataset.source.cloud.points),
        initialOverlap: coverage.overlapFraction,
        finalOverlap: coverage.overlapFraction,
        targetCoverage: coverage.targetCoverageFraction,
        scaleRatios,
        degeneracy,
        symmetryAlternatives: [],
        rigidConsistency,
        inputPoseInliers: corrReport.inlierCount,
        reachableOverlap,
      });
      // 预览对应原因按源点云逐点裁剪，限制日志成本
      const sampleStride = Math.max(1, Math.ceil(pairReport.pairs.length / 400));
      const pairSamples = pairReport.pairs
        .filter((_, i) => i % sampleStride === 0)
        .slice(0, 400);
      return json(res, 200, {
        params: { inlierThreshold, overlapDistance, maxDistance, huberK },
        paramsDerivedFrom: cfg.derivedFrom,
        coverage: {
          overlapFraction: coverage.overlapFraction,
          targetCoverageFraction: coverage.targetCoverageFraction,
          unmatchedSourceCount: coverage.unmatchedSourceCount,
          cloudRms: coverage.cloudRms,
          cloudMedian: coverage.cloudMedian,
          cloudMax: coverage.cloudMax,
          // 与源点云同索引：null 表示搜索半径内无目标点（碎片未重叠）
          pointResiduals: coverage.pointResiduals,
        },
        validCorrespondence: {
          matchedCount: pairReport.matchedCount,
          unmatchedCount: pairReport.unmatchedCount,
          minRequired: pairReport.minRequired,
          satisfiesMinimum: pairReport.satisfiesMinimum,
          residual: pairReport.residual,
          weightSum: pairReport.weightSum,
        },
        pairSamples,
        correspondencePoints: {
          threshold: corrReport.threshold,
          inlierCount: corrReport.inlierCount,
          outlierCount: corrReport.outlierCount,
          inlierRatio: corrReport.inlierRatio,
          residual: corrReport.residual,
          bimodal: corrReport.bimodal,
          items: corrReport.items,
        },
        scaleRatios,
        cause,
      });
    }

    // 证据导出/重放核对：返回完整迭代证据、逐轮重放结果与一致性结论
    if (route.startsWith('GET /api/sessions/') && sub.startsWith('/schemes/') && sub.endsWith('/evidence')) {
      const sid = Number(sub.split('/')[2]);
      const scheme = session.schemes[sid];
      if (!scheme) return json(res, 404, { error: '方案不存在' });
      if (!scheme.evidenceFull)
        return json(res, 409, { error: '该方案没有迭代证据（仅 ICP 方案产生逐轮证据）' });
      const evidence = scheme.evidenceFull;
      const replay = evidence.rounds.map((r) => diag.replayRound(
        session.dataset.source.cloud.points,
        session.dataset.target.cloud.points,
        {
          inputPose: r.inputPose,
          maxDistance: evidence.config.overlapDistance * 2,
          huberK: evidence.config.huberK,
          fingerprint: r.inputFingerprint,
        },
      ).match);
      return json(res, 200, {
        schemeId: sid,
        evidence,
        replay,
        replayStable: replay.every(Boolean),
      });
    }

    if (route.startsWith('GET /api/sessions/') && sub.startsWith('/schemes/') && sub.endsWith('/export')) {
      const sid = Number(sub.split('/')[2]);
      const scheme = session.schemes[sid];
      if (!scheme) return json(res, 404, { error: '方案不存在' });
      const d = session.dataset;
      const payload = {
        exportVersion: 2,
        exportedAt: new Date().toISOString(),
        conventions: {
          unit: d.meta.unit, handedness: d.meta.handedness, upAxis: d.meta.upAxis,
          transformDirection: 'transformed_source = R * source + t（无缩放）',
          evidenceScope: '迭代证据只含点云/对应点/权重/位姿与停止条件，不含测试真值',
        },
        scenario: d.meta.scenario,
        groundTruth: d.meta.groundTruth,
        correspondences: session.correspondences,
        scheme: publicScheme(scheme),
        evidence: scheme.evidenceFull || null,
        replay: scheme.replaySummary || null,
        confirmation: session.confirmation,
        transformedSourceCloud: {
          points: applyTransform(d.source.cloud.points, scheme.transform),
          normals: d.source.cloud.normals.map((n) => mApply(scheme.transform.rotation, n)),
        },
        originalSourceCloud: d.source.cloud,
        targetCloud: d.target.cloud,
      };
      const body = JSON.stringify(payload, null, 2);
      res.writeHead(200, {
        'Content-Type': 'application/json; charset=utf-8',
        'Content-Disposition': `attachment; filename="registration_${session.id}_scheme${sid}.json"`,
        'Cache-Control': 'no-store',
      });
      return res.end(body);
    }

    return json(res, 404, { error: '未知会话子接口' });
  } catch (e) {
    return json(res, 500, { error: e.message, stack: process.env.NODE_ENV === 'dev' ? e.stack : undefined });
  }
});

function mulberryFromSeed(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 目标点云 PCA 主轴，用于构造 180° 对称候选
function symmetryAxes(points) {
  const c = [0, 0, 0];
  for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k];
  for (let k = 0; k < 3; k++) c[k] /= points.length;
  const cov = new Array(9).fill(0);
  for (const p of points) {
    const d = vsub(p, c);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) cov[3 * i + j] += d[i] * d[j];
  }
  // 复用 linalg 的对称特征分解
  const { symmetricEigen3 } = require('./linalg');
  const eig = symmetricEigen3(cov);
  const ratios = [
    eig.values[0] / eig.values[2],
    eig.values[1] / eig.values[2],
  ];
  const axes = [];
  // 最大特征向量为主轴；若两个特征值接近，说明横向也近对称
  axes.push({ axis: eig.vectors[2], label: '绕 PCA 主轴 180° 对称备选' });
  if (ratios[1] > 0.85)
    axes.push({ axis: eig.vectors[1], label: '绕 PCA 次轴 180° 对称备选（形状近轴对称）' });
  return axes;
}

function fingerprint(session) {
  const h = crypto.createHash('sha256');
  h.update(JSON.stringify({
    c: session.correspondences,
    n: session.dataset.target.cloud.points.length + session.dataset.source.cloud.points.length,
    seed: session.dataset.meta.seed,
  }));
  return h.digest('hex').slice(0, 16);
}

// 启动前确认前端运行时依赖存在（vendor 由 scripts/fetch-vendor.js 获取）
for (const f of ['three.module.js', 'OrbitControls.js']) {
  if (!fs.existsSync(path.join(ROOT, 'vendor', f))) {
    console.error(`缺少 vendor/${f}，请先运行：node scripts/fetch-vendor.js`);
    process.exit(1);
  }
}

server.listen(PORT, () => {
  console.log(`博物馆数字碎片配准工作台: http://localhost:${PORT}`);
});

module.exports = { server };

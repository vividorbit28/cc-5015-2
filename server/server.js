// 零依赖 HTTP 服务：静态前端 + 配准工作台 JSON API。
// 会话（session）保存原始模型、用户对应点与全部方案，
// 任何变换结果另存，绝不覆盖原始点云/网格。

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {
  weightedKabsch, applyTransform, mApply, vadd, vsub, vdist, mDet,
} = require('./linalg');
const reg = require('./registration');
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

// 对角尺寸，用于自动阈值
function diagonal(points) {
  const mn = [Infinity, Infinity, Infinity];
  const mx = [-Infinity, -Infinity, -Infinity];
  for (const p of points)
    for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
  return Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
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
    unmatchedSourceCount: sc.unmatchedSourceCount,
    diagnostics: sc.diagnostics,
    warnings: sc.warnings || [],
    icp: sc.icpSummary || null,
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
      const diag = diagonal(tPts.concat(sPts));
      const inlierThreshold = Number.isFinite(body.inlierThreshold)
        ? body.inlierThreshold : diag * 0.02;
      const overlapDistance = Number.isFinite(body.overlapDistance)
        ? body.overlapDistance : diagonal(session.dataset.target.cloud.points) * 0.02;

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
      const overlapDistance = session.lastParams
        ? session.lastParams.overlapDistance
        : diagonal(session.dataset.target.cloud.points) * 0.02;
      const result = reg.icpRefine(
        session.dataset.source.cloud,
        session.dataset.target.cloud,
        scheme.transform,
        {
          maxIterations: body.maxIterations || 40,
          tolerance: 1e-8,
          maxDistance: Number.isFinite(body.maxDistance) ? body.maxDistance : overlapDistance * 2,
          huberK: body.huber === false ? null : overlapDistance,
        },
      );
      // ICP 结果另存为新方案，不覆盖原方案与原始模型
      const moved = applyTransform(session.dataset.source.cloud.points, result.transform);
      const stats = reg.cloudOverlapStats(moved,
        session.dataset.target.cloud.points,
        overlapDistance * 2);
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
          iterations: result.iterations,
          reason: result.reason || '收敛',
          history: result.history,
          rejectedPairs: result.rejected.slice(0, 20),
          rejectedCount: result.rejected.length,
          pairCount: result.pairs.length || result.history.at(-1)?.pairCount || 0,
        },
        warnings: result.converged ? [] : [`ICP 未完全收敛：${result.reason}`],
        createdAt: new Date().toISOString(),
        ...stats,
      };
      if (mDet(refined.transform.rotation) <= 0)
        refined.warnings.push('ICP 结果含反射分量，已标记为非法刚性变换');
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

    if (route.startsWith('GET /api/sessions/') && sub.startsWith('/schemes/') && sub.endsWith('/export')) {
      const sid = Number(sub.split('/')[2]);
      const scheme = session.schemes[sid];
      if (!scheme) return json(res, 404, { error: '方案不存在' });
      const d = session.dataset;
      const payload = {
        exportVersion: 1,
        exportedAt: new Date().toISOString(),
        conventions: {
          unit: d.meta.unit, handedness: d.meta.handedness, upAxis: d.meta.upAxis,
          transformDirection: 'transformed_source = R * source + t（无缩放）',
        },
        scenario: d.meta.scenario,
        groundTruth: d.meta.groundTruth,
        correspondences: session.correspondences,
        scheme: publicScheme(scheme),
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

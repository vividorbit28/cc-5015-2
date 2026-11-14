// 程序化生成博物馆碎片测试数据。
// 坐标约定：右手系，Y 轴向上，单位为毫米（mm）。
// 生成两类器物表面：碗（旋转曲面碎片）与板（带厚度的断板），
// 输出三角网格（顶点+法向）或点云，并记录生成参数用于复算。

'use strict';

const { vsub, vcross, vnormalize, mApply, vadd } = require('./linalg');

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pushNormal(vertices, faces) {
  const normals = new Array(vertices.length).fill(null).map(() => [0, 0, 0]);
  for (const [a, b, c] of faces) {
    const n = vcross(vsub(vertices[b], vertices[a]), vsub(vertices[c], vertices[a]));
    const nn = vnormalize(n);
    if (!nn) continue;
    for (const idx of [a, b, c])
      for (let k = 0; k < 3; k++) normals[idx][k] += nn[k];
  }
  return normals.map((n) => vnormalize(n) || [0, 1, 0]);
}

// 碗形旋转曲面：r(y) 轮廓 + 角度区间（碎片即不完整的角度范围）
function makeBowl(params, rng) {
  const {
    height = 120, radius = 90, thickness = 6,
    thetaStart = 0, thetaEnd = Math.PI * 1.2, // 弧面碎片
    nTheta = 48, nY = 24, jitter = 1.2,
    missingPatch = null, // { centerY, centerTheta, radiusY, radiusTheta }
  } = params;

  const profile = (y) => {
    const t = y / height;
    return radius * (0.35 + 0.65 * Math.pow(t, 0.8));
  };

  const vertices = [];
  const faces = [];
  const grid = [];
  for (let iy = 0; iy <= nY; iy++) {
    const row = [];
    const y = (iy / nY) * height;
    for (let it = 0; it <= nTheta; it++) {
      const theta = thetaStart + (it / nTheta) * (thetaEnd - thetaStart);
      let r = profile(y);
      const j = jitter * (rng() - 0.5);
      r += j;
      let x = r * Math.cos(theta);
      let z = r * Math.sin(theta);
      let keep = true;
      if (missingPatch) {
        const dy = y - missingPatch.centerY;
        let dt = Math.abs(theta - missingPatch.centerTheta);
        dt = Math.min(dt, Math.abs(theta - missingPatch.centerTheta - Math.PI * 2));
        const e = (dy / missingPatch.radiusY) ** 2 + (dt / missingPatch.radiusTheta) ** 2;
        if (e < 1) keep = false;
      }
      if (keep) {
        const idx = vertices.length;
        vertices.push([x, y, z]);
        row.push(idx);
      } else {
        row.push(-1);
      }
    }
    grid.push(row);
  }
  for (let iy = 0; iy < nY; iy++) {
    for (let it = 0; it < nTheta; it++) {
      const a = grid[iy][it], b = grid[iy][it + 1];
      const c = grid[iy + 1][it], d = grid[iy + 1][it + 1];
      if (a >= 0 && b >= 0 && c >= 0) faces.push([a, c, b]);
      if (b >= 0 && c >= 0 && d >= 0) faces.push([b, c, d]);
    }
  }
  const normals = pushNormal(vertices, faces);
  return { vertices, faces, normals };
}

// 平板碎片：带不规则断裂边界的长方体表面
function makePlate(params, rng) {
  const { width = 160, depth = 110, thick = 10, nX = 24, nZ = 18, jitter = 1.5,
    cornerCut = 0.22, missingPatch = null } = params;
  const vertices = [];
  const faces = [];
  const grid = [];
  for (let iz = 0; iz <= nZ; iz++) {
    const row = [];
    const z = (iz / nZ - 0.5) * depth;
    for (let ix = 0; ix <= nX; ix++) {
      const x = (ix / nX - 0.5) * width;
      // 不规则断裂边界：切掉一个角
      const edge = Math.min(ix / nX, 1 - ix / nX, iz / nZ, 1 - iz / nZ);
      const cutNoise = cornerCut * (0.6 + 0.8 * rng());
      let keep = edge > cutNoise * 0.25;
      if (ix < nX * cornerCut && iz < nZ * cornerCut) {
        if ((ix / (nX * cornerCut)) ** 2 + (iz / (nZ * cornerCut)) ** 2 <
          (0.7 + 0.6 * rng()) ** 2 * cutNoise * 4) keep = false;
      }
      if (missingPatch) {
        const e = ((x - missingPatch.centerX) / missingPatch.radiusX) ** 2 +
          ((z - missingPatch.centerZ) / missingPatch.radiusZ) ** 2;
        if (e < 1) keep = false;
      }
      if (keep) {
        const y = jitter * (rng() - 0.5);
        row.push(vertices.length);
        vertices.push([x, y, z]);
      } else row.push(-1);
    }
    grid.push(row);
  }
  for (let iz = 0; iz < nZ; iz++) {
    for (let ix = 0; ix < nX; ix++) {
      const a = grid[iz][ix], b = grid[iz][ix + 1];
      const c = grid[iz + 1][ix], d = grid[iz + 1][ix + 1];
      if (a >= 0 && b >= 0 && c >= 0) faces.push([a, b, c]);
      if (b >= 0 && d >= 0 && c >= 0) faces.push([b, d, c]);
    }
  }
  const normals = pushNormal(vertices, faces);
  return { vertices, faces, normals };
}

// 三角网格面采样点云
function sampleCloud(mesh, count, rng) {
  const areas = mesh.faces.map(([a, b, c]) => {
    const e1 = vsub(mesh.vertices[b], mesh.vertices[a]);
    const e2 = vsub(mesh.vertices[c], mesh.vertices[a]);
    return 0.5 * Math.hypot(...vcross(e1, e2));
  });
  const total = areas.reduce((a, x) => a + x, 0);
  const cum = [];
  let acc = 0;
  for (const ar of areas) { acc += ar; cum.push(acc); }

  const points = [];
  const normals = [];
  for (let k = 0; k < count; k++) {
    const r = rng() * total;
    let fi = cum.findIndex((v) => v >= r);
    if (fi < 0) fi = mesh.faces.length - 1;
    const [a, b, c] = mesh.faces[fi];
    let u = rng(); let v = rng();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const pa = mesh.vertices[a]; const pb = mesh.vertices[b]; const pc = mesh.vertices[c];
    const p = [
      pa[0] + u * (pb[0] - pa[0]) + v * (pc[0] - pa[0]),
      pa[1] + u * (pb[1] - pa[1]) + v * (pc[1] - pa[1]),
      pa[2] + u * (pb[2] - pa[2]) + v * (pc[2] - pa[2]),
    ];
    const na = mesh.normals[a]; const nb = mesh.normals[b]; const nc = mesh.normals[c];
    const n = vnormalize([
      na[0] + u * (nb[0] - na[0]) + v * (nc[0] - na[0]),
      na[1] + u * (nb[1] - na[1]) + v * (nc[1] - na[1]),
      na[2] + u * (nb[2] - na[2]) + v * (nc[2] - na[2]),
    ]) || [0, 1, 0];
    points.push(p);
    normals.push(n);
  }
  return { points, normals };
}

// 轴角 -> 旋转矩阵
function axisAngleMatrix(axis, angleRad) {
  const a = vnormalize(axis);
  const c = Math.cos(angleRad), s = Math.sin(angleRad), t = 1 - c;
  const [x, y, z] = a;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ];
}

function transformMesh(mesh, R, t) {
  return {
    vertices: mesh.vertices.map((p) => vadd(mApply(R, p), t)),
    faces: mesh.faces.map((f) => f.slice()),
    normals: mesh.normals.map((n) => mApply(R, n)),
  };
}

function transformCloud(cloud, R, t) {
  return {
    points: cloud.points.map((p) => vadd(mApply(R, p), t)),
    normals: cloud.normals.map((n) => mApply(R, n)),
  };
}

// 生成完整数据集：target 固定，source 为同一器物经已知刚性变换后的碎片
// scenario: intact | partial | mirrored | wrongpairs
function generateDataset(scenario, seed) {
  const rng = mulberry32(seed);
  const baseParams = scenario === 'plate'
    ? { width: 160, depth: 110, thick: 10, nX: 26, nZ: 20 }
    : {
      height: 120, radius: 90, thickness: 6, nTheta: 52, nY: 26,
      thetaStart: 0, thetaEnd: Math.PI * 1.2,
    };

  const kind = scenario === 'plate' ? 'plate' : 'bowl';
  const targetMesh = kind === 'bowl' ? makeBowl(baseParams, rng) : makePlate(baseParams, rng);

  // 真值变换：绕倾斜轴旋转 + 平移（毫米）
  const gtAxis = vnormalize([0.35, 1, 0.45]);
  const gtAngle = (scenario === 'bigtwist' ? 140 : 57) * Math.PI / 180;
  const gtR = axisAngleMatrix(gtAxis, gtAngle);
  const gtT = [42.5, -18.2, 77.3];

  // 源碎片参数
  let sourceParams = JSON.parse(JSON.stringify(baseParams));
  if (scenario === 'partial') {
    if (kind === 'bowl') {
      sourceParams.thetaEnd = sourceParams.thetaStart + Math.PI * 0.75;
      sourceParams.missingPatch = {
        centerY: 70, centerTheta: sourceParams.thetaStart + Math.PI * 0.4,
        radiusY: 26, radiusTheta: 0.5,
      };
    } else {
      sourceParams.missingPatch = { centerX: 20, centerZ: -10, radiusX: 34, radiusZ: 28 };
    }
  }
  const sourceMeshBase = kind === 'bowl'
    ? makeBowl(sourceParams, mulberry32(seed + 101))
    : makePlate(sourceParams, mulberry32(seed + 101));

  let sourceMesh;
  let mirrorUsed = false;
  let effectiveR = gtR;
  if (scenario === 'mirrored') {
    // 镜像：先反射 M（det=-1）再施加真值旋转，R_eff = R_gt · M
    const M = [-1, 0, 0, 0, 1, 0, 0, 0, 1];
    effectiveR = mMul3(gtR, M);
    sourceMesh = transformMesh(sourceMeshBase, effectiveR, gtT);
    mirrorUsed = true;
  } else {
    sourceMesh = transformMesh(sourceMeshBase, gtR, gtT);
  }

  const targetCloud = sampleCloud(targetMesh, 2600, mulberry32(seed + 7));
  const sourceCloud = sampleCloud(sourceMesh, 2200, mulberry32(seed + 11));

  // 候选对应点：在基准网格上取特征位置，按实际施加的变换映射到 source
  const landmarks = buildLandmarks(kind, baseParams, targetMesh, sourceMeshBase,
    effectiveR, gtT, scenario, rng);

  return {
    meta: {
      scenario, seed, kind, unit: 'mm', handedness: 'right', upAxis: 'Y',
      generatedAt: new Date().toISOString(),
      groundTruth: {
        rotation: effectiveR, translation: gtT,
        properRotation: gtR,
        axis: gtAxis, angleDeg: gtAngle * 180 / Math.PI, mirrored: mirrorUsed,
      },
      params: { target: baseParams, source: sourceParams },
    },
    target: {
      mesh: targetMesh,
      cloud: targetCloud,
    },
    source: {
      mesh: sourceMesh,
      cloud: sourceCloud,
    },
    landmarks,
  };
}

function mMul3(a, b) {
  const r = new Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  return r;
}

// 生成带真值的对应点；wrongpairs 场景故意混入错误配对
function buildLandmarks(kind, params, targetMesh, sourceMeshBase, gtR, gtT, scenario, rng) {
  const pairs = [];
  const n = 8;
  const thetaStart = params.thetaStart ?? 0;
  const thetaEnd = params.thetaEnd ?? Math.PI * 1.2;
  const height = params.height ?? 120;
  const radius = params.radius ?? 90;
  const width = params.width ?? 160;
  const depth = params.depth ?? 110;
  for (let k = 0; k < n; k++) {
    let p;
    if (kind === 'bowl') {
      const y = 18 + (k / (n - 1)) * (height - 32);
      const theta = thetaStart + (0.15 + 0.7 * rng()) * (thetaEnd - thetaStart);
      const t = y / height;
      const r = radius * (0.35 + 0.65 * Math.pow(t, 0.8));
      p = [r * Math.cos(theta), y, r * Math.sin(theta)];
    } else {
      // 板形：在厚度方向（Y）布置两层标志点，避免共面导致旋转欠定
      const layer = k % 2 === 0 ? (params.thick ?? 6) / 2 : -(params.thick ?? 6) / 2;
      p = [
        (rng() - 0.5) * width * 0.7,
        layer,
        (rng() - 0.5) * depth * 0.7,
      ];
    }
    const q = vadd(mApply(gtR, p), gtT);
    pairs.push({ target: p, sourceTruth: q, corrupt: false });
  }
  if (scenario === 'wrongpairs') {
    // 打乱后两对的 source 端，制造错误对应点
    const i0 = 2, i1 = 5;
    const tmp = pairs[i0].sourceTruth;
    pairs[i0].sourceTruth = pairs[i1].sourceTruth;
    pairs[i1].sourceTruth = tmp;
    pairs[i0].corrupt = true;
    pairs[i1].corrupt = true;
  }
  return pairs.map((pair, i) => ({
    id: i,
    targetPoint: pair.target,
    sourcePoint: pair.sourceTruth,
    targetWeight: 1,
    sourceWeight: 1,
    groundTruthCorrupt: pair.corrupt,
  }));
}

module.exports = {
  generateDataset,
  axisAngleMatrix,
  transformMesh,
  transformCloud,
  sampleCloud,
  mulberry32,
};

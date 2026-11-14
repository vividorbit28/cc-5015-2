// 3D 线性代数原语：向量、3x3 矩阵、Jacobi 对称特征分解、
// 极分解 SVD、加权 Kabsch 刚性配准。
// 约定：所有点为长度 3 的普通数组 [x, y, z]，右手坐标系。

'use strict';

const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const vnorm = (a) => Math.sqrt(vdot(a, a));
const vdist = (a, b) => vnorm(vsub(a, b));
const vnormalize = (a) => {
  const n = vnorm(a);
  return n < 1e-15 ? null : vscale(a, 1 / n);
};

// 矩阵以行主序 9 元素数组存储：m[3*i+j] 为第 i 行第 j 列
const mIdentity = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];

const mTranspose = (m) => [
  m[0], m[3], m[6],
  m[1], m[4], m[7],
  m[2], m[5], m[8],
];

const mMul = (a, b) => {
  const r = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[3 * i + j] =
        a[3 * i] * b[j] +
        a[3 * i + 1] * b[3 + j] +
        a[3 * i + 2] * b[6 + j];
    }
  }
  return r;
};

const mApply = (m, p) => [
  m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
  m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
  m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
];

const mDet = (m) =>
  m[0] * (m[4] * m[8] - m[5] * m[7]) -
  m[1] * (m[3] * m[8] - m[5] * m[6]) +
  m[2] * (m[3] * m[7] - m[4] * m[6]);

// 对称 3x3 矩阵的循环 Jacobi 特征分解
// 返回 { values: [λ1..λ3] 升序, vectors: [v1..v3]（对应特征向量）}
function symmetricEigen3(s) {
  const a = s.slice();
  const v = mIdentity(); // 累积 V，使 A0 = V diag(λ) V^T
  for (let sweep = 0; sweep < 80; sweep++) {
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    if (off < 1e-26) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[3 * p + q];
      if (Math.abs(apq) < 1e-18) continue;
      const app = a[3 * p + p];
      const aqq = a[3 * q + q];
      let tau = (aqq - app) / (2 * apq);
      let t;
      if (tau >= 0) t = 1 / (tau + Math.sqrt(1 + tau * tau));
      else t = -1 / (-tau + Math.sqrt(1 + tau * tau));
      const c = 1 / Math.sqrt(1 + t * t);
      const sn = t * c;
      // A ← J^T A J：先改列再改行（利用对称性）
      for (let k = 0; k < 3; k++) {
        const akp = a[3 * k + p];
        const akq = a[3 * k + q];
        a[3 * k + p] = c * akp - sn * akq;
        a[3 * k + q] = sn * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[3 * p + k];
        const aqk = a[3 * q + k];
        a[3 * p + k] = c * apk - sn * aqk;
        a[3 * q + k] = sn * apk + c * aqk;
      }
      // V ← V J：仅旋转第 p、q 列
      for (let k = 0; k < 3; k++) {
        const vkp = v[3 * k + p];
        const vkq = v[3 * k + q];
        v[3 * k + p] = c * vkp - sn * vkq;
        v[3 * k + q] = sn * vkp + c * vkq;
      }
    }
  }
  const idx = [0, 1, 2].sort((i, j) => a[3 * i + i] - a[3 * j + j]);
  return {
    values: idx.map((i) => a[3 * i + i]),
    vectors: idx.map((i) => [v[3 * 0 + i], v[3 * 1 + i], v[3 * 2 + i]]),
  };
}

// 由 H 的 SVD 构造两个正交极因子。
// H^T H = V diag(σ²) V^T，U 的列取 u_k = H v_k / σ_k。
// natural：U V^T（det 可能为 -1，镜像数据的真实最优解）；
// proper：强制 det=+1 的旋转（翻转最小奇异值方向）。
function rotationPolar(H) {
  const Ht = mTranspose(H);
  const HtH = mMul(Ht, H);
  const eig = symmetricEigen3(HtH);
  const sv = eig.values.map((x) => Math.sqrt(Math.max(0, x))); // 升序
  const cond = sv[0] > 1e-12 ? sv[2] / sv[0] : Infinity;

  const U = eig.vectors.map((v, k) => {
    const sigma = sv[k] > 1e-12 ? sv[k] : 1e-12;
    return mApply(H, v).map((x) => x / sigma);
  });
  const build = (signs) => {
    const R = new Array(9).fill(0);
    for (let k = 0; k < 3; k++) {
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          R[3 * i + j] += signs[k] * U[k][i] * eig.vectors[k][j];
    }
    return R;
  };
  const natural = build([1, 1, 1]);
  const detNatural = mDet(natural);
  const proper = detNatural > 0 ? natural : build([-1, 1, 1]);
  const det = mDet(proper);

  const RtR = mMul(mTranspose(proper), proper);
  const I = mIdentity();
  let polarResidual = 0;
  for (let i = 0; i < 9; i++) polarResidual = Math.max(polarResidual, Math.abs(RtR[i] - I[i]));

  return {
    R: proper,
    naturalRotation: natural,
    naturalDeterminant: detNatural,
    singularValues: sv,
    conditionNumber: cond,
    polarResidual,
    determinant: det,
  };
}

// 将旋转矩阵转为轴角（用于报告旋转幅度，单位度）
function rotationAxisAngle(R) {
  const c = Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2));
  const angle = Math.acos(c);
  if (angle < 1e-10) return { axis: [1, 0, 0], angleDeg: 0 };
  const s = 2 * Math.sin(angle);
  const axis = [
    (R[7] - R[5]) / s,
    (R[2] - R[6]) / s,
    (R[3] - R[1]) / s,
  ];
  return { axis, angleDeg: (angle * 180) / Math.PI };
}

// 加权 Kabsch：求 R, t 使 sum w_i ||R s_i + t - t_i||^2 最小
// sourcePoints -> targetPoints，仅允许旋转+平移，不做缩放
// weights 为每对对应点的总权重；targetWeight/sourceWeight 先按
// w = w_i * (wt_i*ws_i)/(wt_i+ws_i) 融合（测量方差倒数融合）
function weightedKabsch(sourcePoints, targetPoints, options = {}) {
  const n = sourcePoints.length;
  if (n !== targetPoints.length) throw new Error('对应点数量不一致');
  if (n < 3) throw new Error('至少需要 3 对对应点');

  const tw = options.targetWeights || new Array(n).fill(1);
  const sw = options.sourceWeights || new Array(n).fill(1);
  const w = new Array(n);
  let wSum = 0;
  for (let i = 0; i < n; i++) {
    const denom = tw[i] + sw[i];
    w[i] = denom > 1e-15 ? (tw[i] * sw[i]) / denom : 0;
    wSum += w[i];
  }
  if (wSum <= 0) throw new Error('有效权重之和为零');

  const cS = [0, 0, 0];
  const cT = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      cS[k] += w[i] * sourcePoints[i][k];
      cT[k] += w[i] * targetPoints[i][k];
    }
  }
  for (let k = 0; k < 3; k++) {
    cS[k] /= wSum;
    cT[k] /= wSum;
  }

  // H = sum w (t-cT)(s-cS)^T
  const H = new Array(9).fill(0);
  for (let i = 0; i < n; i++) {
    const ds = vsub(sourcePoints[i], cS);
    const dt = vsub(targetPoints[i], cT);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) H[3 * r + c] += w[i] * dt[r] * ds[c];
  }

  const polar = rotationPolar(H);
  const R = polar.R;
  const t = vsub(cT, mApply(R, cS));

  // 自然正交解若 det=-1，则为镜像数据的真实最优；仅用于诊断，不替代默认刚性输出
  const isReflection = polar.naturalDeterminant < 0;
  const Rref = isReflection ? polar.naturalRotation : null;
  let rmsReflected = Infinity;
  let tRef = null;
  if (Rref) {
    tRef = vsub(cT, mApply(Rref, cS));
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const d = vdist(vadd(mApply(Rref, sourcePoints[i]), tRef), targetPoints[i]);
      sq += w[i] * d * d;
    }
    rmsReflected = Math.sqrt(sq / wSum);
  }

  // 逐对残差与 RMS
  const residuals = [];
  let sqSum = 0;
  for (let i = 0; i < n; i++) {
    const d = vdist(vadd(mApply(R, sourcePoints[i]), t), targetPoints[i]);
    residuals.push(d);
    sqSum += w[i] * d * d;
  }
  const rms = Math.sqrt(sqSum / wSum);

  // 退化诊断：对加权去重心点做 PCA
  const cov = new Array(9).fill(0);
  for (let i = 0; i < n; i++) {
    const ds = vsub(sourcePoints[i], cS);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) cov[3 * r + c] += w[i] * ds[r] * ds[c];
  }
  const pca = symmetricEigen3(cov);
  const eig = pca.values;
  const span = eig[2] > 1e-12 ? eig[2] : 1e-12;
  const degeneracy = {
    eigenvalues: eig,
    planarRatio: eig[0] / span, // 接近 0 => 近共面
    linearRatio: eig[1] / span, // 接近 0 => 近共线
    rank: eig[0] / span < 1e-6 ? (eig[1] / span < 1e-6 ? 1 : 2) : 3,
  };

  const aa = rotationAxisAngle(R);
  return {
    rotation: R,
    translation: t,
    rms,
    residuals,
    weights: w,
    polar: {
      singularValues: polar.singularValues,
      conditionNumber: Number.isFinite(polar.conditionNumber)
        ? polar.conditionNumber
        : null,
      determinant: polar.determinant,
      polarResidual: polar.polarResidual,
    },
    reflectionFit: Rref
      ? { rotation: Rref, translation: tRef, rms: rmsReflected, determinant: -1 }
      : null,
    // 镜像线索：自然最优解为反射且其残差显著优于强制刚性解
    mirrorSuspected: isReflection && rmsReflected < rms * 0.5,
    degeneracy,
    rotationAngleDeg: aa.angleDeg,
    rotationAxis: aa.axis,
    centroids: { source: cS, target: cT },
    pairCount: n,
  };
}

// 应用刚性变换（不含缩放）
function applyTransform(points, tf) {
  return points.map((p) => vadd(mApply(tf.rotation, p), tf.translation));
}

module.exports = {
  vadd, vsub, vscale, vdot, vcross, vnorm, vdist, vnormalize,
  mIdentity, mTranspose, mMul, mApply, mDet,
  symmetricEigen3, rotationPolar, rotationAxisAngle,
  weightedKabsch, applyTransform,
};

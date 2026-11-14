// 三维 KD 树：最近邻、k 近邻、半径搜索。节点保存点集索引，不复制点数据。

'use strict';

const { vsub, vdot, vdist } = require('./linalg');

function buildKDTree(points, indices, depth) {
  if (indices.length === 0) return null;
  const axis = depth % 3;
  indices.sort((i, j) => points[i][axis] - points[j][axis]);
  const mid = indices.length >> 1;
  return {
    index: indices[mid],
    axis,
    left: buildKDTree(points, indices.slice(0, mid), depth + 1),
    right: buildKDTree(points, indices.slice(mid + 1), depth + 1),
  };
}

class KDTree {
  constructor(points) {
    this.points = points;
    this.root = buildKDTree(points, points.map((_, i) => i), 0);
  }

  // 最近邻：返回 { index, distance }
  nearest(point, maxDistance = Infinity) {
    let best = null;
    let bestSq = maxDistance * maxDistance;
    const p = this.points;
    const visit = (node) => {
      if (!node) return;
      const nodePoint = p[node.index];
      const d = vdist(point, nodePoint);
      if (d * d < bestSq) {
        best = { index: node.index, distance: d };
        bestSq = d * d;
      }
      const diff = point[node.axis] - nodePoint[node.axis];
      const first = diff < 0 ? node.left : node.right;
      const second = diff < 0 ? node.right : node.left;
      visit(first);
      // 分割平面距离小于当前最佳半径时，另一侧才可能存在更优点
      if (diff * diff < bestSq) visit(second);
    };
    visit(this.root);
    return best;
  }

  // 半径内全部邻居
  radius(point, radius) {
    const out = [];
    const p = this.points;
    const r2 = radius * radius;
    const visit = (node) => {
      if (!node) return;
      const nodePoint = p[node.index];
      const d2 = vdot(vsub(point, nodePoint), vsub(point, nodePoint));
      if (d2 <= r2) out.push({ index: node.index, distance: Math.sqrt(d2) });
      const diff = point[node.axis] - nodePoint[node.axis];
      if (diff * diff <= r2) {
        visit(node.left);
        visit(node.right);
      } else {
        visit(diff < 0 ? node.left : node.right);
      }
    };
    visit(this.root);
    return out;
  }
}

module.exports = { KDTree };

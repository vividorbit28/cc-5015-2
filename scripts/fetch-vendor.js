#!/usr/bin/env node
// 下载前端运行时依赖（Three.js）到本地 vendor 目录，保证离线可复现。
// 克隆仓库后执行：node scripts/fetch-vendor.js
'use strict';
const https = require('https');
const fs = require('fs');
const path = require('path');

const VERSION = '0.160.0';
const BASE = `https://cdn.jsdelivr.net/npm/three@${VERSION}`;
const OUT = path.resolve(__dirname, '..', 'vendor');
const FILES = {
  'three.module.js': `${BASE}/build/three.module.js`,
  'OrbitControls.js': `${BASE}/examples/jsm/controls/OrbitControls.js`,
};

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (res) => {
      if (res.statusCode !== 200) {
        fs.unlinkSync(dest);
        reject(new Error(`${url} -> HTTP ${res.statusCode}`));
        return;
      }
      res.pipe(file);
      file.on('finish', () => file.close(resolve));
    }).on('error', (e) => { fs.unlinkSync(dest); reject(e); });
  });
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  for (const [name, url] of Object.entries(FILES)) {
    const dest = path.join(OUT, name);
    if (fs.existsSync(dest) && fs.statSync(dest).size > 1000) {
      console.log(`已存在，跳过 ${name}`);
      continue;
    }
    console.log(`下载 ${name} ...`);
    await download(url, dest);
  }
  console.log('vendor 依赖就绪');
})().catch((e) => { console.error(e.message); process.exit(1); });

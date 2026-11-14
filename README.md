# 界面启动与配准诊断使用说明

工作台由零依赖 Node 服务提供前端页面与 JSON API；配准计算全部在后端内核完成，前端只负责可视化与交互。

## 1. 启动界面

```bash
npm start            # 等价于 node server/server.js，默认 8080
# 或指定端口
PORT=9000 node server/server.js
```

浏览器打开 `http://localhost:8080`（端口以实际监听为准）。

首次运行若提示缺少 `vendor/three.module.js`，先执行 `node scripts/fetch-vendor.js` 获取第三方运行时。
无需 `npm install` 安装任何运行依赖。

## 2. 界面上的配准流程

1. **载入程序生成数据**：选择碗形 / 平板 / 局部缺失 / 镜像 / 错误对应 / 大角度旋转，点击“生成并载入”。
2. **估计刚性变换**：生成 Kabsch 基线与多个 RANSAC 候选；候选按内点与重叠率排序，对称备选并列保留。
3. **拖动初始位姿**（第 3 面板下半部）：
   - 位姿基准来自**内核产出的候选方案**（下拉框），滑块只叠加绕 X/Y/Z 与平移增量；
   - 每次拖动都会调用 `POST /api/sessions/:id/diagnose`，实时显示
     **碎片重合率 / 距离残差（中位、均值）/ 有效对应数 / 根因**，并在叠合视口按逐点残差着色、
     按逐对原因给对应点连线着色（内点黄、超阈值红）；
   - 诊断失败会直接回到配准面板与提示条，不会用默认值伪装成功。
4. **ICP 局部优化**：对选中方案执行，另存为新方案；右侧“配准诊断证据”面板显示
   证据指纹、停止条件、迭代轮数（受日志预算裁剪）、初始→最终重叠、噪声地板、尺度距离比、
   残差下降曲线、有效对应趋势、阶段定位（input/correspondence/weighting/pose-update/stop）、
   根因与逐轮重放一致性。
5. **人工确认 / 导出**：确认记录绑定对应点快照与参数；导出 JSON（`exportVersion: 2`）含迭代证据与重放摘要。

## 3. 命令行诊断（不启动界面）

```bash
node scripts/diagnose.js --scenario partial --seed 11 --perturb-angle 35 --perturb-translation 30,0,0
node scripts/diagnose.js --scenario mirrored
node scripts/diagnose.js --scenario bowl --source-scale 1.08    # 复现单位/尺度问题
node scripts/diagnose.js --evidence exports/evidence.json --replay   # 离线重放核对
```

`--check-truth` 是独立的离线校验器，只在此时读取测试真值，且不回流到求解或根因分类；
诊断实现与内核的生产路径不访问 `dataset.meta.groundTruth`。

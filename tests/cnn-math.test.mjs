import test from 'node:test';
import assert from 'node:assert/strict';
import M from '../teaching/ai-built-environment/interactive-lab/cnn-math.js';

const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} ≉ ${expected}`);

/* The convolution and pooling exactly as they were in cnn.js before the extraction (commit 7068610).
   Used as an oracle so existing deep links keep producing identical numbers. */
function legacyPad(matrix, padding) { if (!padding) return matrix.map(row => [...row]); const size = matrix.length + padding * 2; return Array.from({ length: size }, (_, row) => Array.from({ length: size }, (_, column) => row < padding || column < padding || row >= size - padding || column >= size - padding ? 0 : matrix[row - padding][column - padding])); }
function legacyConvolve(pixels, kernel, stride, padding) { const padded = legacyPad(pixels, padding), rows = Math.floor((padded.length - 3) / stride) + 1, columns = Math.floor((padded[0].length - 3) / stride) + 1, map = []; for (let row = 0; row < rows; row++) { const line = []; for (let column = 0; column < columns; column++) { let sum = 0; for (let kr = 0; kr < 3; kr++) for (let kc = 0; kc < 3; kc++) sum += padded[row * stride + kr][column * stride + kc] * kernel[kr][kc]; line.push(sum); } map.push(line); } return { map, padded }; }
function legacyPool(map, type) { if (type === 'none') return null; const output = []; for (let row = 0; row < map.length; row += 2) { const line = []; for (let column = 0; column < map[0].length; column += 2) { const values = []; for (let r = row; r < Math.min(row + 2, map.length); r++) for (let c = column; c < Math.min(column + 2, map[0].length); c++) values.push(map[r][c]); line.push(type === 'avg' ? values.reduce((sum, value) => sum + value, 0) / values.length : Math.max(...values)); } output.push(line); } return output; }

test('existing patterns, kernels and geometries give identical results to the original code', () => {
  for (const pattern of ['edge', 'cross', 'corner', 'checker', 'diagonal', 'custom']) {
    const pixels = M.cnnPlayPattern(pattern);
    assert.equal(pixels.length, 7);
    for (const [name, kernel] of Object.entries(M.CNN_PLAY_KERNELS)) for (const padding of [0, 1]) for (const stride of [1, 2]) {
      const now = M.cnnPlayConvolve(pixels, kernel, stride, padding), before = legacyConvolve(pixels, kernel, stride, padding);
      assert.deepEqual(now.map, before.map, `${pattern}/${name}/p${padding}/s${stride}`);
      assert.deepEqual(now.padded, before.padded);
      for (const pool of ['max', 'avg', 'none']) assert.deepEqual(M.cnnPlayPool(M.cnnPlayActivate(now.map, 'relu'), pool), legacyPool(before.map.map(row => row.map(value => Math.max(0, value))), pool));
    }
  }
  assert.deepEqual(M.cnnPlayActivate([[-2, 3]], 'none'), [[-2, 3]]);
});

test('output size ⌊(N + 2P − D(F−1) − 1)/S⌋ + 1, including dilation and stride 3', () => {
  assert.equal(M.cnnOutputSize(7, 3, 0, 1), 5);
  assert.equal(M.cnnOutputSize(7, 3, 1, 1), 7);
  assert.equal(M.cnnOutputSize(7, 3, 0, 2), 3);
  assert.equal(M.cnnOutputSize(7, 3, 1, 2), 4);
  assert.equal(M.cnnOutputSize(7, 3, 0, 3), 2);
  assert.equal(M.cnnOutputSize(7, 3, 0, 1, 2), 3);
  assert.equal(M.cnnOutputSize(7, 3, 2, 1, 2), 7, 'padding 2 keeps a dilated 3×3 "same"');
  assert.equal(M.cnnOutputSize(7, 3, 0, 3, 2), 1);
  assert.equal(M.cnnOutputSize(227, 11, 0, 4), 55, 'AlexNet conv1');
  assert.equal(M.cnnOutputSize(4, 3, 0, 1, 2), 0, 'dilated filter does not fit');
  const { map } = M.cnnPlayConvolve(M.cnnPlayPattern('edge'), M.CNN_PLAY_KERNELS.vertical, 1, 0, 2);
  assert.equal(map.length, 3);
  // Dilation 2 at output (0,1): taps at columns 1, 3, 5 → (0 − 0 + 1) per row × 3 rows = 3
  assert.equal(map[0][1], 3);
});

test('a 3-channel filter produces one map: the sum of per-channel convolutions plus bias', () => {
  const gray = M.cnnPlayPattern('edge'), channels = [gray, gray, gray], v = M.CNN_PLAY_KERNELS.vertical;
  const single = M.cnnPlayConvolve(gray, v, 1, 0).map;
  const result = M.cnnMultiChannelConvolve(channels, [v, v, v], 0.5, 1, 0);
  assert.equal(result.partials.length, 3);
  result.map.forEach((row, r) => row.forEach((value, c) => close(value, 3 * single[r][c] + 0.5)));
  const colour = [M.cnnPlayPattern('edge'), M.cnnPlayPattern('cross'), M.cnnPlayPattern('corner')];
  const filter = [v, M.CNN_PLAY_KERNELS.horizontal, M.CNN_PLAY_KERNELS.diagonal];
  const mixed = M.cnnMultiChannelConvolve(colour, filter, -1, 2, 1);
  mixed.map.forEach((row, r) => row.forEach((value, c) => close(value, colour.reduce((sum, channel, k) => sum + M.cnnPlayConvolve(channel, filter[k], 2, 1).map[r][c], 0) - 1)));
});

test('parameter count (F·F·K + 1)·L', () => {
  assert.equal(M.cnnParamCount(3, 1, 1), 10);
  assert.equal(M.cnnParamCount(3, 3, 4), 112);
  assert.equal(M.cnnParamCount(5, 3, 6), 456, 'LeNet-5 C1 has 156 params for 1 input channel; 3 channels: 76 × 6');
  assert.equal(M.cnnParamCount(5, 1, 6), 156, 'LeNet-5 C1');
});

test('activations: ReLU, Leaky ReLU (0.1) and ELU (α = 1)', () => {
  assert.equal(M.cnnActivationValue('relu', -2), 0);
  close(M.cnnActivationValue('leaky', -2), -0.2);
  assert.equal(M.cnnActivationValue('leaky', 3), 3);
  close(M.cnnActivationValue('elu', -1), Math.exp(-1) - 1);
  assert.equal(M.cnnActivationValue('elu', 2), 2);
  assert.equal(M.cnnActivationValue('none', -2), -2);
});

test('receptive field arithmetic matches the guide and grows 3 → 5 → 7', () => {
  const conv = { type: 'conv', k: 3, s: 1, p: 0 };
  const three = M.cnnReceptiveField([conv, conv, conv], 7);
  assert.deepEqual(three.map(row => row.r), [1, 3, 5, 7]);
  assert.deepEqual(three.map(row => row.n), [7, 5, 3, 1]);
  // Guide's worked example: 5 × 5 input, k = 3, p = 1, s = 2 twice → n 3 then 2, j 2 then 4, r 3 then 7
  const guide = M.cnnReceptiveField([{ type: 'conv', k: 3, s: 2, p: 1 }, { type: 'conv', k: 3, s: 2, p: 1 }], 5);
  assert.deepEqual(guide.slice(1).map(row => [row.n, row.j, row.r, row.start]), [[3, 2, 3, 0.5], [2, 4, 7, 0.5]]);
  // conv 3 → 2×2 max pool (stride 2) → conv 3: r = 3, 4, 8
  const pooled = M.cnnReceptiveField([conv, { type: 'pool', k: 2, s: 2 }, conv], 16);
  assert.deepEqual(pooled.map(row => row.r), [1, 3, 4, 8]);
  assert.deepEqual(pooled.map(row => row.n), [16, 14, 7, 5]);
  assert.equal(M.cnnReceptiveField([conv, { type: 'pool', k: 2, s: 2 }], 5)[2].n, 2, 'ceil-mode pool of 3 × 3 gives 2 × 2');
  // dilation 2 twice: k_eff = 5 → r = 5, 9
  const dilated = M.cnnReceptiveField([{ ...conv, d: 2 }, { ...conv, d: 2 }], 16);
  assert.deepEqual(dilated.map(row => row.r), [1, 5, 9]);
  // stride 2 first layer: r = 3, then 3 + 2·2 = 7
  assert.equal(M.cnnReceptiveField([{ ...conv, s: 2 }, conv], 15)[2].r, 7);
});

test('tracing a top cell back to the input matches the receptive-field size', () => {
  const conv = { type: 'conv', k: 3, s: 1, p: 0 };
  const levels = M.cnnTraceIndices([conv, conv, conv], [7, 5, 3], 0);
  assert.deepEqual(levels.map(level => level.length), [1, 3, 5, 7]);
  assert.deepEqual(levels[3], [0, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual(M.cnnTraceIndices([{ ...conv, d: 2 }], [7], 0)[1], [0, 2, 4], 'dilated taps skip cells');
  const gridding = M.cnnTraceIndices([{ ...conv, d: 2 }, { ...conv, d: 2 }], [16, 12], 0)[2];
  assert.deepEqual(gridding, [0, 2, 4, 6, 8], 'two dilation-2 layers touch only even cells across a span of 9');
  assert.deepEqual(M.cnnTraceIndices([{ ...conv, p: 1 }], [7], 0)[1], [-1, 0, 1], 'index −1 is padding');
  assert.deepEqual(M.cnnTraceIndices([{ type: 'pool', k: 2, s: 2 }], [5], 2)[1], [4], 'partial ceil-mode window');
});

test('cost calculator: AlexNet conv1, VGG 3 × 3, ResNet bottleneck and MobileNet', () => {
  const alex = M.cnnConvCost({ H: 227, W: 227, Kin: 3, Kout: 96, F: 11, S: 4, P: 0 });
  assert.deepEqual([alex.Hout, alex.Wout, alex.params, alex.macs], [55, 55, 34944, 105415200]);
  const vgg = M.cnnConvCost({ H: 56, W: 56, Kin: 256, Kout: 256, F: 3, S: 1, P: 1 });
  assert.deepEqual([vgg.Hout, vgg.params, vgg.macs], [56, 590080, 1849688064]);
  const bottleneck = M.cnnBottleneckCost({ H: 56, W: 56, Kin: 256, Kout: 256, F: 3, S: 1, P: 1, Kmid: 64 });
  assert.deepEqual(bottleneck.parts.map(part => part.params), [16448, 36928, 16640]);
  assert.equal(bottleneck.params, 70016);
  assert.equal(bottleneck.macs, 3136 * (256 * 64 + 9 * 64 * 64 + 64 * 256));
  const strided = M.cnnBottleneckCost({ H: 56, W: 56, Kin: 256, Kout: 512, F: 3, S: 2, P: 1, Kmid: 128 });
  assert.equal(strided.Hout, 28);
  assert.equal(strided.parts[0].macs, 56 * 56 * 256 * 128, '1 × 1 reduce runs at the input resolution');
  assert.equal(strided.parts[1].macs, 28 * 28 * 9 * 128 * 128);
  for (const K of [32, 64, 256, 1024]) {
    const args = { H: 14, W: 14, Kin: K, Kout: K, F: 3, S: 1, P: 1 };
    const ratio = M.cnnConvCost(args).macs / M.cnnSeparableCost(args).macs;
    close(ratio, 9 * K / (9 + K), 1e-9);
    close(M.cnnSeparableSpeedup(3, K), 9 * K / (9 + K));
  }
  const mobile = M.cnnSeparableCost({ H: 112, W: 112, Kin: 32, Kout: 64, F: 3, S: 1, P: 1 });
  assert.deepEqual(mobile.parts.map(part => part.macs), [112 * 112 * 9 * 32, 112 * 112 * 32 * 64]);
});

test('three 3 × 3 convs use 27K² weights versus 49K² for one 7 × 7', () => {
  const K = 64, threeByThree = 3 * (M.cnnParamCount(3, K, K) - K), sevenBySeven = M.cnnParamCount(7, K, K) - K;
  assert.equal(threeByThree, 27 * K * K);
  assert.equal(sevenBySeven, 49 * K * K);
});

test('pool backward: max routes to the argmax, average splits by window size, sums are conserved', () => {
  const map = [[1, 3, 2], [3, 0, 5], [4, 4, 1]], grads = [[1, -2], [0.5, 3]];
  const max = M.cnnPoolBackward(map, 'max', grads);
  assert.deepEqual(max.grad, [[0, 1, 0], [0, 0, -2], [0.5, 0, 3]], 'tie 3/3 goes to the first in row-major order; 4/4 likewise');
  const avg = M.cnnPoolBackward(map, 'avg', grads);
  assert.deepEqual(avg.grad, [[0.25, 0.25, -1], [0.25, 0.25, -1], [0.25, 0.25, 3]]);
  for (const result of [max, avg]) close(result.grad.flat().reduce((a, b) => a + b, 0), grads.flat().reduce((a, b) => a + b, 0));
  assert.equal(max.grad.flat().filter(value => value !== 0).length, 4);
  // The forward pass picks the same cells
  assert.deepEqual(M.cnnPlayPool(map, 'max'), [[3, 5], [4, 1]]);
});

test('softmax is normalised and stable; prototype head uses normalised templates', () => {
  const p = M.cnnSoftmax([2, 1, 0.1]);
  close(p.reduce((a, b) => a + b, 0), 1);
  close(p[0], Math.exp(2) / (Math.exp(2) + Math.exp(1) + Math.exp(0.1)));
  assert.deepEqual(M.cnnSoftmax([1000, 1000]), [0.5, 0.5]);
  const head = M.cnnHeadLogits([1, 1], [[3, 4], [0, 0]], 2);
  assert.deepEqual(head.weights, [[0.6, 0.8], [0, 0]]);
  close(head.logits[0], 2 * 1.4);
  assert.equal(head.logits[1], 0);
});

test('self-generated scenes are 16 × 16 grayscale in [0, 1] and deterministic', () => {
  for (const name of Object.keys(M.CNN_SCENES)) {
    const scene = M.cnnScene(name);
    assert.equal(scene.length, 16);
    assert.ok(scene.every(row => row.length === 16 && row.every(value => value >= 0 && value <= 1)));
    assert.deepEqual(scene, M.cnnScene(name));
  }
  assert.deepEqual(M.cnnPlayPattern('cross', 15)[7], Array(15).fill(1));
});

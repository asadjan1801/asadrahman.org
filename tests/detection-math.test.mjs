import test from 'node:test';
import assert from 'node:assert/strict';
import D from '../teaching/ai-built-environment/interactive-lab/detection-math.js';

const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} ≉ ${expected}`);

test('IoU: overlap, identical, contained, disjoint and touching boxes', () => {
  const parts = D.detIouParts([100, 60, 200, 140], [120, 70, 215, 150]);
  assert.equal(parts.intersection, 80 * 70);
  assert.equal(parts.union, 8000 + 95 * 80 - 5600);
  close(parts.iou, 5600 / 10000);
  assert.equal(D.detIou([0, 0, 10, 10], [0, 0, 10, 10]), 1);
  close(D.detIou([0, 0, 10, 10], [2, 2, 7, 7]), 25 / 100, 1e-12);
  assert.equal(D.detIou([0, 0, 10, 10], [20, 20, 30, 30]), 0);
  assert.equal(D.detIou([0, 0, 10, 10], [10, 0, 20, 10]), 0, 'shared edge has zero area');
  assert.equal(D.detIntersection([0, 0, 10, 10], [10, 0, 20, 10]), null);
  // A box covering the ground truth at 1.6× its width and height: IoU = 1 / 1.6² = 0.390625
  close(D.detIou([100, 60, 200, 140], [70, 36, 230, 164]), 1 / 2.56, 1e-12);
});

test('NMS reproduces the d2l.ai example (class-agnostic, threshold 0.5)', () => {
  const boxes = [[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95], [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]];
  const scores = [0.9, 0.8, 0.7, 0.9];
  const result = D.detNms(boxes, scores, 0.5);
  assert.deepEqual(result.keep, [0, 3]);
  assert.deepEqual(result.steps[0].suppressed.map(item => item.index), [1, 2]);
  close(result.steps[0].suppressed[0].iou, 0.3024 / 0.4104, 1e-9);
  assert.deepEqual(D.detNms(boxes, scores, 0.6).keep, [0, 3, 2], 'IoU(0,1) = 0.737 > 0.6 is suppressed; IoU(0,2) = 0.545 survives');
  assert.deepEqual(D.detNms(boxes, scores, 0.75).keep, [0, 3, 1, 2]);
});

test('anchors: 9 per position, area s² and width/height = r', () => {
  const anchors = D.detAnchors(8, 5, 40, [40, 80, 120], [0.5, 1, 2]);
  assert.equal(anchors.length, 8 * 5 * 9);
  for (const anchor of anchors.slice(0, 9)) {
    const c = D.detCenter(anchor.box);
    close(c.w * c.h, anchor.scale ** 2, 1e-9);
    close(c.w / c.h, anchor.ratio, 1e-12);
    close(c.x, 20, 1e-9); close(c.y, 20, 1e-9);
  }
  close(D.detCenter(anchors[8 * 9].box).x, 20, 1e-9);
  close(D.detCenter(anchors[8 * 9].box).y, 60, 1e-9, 'row 1 starts after the 8 positions of row 0');
});

test('anchor assignment reproduces the d2l.ai example: classes [0, 1, 2, 0, 2]', () => {
  const gts = [[0.1, 0.08, 0.52, 0.92], [0.55, 0.2, 0.9, 0.88]];
  const anchors = [[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4], [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]];
  const map = D.detAssignD2l(anchors, gts, 0.5);
  assert.deepEqual(map.map(j => j + 1), [0, 1, 2, 0, 2]);
  // d2l offsets use σ = (0.1, 0.1, 0.2, 0.2)
  const sigma = [0.1, 0.1, 0.2, 0.2];
  const expected = { 1: [1.40, 10.0, 2.59, 7.18], 2: [-1.20, 0.269, 1.68, -1.57], 4: [-0.571, -1.00, 4.17e-6, 0.626] };
  for (const [index, values] of Object.entries(expected)) D.detEncode(anchors[index], gts[map[index]], sigma).forEach((value, k) => close(value, values[k], 0.006));
  // Faster R-CNN labels on the same data: positives ≥ 0.5, background < 0.3, best anchor per object forced positive
  const labelled = D.detLabelAnchors(anchors, gts, 0.5, 0.3);
  assert.deepEqual(labelled.anchors.map(item => item.label), [0, 1, 1, 0, 1]);
  assert.equal(labelled.anchors[1].forced, true, 'anchor 1 has IoU < 0.5 with the dog but is its best anchor');
});

test('box regression: decode inverts encode, zero offsets give the anchor', () => {
  const anchor = [40, 30, 120, 90], box = [55, 20, 150, 110];
  const t = D.detEncode(anchor, box);
  close(t[0], (102.5 - 80) / 80);
  close(t[1], (65 - 60) / 60);
  close(t[2], Math.log(95 / 80));
  close(t[3], Math.log(90 / 60));
  D.detDecode(anchor, t).forEach((value, k) => close(value, box[k], 1e-9));
  D.detDecode(anchor, [0, 0, 0, 0]).forEach((value, k) => close(value, anchor[k], 1e-12));
  D.detDecode(anchor, D.detEncode(anchor, box, [0.1, 0.1, 0.2, 0.2]), [0.1, 0.1, 0.2, 0.2]).forEach((value, k) => close(value, box[k], 1e-9));
});

test('VOC matching: duplicates and poor boxes are false positives', () => {
  const gts = [[0, 0, 10, 10], [20, 20, 30, 30]];
  const detections = [{ box: [0, 0, 10, 10], score: 0.9 }, { box: [1, 0, 11, 10], score: 0.8 }, { box: [20, 20, 30, 40], score: 0.7 }, { box: [21, 21, 30, 30], score: 0.6 }];
  const outcomes = D.detMatch(detections, gts, 0.5);
  assert.deepEqual(outcomes.map(item => item.tp), [true, false, true, false]);
  assert.equal(outcomes[1].duplicate, true, 'IoU 0.82 with an already-matched object');
  close(outcomes[2].iou, 0.5, 1e-12);
  assert.equal(outcomes[2].tp, true, 'IoU exactly at the threshold counts (≥)');
  assert.equal(outcomes[3].duplicate, true, 'lower-scoring second box on the same object');
  assert.equal(D.detMatch(detections, gts, 0.85).filter(item => item.tp).length, 1);
});

test('AP: all-point interpolation and the 11-point variant on a hand-computed case', () => {
  // Ranked outcomes TP, FP, TP, TP, FP with 4 objects: recall .25 .25 .5 .75 .75, precision 1 .5 .667 .75 .6
  const outcomes = [true, false, true, true, false].map((tp, i) => ({ tp, score: 1 - i / 10 }));
  const curve = D.detPrCurve(outcomes, 4);
  assert.deepEqual(curve.map(point => point.recall), [0.25, 0.25, 0.5, 0.75, 0.75]);
  close(curve[2].precision, 2 / 3);
  const { ap, envelope } = D.detAveragePrecision(curve);
  close(ap, 0.25 * 1 + 0.25 * 0.75 + 0.25 * 0.75, 1e-12);
  assert.equal(envelope.at(-1).precision, 0);
  close(D.detElevenPointAp(curve), (3 * 1 + 5 * 0.75) / 11, 1e-12);
  assert.equal(D.detAveragePrecision(D.detPrCurve([{ tp: true }, { tp: true }], 2)).ap, 1);
  assert.equal(D.detAveragePrecision([]).ap, 0);
});

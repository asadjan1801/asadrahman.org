import test from 'node:test';
import assert from 'node:assert/strict';
import O from '../teaching/ai-built-environment/interactive-lab/optim-math.js';

const close = (actual, expected, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} ≉ ${expected}`);
// L = x² + 4y², ∇L = (2x, 8y)
const gradient = (x, y) => ({ x: 2 * x, y: 8 * y });

test('momentum: first step is a gradient step, the second adds β·v', () => {
  let state = O.optimInit({ x: 1, y: 1 });
  state = O.optimStep('momentum', state, gradient, { rate: 0.1, momentum: 0.5 });
  close(state.velocity.x, -0.2); close(state.velocity.y, -0.8);
  close(state.position.x, 0.8); close(state.position.y, 0.2);
  state = O.optimStep('momentum', state, gradient, { rate: 0.1, momentum: 0.5 });
  close(state.velocity.x, 0.5 * -0.2 - 0.1 * 1.6); // −0.26
  close(state.position.x, 0.8 - 0.26);
  close(state.velocity.y, 0.5 * -0.8 - 0.1 * 1.6); // −0.56
  close(state.position.y, 0.2 - 0.56);
});

test('momentum reproduces the original Gradient Descent module arithmetic exactly', () => {
  const grad = (x, y) => ({ x: 0.44 * (x - 1), y: 6 * (y + 0.6) });
  let state = O.optimInit({ x: -3, y: 3 }), position = { x: -3, y: 3 }, velocity = { x: 0, y: 0 };
  const clampValue = (value, min, max) => Math.max(min, Math.min(max, value));
  for (let i = 0; i < 40; i++) {
    const g = grad(position.x, position.y);
    velocity = { x: 0.5 * velocity.x - 0.3 * g.x, y: 0.5 * velocity.y - 0.3 * g.y };
    position = { x: clampValue(position.x + velocity.x, -5, 5), y: clampValue(position.y + velocity.y, -5, 5) };
    state = O.optimStep('momentum', state, grad, { rate: 0.3, momentum: 0.5 });
    assert.deepEqual(state.position, position);
  }
});

test('Nesterov evaluates the gradient at the look-ahead point θ + βv', () => {
  let state = { ...O.optimInit({ x: 1, y: 1 }), velocity: { x: -0.2, y: -0.8 } };
  state = O.optimStep('nesterov', state, gradient, { rate: 0.1, momentum: 0.5 });
  // look-ahead (0.9, 0.6): ∇ = (1.8, 4.8)
  close(state.gradient.x, 1.8); close(state.gradient.y, 4.8);
  close(state.velocity.x, -0.1 - 0.18); close(state.velocity.y, -0.4 - 0.48);
  close(state.position.x, 1 - 0.28); close(state.position.y, 1 - 0.88);
});

test('RMSProp: first step is α·g/(√(0.1·g²) + ε) ≈ α/√0.1 in every coordinate', () => {
  const state = O.optimStep('rmsprop', O.optimInit({ x: 1, y: 1 }), gradient, { rate: 0.1 });
  close(state.s.x, 0.1 * 4); close(state.s.y, 0.1 * 64);
  close(state.velocity.x, -0.1 * 2 / (Math.sqrt(0.4) + 1e-8));
  close(state.velocity.y, -0.1 * 8 / (Math.sqrt(6.4) + 1e-8));
  close(state.velocity.x, state.velocity.y, 1e-8); // equal up to the effect of ε
});

test('Adam: bias correction makes the first step −α·sign(g)', () => {
  let state = O.optimStep('adam', O.optimInit({ x: 1, y: -1 }), gradient, { rate: 0.1 });
  close(state.velocity.x, -0.1, 1e-8); close(state.velocity.y, 0.1, 1e-8);
  assert.equal(state.t, 1);
  // second step by hand
  const g = gradient(state.position.x, state.position.y), m = 0.9 * state.m.x + 0.1 * g.x, s = 0.999 * state.s.x + 0.001 * g.x * g.x;
  const expected = -0.1 * (m / (1 - 0.9 ** 2)) / (Math.sqrt(s / (1 - 0.999 ** 2)) + 1e-8);
  state = O.optimStep('adam', state, gradient, { rate: 0.1 });
  close(state.velocity.x, expected);
});

test('positions are clipped to the plotted square and β* = (1 − √(αλ))²', () => {
  const state = O.optimStep('momentum', O.optimInit({ x: 4.9, y: 0 }), () => ({ x: -10, y: 0 }), { rate: 1 });
  assert.equal(state.position.x, 5);
  close(O.optimCriticalMomentum(0.1, 0.1), (1 - 0.1) ** 2);
});

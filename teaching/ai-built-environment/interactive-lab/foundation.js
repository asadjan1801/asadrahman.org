"use strict";

Object.assign(CONCEPT_META, {
  diffusion: {
    category: "Generative & Foundation Models",
    title: "Diffusion Models",
    summary: "Move between a structured signal and noise while inspecting the schedule and a prepared denoising estimate.",
    question: "How can repeated denoising turn randomness into structure?"
  },
  transformer: {
    category: "Generative & Foundation Models",
    title: "Attention & Transformers",
    summary: "Inspect how query–key scores become attention weights and mix information between tokens.",
    question: "How does each token decide which other tokens matter?"
  },
  llm: {
    category: "Generative & Foundation Models",
    title: "Large Language Models",
    summary: "Generate a sequence one token at a time from a visible next-token probability distribution.",
    question: "How do local next-token choices create a complete sequence?"
  },
  vlm: {
    category: "Generative & Foundation Models",
    title: "Vision–Language Models",
    summary: "Compare visual patch features with text features inside a small shared representation space.",
    question: "How can images and words be compared in one feature space?"
  }
});

function foundationSoftmax(values, temperature = 1) {
  const finite = values.filter(Number.isFinite);
  const maximum = finite.length ? Math.max(...finite) : 0;
  const scaled = values.map(value => Number.isFinite(value) ? Math.exp((value - maximum) / Math.max(.05, temperature)) : 0);
  const total = scaled.reduce((sum, value) => sum + value, 0) || 1;
  return scaled.map(value => value / total);
}

function foundationEntropy(probabilities) {
  return probabilities.reduce((sum, probability) => probability > 0 ? sum - probability * Math.log2(probability) : sum, 0);
}

function foundationCosine(a, b) {
  const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
  const lengthA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
  const lengthB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
  return lengthA && lengthB ? dot / (lengthA * lengthB) : 0;
}

function foundationNormalValues(seed, count) {
  const random = seededRandom(seed), values = [];
  while (values.length < count) {
    const first = Math.max(1e-8, random()), second = random();
    const magnitude = Math.sqrt(-2 * Math.log(first));
    values.push(magnitude * Math.cos(2 * Math.PI * second));
    if (values.length < count) values.push(magnitude * Math.sin(2 * Math.PI * second));
  }
  return values;
}

function diffusionPattern(name) {
  const size = 8;
  return Array.from({length: size}, (_, row) => Array.from({length: size}, (_, column) => {
    if (name === "box") return row === 1 || row === 6 || column === 1 || column === 6 ? 1 : -1;
    if (name === "checker") return (row + column) % 2 ? 1 : -1;
    if (name === "diamond") return Math.abs(row - 3.5) + Math.abs(column - 3.5) <= 3 ? 1 : -1;
    return row === 3 || row === 4 || column === 3 || column === 4 ? 1 : -1;
  }));
}

function diffusionSignal(step, schedule) {
  const fraction = step / 10;
  if (schedule === "cosine") return Math.max(.025, Math.cos(fraction * Math.PI / 2) ** 2);
  if (schedule === "gentle") return Math.max(.06, 1 - .94 * fraction ** 1.7);
  return Math.max(.05, 1 - .95 * fraction);
}

function setupDiffusion() {
  setConcept(CONCEPT_META.diffusion, "Noising and denoising", "A short diffusion trajectory", "Use the buttons, slider, or arrow keys to move through noise time. The prepared estimate stands in for a learned denoising network.", "<strong>Experiment</strong><p>Add noise until the pattern is difficult to see, then increase guidance and compare the estimated clean signal.</p>");
  conceptElements.controls.innerHTML = selectControl("diffPattern", "Target pattern", [["plus", "Plus"], ["box", "Box"], ["diamond", "Diamond"], ["checker", "Checkerboard"]]) + selectControl("diffSchedule", "Noise schedule", [["linear", "Linear"], ["cosine", "Cosine"], ["gentle", "Gentle start"]]) + rangeControl("diffStep", "Noise time", "t", 0, 10, 1, 4) + rangeControl("diffGuidance", "Denoising guidance", "g", 0, 2, .1, 1);
  conceptElements.actions.innerHTML = '<button class="button primary" id="diffNoise" type="button">Add noise</button><button class="button" id="diffDenoise" type="button">Denoise</button><button class="button" id="diffAuto" type="button">Animate</button><button class="button quiet" id="diffNew" type="button">New noise</button>';
  legend([{label: "Positive signal", color: conceptPalette.purple}, {label: "Negative signal", color: "#e7eef3"}, {label: "Current time", color: conceptPalette.red}]);
  setExplanation('<p>Forward diffusion gradually mixes a clean signal with random noise. A reverse model predicts the noise component at each time and removes a small amount repeatedly.</p><div class="try-card"><strong>Try this</strong><p>Compare the linear and cosine schedules at t = 5. The same step number does not necessarily preserve the same amount of signal.</p></div>', '<div class="formula compact">xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε &nbsp; · &nbsp; x̂₀ = (xₜ − √(1−ᾱₜ) ε̂) / √ᾱₜ</div><p id="diffMath"></p>', '<ul class="plain-list"><li>The forward process is fixed; the difficult part is learning to predict noise during reversal.</li><li>A schedule controls how quickly signal is destroyed across time.</li><li>Generation begins from noise and applies many small reverse updates.</li><li>This demo uses known noise plus a controlled error. A real diffusion network learns its estimate from data.</li></ul>');
  conceptState = {pattern: "plus", schedule: "linear", step: 4, guidance: 1, seed: 1201, direction: 1};

  const drawGrid = (context, matrix, x, y, cell, label) => {
    context.fillStyle = conceptPalette.ink;
    context.font = "800 12px system-ui";
    context.textAlign = "left";
    context.fillText(label, x, y - 11);
    matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
      const normalized = clamp((value + 1) / 2, 0, 1);
      context.fillStyle = `rgb(${Math.round(237 - 91 * normalized)},${Math.round(242 - 112 * normalized)},${Math.round(247 - 62 * normalized)})`;
      context.fillRect(x + columnIndex * cell, y + rowIndex * cell, cell, cell);
      context.strokeStyle = "#c8d4dc";
      context.lineWidth = 1;
      context.strokeRect(x + columnIndex * cell, y + rowIndex * cell, cell, cell);
    }));
  };
  const setStep = step => {
    conceptState.step = clamp(step, 0, 10);
    $("#diffStep").value = conceptState.step;
    $("#diffStepOut").textContent = String(conceptState.step);
    conceptRender();
  };
  const advance = () => {
    if (conceptState.step >= 10) conceptState.direction = -1;
    if (conceptState.step <= 0) conceptState.direction = 1;
    setStep(conceptState.step + conceptState.direction);
  };

  conceptRender = () => {
    const clean = diffusionPattern(conceptState.pattern), alpha = diffusionSignal(conceptState.step, conceptState.schedule), noise = foundationNormalValues(conceptState.seed, 64), modelError = foundationNormalValues(conceptState.seed + 97, 64);
    const noisy = [], estimate = [];
    const accuracy = clamp(.58 + .18 * conceptState.guidance - .12 * (1 - alpha), .3, .94);
    let noiseError = 0;
    for (let row = 0; row < 8; row++) {
      noisy[row] = [];
      estimate[row] = [];
      for (let column = 0; column < 8; column++) {
        const index = row * 8 + column, epsilon = noise[index], predicted = accuracy * epsilon + (1 - accuracy) * modelError[index];
        const value = Math.sqrt(alpha) * clean[row][column] + Math.sqrt(1 - alpha) * epsilon;
        noisy[row][column] = value;
        estimate[row][column] = alpha > .001 ? (value - Math.sqrt(1 - alpha) * predicted) / Math.sqrt(alpha) : 0;
        noiseError += (epsilon - predicted) ** 2;
      }
    }
    const size = conceptSize(), context = conceptContext, cell = Math.min(27, size.width * .25 / 8, size.height * .48 / 8), gridY = 88, cleanX = 20, noisyX = size.width / 2 - cell * 4, estimateX = size.width - cell * 8 - 20;
    context.clearRect(0, 0, size.width, size.height);
    context.fillStyle = "#f8fbfd";
    context.fillRect(0, 0, size.width, size.height);
    drawGrid(context, clean, cleanX, gridY, cell, "CLEAN SIGNAL x₀");
    drawGrid(context, noisy, noisyX, gridY, cell, `NOISY STATE x${conceptState.step}`);
    drawGrid(context, estimate, estimateX, gridY, cell, "DENOISING ESTIMATE x̂₀");
    const arrowY = gridY + cell * 4;
    [[cleanX + cell * 8 + 7, noisyX - 7], [noisyX + cell * 8 + 7, estimateX - 7]].forEach(([start, end]) => {
      context.beginPath(); context.moveTo(start, arrowY); context.lineTo(end, arrowY); context.strokeStyle = "#8095a6"; context.lineWidth = 2; context.stroke();
    });
    const chart = {x: 48, y: gridY + cell * 8 + 58, w: size.width - 78, h: Math.max(55, size.height - (gridY + cell * 8 + 85))};
    context.strokeStyle = "#c8d5de"; context.strokeRect(chart.x, chart.y, chart.w, chart.h);
    context.fillStyle = conceptPalette.muted; context.font = "800 11px system-ui"; context.fillText("SIGNAL RETAINED ACROSS NOISE TIME", chart.x, chart.y - 11);
    context.beginPath();
    for (let step = 0; step <= 10; step++) {
      const x = chart.x + step / 10 * chart.w, y = chart.y + (1 - diffusionSignal(step, conceptState.schedule)) * chart.h;
      step ? context.lineTo(x, y) : context.moveTo(x, y);
    }
    context.strokeStyle = conceptPalette.purple; context.lineWidth = 3; context.stroke();
    const markerX = chart.x + conceptState.step / 10 * chart.w, markerY = chart.y + (1 - alpha) * chart.h;
    context.beginPath(); context.arc(markerX, markerY, 6, 0, Math.PI * 2); context.fillStyle = conceptPalette.red; context.fill();
    const snr = alpha / Math.max(.001, 1 - alpha);
    metricCards([{label: "Noise time", value: `${conceptState.step} / 10`, note: "Position in trajectory"}, {label: "Signal retained", value: `${Math.round(alpha * 100)}%`, note: "Cumulative ᾱ"}, {label: "Signal-to-noise", value: fmt(snr, 2), note: "ᾱ / (1 − ᾱ)"}, {label: "Noise prediction MSE", value: fmt(noiseError / 64, 3), note: "Prepared denoiser error"}]);
    const math = $("#diffMath");
    if (math) math.innerHTML = `At t = <b>${conceptState.step}</b>, √ᾱ = <b>${fmt(Math.sqrt(alpha), 3)}</b> and √(1−ᾱ) = <b>${fmt(Math.sqrt(1 - alpha), 3)}</b>. Guidance gives the toy denoiser an accuracy factor of <b>${fmt(accuracy, 2)}</b>.`;
  };

  $("#diffPattern").addEventListener("change", event => {conceptState.pattern = event.target.value; conceptRender();});
  $("#diffSchedule").addEventListener("change", event => {conceptState.schedule = event.target.value; conceptRender();});
  bindRange("diffStep", value => conceptState.step = value, 0);
  bindRange("diffGuidance", value => conceptState.guidance = value, 1);
  $("#diffNoise").addEventListener("click", () => {stopConceptAnimation(); setStep(conceptState.step + 1);});
  $("#diffDenoise").addEventListener("click", () => {stopConceptAnimation(); setStep(conceptState.step - 1);});
  $("#diffAuto").addEventListener("click", () => {if (conceptTimer) {stopConceptAnimation(); return;} $("#diffAuto").textContent = "Pause"; conceptTimer = setInterval(advance, window.labDelay ? window.labDelay(520) : 520);});
  $("#diffNew").addEventListener("click", () => {conceptState.seed++; conceptRender();});
  conceptKeyDown = event => {if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return; event.preventDefault(); stopConceptAnimation(); setStep(conceptState.step + (event.key === 'ArrowRight' ? 1 : -1));};
  conceptCanvas.setAttribute("aria-label", "Diffusion noising and denoising trajectory. Use left and right arrows to change noise time.");
  conceptRender();
}

const ATTENTION_SEQUENCES = {
  shapes: {
    label: "Colours and shapes",
    tokens: ["the", "red", "circle", "touches", "the", "blue", "square"],
    vectors: [[0,0,0,.7], [1,0,0,0], [.65,.8,0,0], [0,0,1,0], [0,0,0,.7], [-1,0,0,0], [-.65,.8,0,0]],
    roles: ["det", "modifier", "noun", "verb", "det", "modifier", "noun"]
  },
  animals: {
    label: "Simple action",
    tokens: ["the", "small", "cat", "chased", "the", "mouse"],
    vectors: [[0,0,0,.7], [0,.3,0,0], [.2,.8,0,0], [0,0,1,0], [0,0,0,.7], [.25,.8,0,0]],
    roles: ["det", "modifier", "noun", "verb", "det", "noun"]
  },
  sequence: {
    label: "Ordered symbols",
    tokens: ["A", "B", "A", "C", "B", "C"],
    vectors: [[1,0,0,0], [0,1,0,0], [1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,1,0]],
    roles: ["symbol", "symbol", "symbol", "symbol", "symbol", "symbol"]
  }
};

function attentionRawScore(data, query, key, head) {
  if (head === "local") return 2.2 - Math.abs(query - key) * .8;
  if (head === "previous") return key === query - 1 ? 3 : -Math.abs(query - key) * .65;
  if (head === "syntax") {
    const queryRole = data.roles[query], keyRole = data.roles[key];
    if (queryRole === "noun" && key < query && ["modifier", "det"].includes(keyRole)) return 2.5 - Math.abs(query - key) * .25;
    if (queryRole === "modifier" && key > query && keyRole === "noun") return 2.4 - Math.abs(query - key) * .2;
    if (queryRole === "verb" && keyRole === "noun") return 1.9 - Math.abs(query - key) * .12;
    if (queryRole === "det" && key > query && keyRole === "noun") return 2.1 - Math.abs(query - key) * .2;
    return -.35 * Math.abs(query - key);
  }
  return data.vectors[query].reduce((sum, value, index) => sum + value * data.vectors[key][index], 0) * 2;
}

function setupTransformer() {
  setConcept(CONCEPT_META.transformer, "Attention routing", "Token connections and attention matrix", "Select a query token or click a token in the diagram. Line thickness and matrix colour show attention weight.", "<strong>Experiment</strong><p>Select <b>Colours and shapes</b>, focus on “circle,” and compare semantic, local, and syntax heads.</p>");
  conceptElements.controls.innerHTML = selectControl("attnSequence", "Token sequence", [["shapes", "Colours and shapes"], ["animals", "Simple action"], ["sequence", "Ordered symbols"]]) + '<label class="field-label" for="attnQuery">Query token</label><select id="attnQuery"></select>' + selectControl("attnHead", "Prepared attention head", [["semantic", "Semantic similarity"], ["local", "Nearby tokens"], ["syntax", "Syntactic relation"], ["previous", "Previous token"]]) + rangeControl("attnTemperature", "Softmax temperature", "τ", .2, 2, .1, .8) + '<div class="toggle-row"><label><input id="attnCausal" type="checkbox"> Causal mask: hide future tokens</label></div>';
  conceptElements.actions.innerHTML = '<button class="button primary" id="attnNext" type="button">Next query</button><button class="button" id="attnAuto" type="button">Animate focus</button><button class="button quiet" id="attnReset" type="button">Reset view</button>';
  legend([{label: "Query token", color: conceptPalette.red}, {label: "Strong attention", color: conceptPalette.blue}, {label: "Masked", color: "#d9e0e5"}]);
  setExplanation('<p>Attention compares a query from one token with keys from every allowed token. Softmax turns the scores into weights; those weights mix the value vectors into a context representation.</p><div class="try-card"><strong>Try this</strong><p>Lower the temperature to make attention selective. Turn on the causal mask and notice that a token can no longer use future information.</p></div>', '<div class="formula compact">score(q,k) = q·k / √d &nbsp; · &nbsp; attention = softmax(score / τ) &nbsp; · &nbsp; context = Σ weightᵢvᵢ</div><p id="attnMath"></p>', '<ul class="plain-list"><li>Queries ask what information is needed; keys advertise what each position contains.</li><li>Values carry the information that is actually mixed.</li><li>Different heads can learn different relationships in parallel.</li><li>Transformers add residual paths, normalization, and feed-forward layers around attention.</li><li>These prepared heads expose the mechanism; trained transformers learn their projections.</li></ul>');
  conceptState = {sequence: "shapes", query: 2, head: "semantic", temperature: .8, causal: false, tokenAreas: []};

  const updateQueryOptions = () => {
    const data = ATTENTION_SEQUENCES[conceptState.sequence], select = $("#attnQuery");
    select.innerHTML = data.tokens.map((token, index) => `<option value="${index}">${index + 1}: ${token}</option>`).join("");
    conceptState.query = clamp(conceptState.query, 0, data.tokens.length - 1);
    select.value = conceptState.query;
  };
  const rowWeights = (data, query) => {
    const scores = data.tokens.map((_, key) => conceptState.causal && key > query ? -Infinity : attentionRawScore(data, query, key, conceptState.head));
    return {scores, weights: foundationSoftmax(scores, conceptState.temperature)};
  };
  const setQuery = query => {const data = ATTENTION_SEQUENCES[conceptState.sequence]; conceptState.query = (query + data.tokens.length) % data.tokens.length; $("#attnQuery").value = conceptState.query; conceptRender();};

  conceptRender = () => {
    const data = ATTENTION_SEQUENCES[conceptState.sequence], current = rowWeights(data, conceptState.query), matrix = data.tokens.map((_, query) => rowWeights(data, query)), size = conceptSize(), context = conceptContext, n = data.tokens.length, left = 52, right = size.width - 22, spacing = (right - left) / Math.max(1, n - 1), tokenY = 56, boxW = Math.min(72, spacing - 7), boxH = 36;
    context.clearRect(0, 0, size.width, size.height); context.fillStyle = "#f8fbfd"; context.fillRect(0, 0, size.width, size.height);
    conceptState.tokenAreas = [];
    current.weights.forEach((weight, key) => {
      if (key === conceptState.query || weight < .002) return;
      const startX = left + conceptState.query * spacing, endX = left + key * spacing, controlY = 126 + Math.abs(key - conceptState.query) * 8;
      context.beginPath(); context.moveTo(startX, tokenY + boxH / 2); context.quadraticCurveTo((startX + endX) / 2, controlY, endX, tokenY + boxH / 2); context.strokeStyle = `rgba(36,110,170,${.16 + .78 * weight})`; context.lineWidth = 1 + 11 * weight; context.stroke();
    });
    data.tokens.forEach((token, index) => {
      const x = left + index * spacing - boxW / 2, active = index === conceptState.query;
      context.fillStyle = active ? "#fbe0dd" : "#e6f1f7"; context.fillRect(x, tokenY - boxH / 2, boxW, boxH);
      context.strokeStyle = active ? conceptPalette.red : conceptPalette.blue; context.lineWidth = active ? 3 : 1.5; context.strokeRect(x, tokenY - boxH / 2, boxW, boxH);
      context.fillStyle = conceptPalette.ink; context.font = "800 11px system-ui"; context.textAlign = "center"; context.fillText(token, x + boxW / 2, tokenY + 4);
      conceptState.tokenAreas.push({x, y: tokenY - boxH / 2, w: boxW, h: boxH, index});
    });
    const cell = Math.min(35, (size.width - 125) / n, (size.height - 235) / n), matrixX = Math.max(82, (size.width - n * cell) / 2), matrixY = 218;
    context.textAlign = "left"; context.fillStyle = conceptPalette.ink; context.font = "800 11px system-ui"; context.fillText("FULL ATTENTION MATRIX · ROWS ARE QUERIES", matrixX, matrixY - 31);
    data.tokens.forEach((token, index) => {context.fillStyle = conceptPalette.muted; context.font = "700 9px system-ui"; context.textAlign = "center"; context.fillText(token.slice(0, 6), matrixX + (index + .5) * cell, matrixY - 8); context.textAlign = "right"; context.fillText(token.slice(0, 7), matrixX - 7, matrixY + (index + .62) * cell);});
    matrix.forEach((row, query) => row.weights.forEach((weight, key) => {
      const masked = !Number.isFinite(row.scores[key]), intensity = Math.round(241 - weight * 125);
      context.fillStyle = masked ? "#d9e0e5" : `rgb(${intensity},${Math.round(244 - weight * 105)},${Math.round(248 - weight * 48)})`;
      context.fillRect(matrixX + key * cell, matrixY + query * cell, cell, cell); context.strokeStyle = "#fff"; context.lineWidth = 1; context.strokeRect(matrixX + key * cell, matrixY + query * cell, cell, cell);
      if (query === conceptState.query) {context.strokeStyle = conceptPalette.red; context.lineWidth = 2; context.strokeRect(matrixX + key * cell + 1, matrixY + query * cell + 1, cell - 2, cell - 2);}
    }));
    context.textAlign = "left";
    const strongest = current.weights.indexOf(Math.max(...current.weights)), raw = current.scores[strongest], entropy = foundationEntropy(current.weights), contextVector = data.vectors[0].map((_, dimension) => current.weights.reduce((sum, weight, index) => sum + weight * data.vectors[index][dimension], 0));
    metricCards([{label: "Strongest key", value: data.tokens[strongest], note: `Token ${strongest + 1}`}, {label: "Attention weight", value: `${Math.round(current.weights[strongest] * 100)}%`, note: "Largest softmax value"}, {label: "Raw score", value: fmt(raw, 2), note: "Before softmax"}, {label: "Attention entropy", value: fmt(entropy, 2), note: "Bits of spread"}]);
    const math = $("#attnMath");
    if (math) math.innerHTML = `For query <b>${data.tokens[conceptState.query]}</b>, the strongest key is <b>${data.tokens[strongest]}</b>: score ${fmt(raw, 2)} becomes weight <b>${fmt(current.weights[strongest], 3)}</b>. The mixed context begins [${contextVector.slice(0, 3).map(value => fmt(value, 2)).join(", ")}].`;
  };

  $("#attnSequence").addEventListener("change", event => {stopConceptAnimation(); conceptState.sequence = event.target.value; conceptState.query = Math.min(2, ATTENTION_SEQUENCES[conceptState.sequence].tokens.length - 1); updateQueryOptions(); conceptRender();});
  $("#attnQuery").addEventListener("change", event => {conceptState.query = Number(event.target.value); conceptRender();});
  $("#attnHead").addEventListener("change", event => {conceptState.head = event.target.value; conceptRender();});
  bindRange("attnTemperature", value => conceptState.temperature = value, 1);
  $("#attnCausal").addEventListener("change", event => {conceptState.causal = event.target.checked; conceptRender();});
  $("#attnNext").addEventListener("click", () => {stopConceptAnimation(); setQuery(conceptState.query + 1);});
  $("#attnAuto").addEventListener("click", () => {if (conceptTimer) {stopConceptAnimation(); return;} $("#attnAuto").textContent = "Pause"; conceptTimer = setInterval(() => setQuery(conceptState.query + 1), window.labDelay ? window.labDelay(720) : 720);});
  $("#attnReset").addEventListener("click", () => {stopConceptAnimation(); conceptState.query = Math.min(2, ATTENTION_SEQUENCES[conceptState.sequence].tokens.length - 1); conceptState.head = "semantic"; conceptState.temperature = .8; conceptState.causal = false; $("#attnHead").value = "semantic"; $("#attnTemperature").value = .8; $("#attnTemperatureOut").textContent = "0.8"; $("#attnCausal").checked = false; updateQueryOptions(); conceptRender();});
  conceptPointerDown = event => {const position = conceptPosition(event), area = conceptState.tokenAreas.find(item => position.x >= item.x && position.x <= item.x + item.w && position.y >= item.y && position.y <= item.y + item.h); if (area) {stopConceptAnimation(); setQuery(area.index);}};
  conceptKeyDown = event => {if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return; event.preventDefault(); stopConceptAnimation(); setQuery(conceptState.query + (event.key === 'ArrowRight' ? 1 : -1));};
  conceptCanvas.style.cursor = "pointer";
  updateQueryOptions(); conceptRender();
}

const LLM_PROMPTS = {
  story: {label: "the small …", tokens: ["the", "small"]},
  colour: {label: "a red …", tokens: ["a", "red"]},
  animal: {label: "the cat …", tokens: ["the", "cat"]}
};

const LLM_TRANSITIONS = {
  "the": {small: 2.2, red: 1.7, blue: 1.5, cat: 1.4, robot: 1.2},
  "a": {red: 2.3, blue: 2, small: 1.2, quiet: .8},
  "small": {robot: 2.7, cat: 2.1, mouse: 1.7, circle: .8},
  "red": {circle: 2.8, square: 2.3, robot: 1.2},
  "blue": {square: 2.8, circle: 2, robot: 1},
  "cat": {follows: 2.5, finds: 2, moves: 1.5},
  "mouse": {moves: 2.2, follows: 1.5, ".": 1.2},
  "robot": {finds: 2.6, follows: 2, moves: 1.7},
  "circle": {moves: 2.1, ".": 1.8},
  "square": {moves: 1.9, ".": 2},
  "finds": {a: 2.5, the: 2.1},
  "follows": {the: 2.6, a: 1.7},
  "moves": {quietly: 2.8, ".": 1.6},
  "quietly": {".": 3.2},
  ".": {the: 1.7, a: 1.5}
};

const LLM_PAIR_TRANSITIONS = {
  "the small": {robot: 3.2, cat: 2.2, mouse: 1.5},
  "a red": {circle: 3.3, square: 2.5, robot: 1},
  "the cat": {follows: 3, finds: 2.2, moves: 1.2},
  "finds a": {red: 2.8, blue: 2.5, small: 1.2},
  "follows the": {red: 2.2, small: 2, blue: 1.8},
  "moves quietly": {".": 3.5}
};

function llmDistribution(history, contextWindow, temperature, topK, repetition) {
  const context = history.slice(-contextWindow), last = context[context.length - 1], pair = context.length >= 2 ? context.slice(-2).join(" ") : "", source = LLM_PAIR_TRANSITIONS[pair] || LLM_TRANSITIONS[last] || LLM_TRANSITIONS["."];
  const entries = Object.entries(source).map(([token, logit]) => ({token, logit: logit - (context.includes(token) ? repetition : 0)})).sort((a, b) => b.logit - a.logit);
  const kept = entries.slice(0, Math.min(topK, entries.length)), probabilities = foundationSoftmax(kept.map(entry => entry.logit), temperature);
  return kept.map((entry, index) => ({...entry, probability: probabilities[index]}));
}

function setupLlm() {
  setConcept(CONCEPT_META.llm, "Autoregressive decoding", "Context, probabilities, and chosen tokens", "Generate one token at a time or click any probability bar to choose that token directly.", "<strong>Experiment</strong><p>Generate the same prompt greedily, then restart and use sampling with a higher temperature. Compare the paths.</p>");
  conceptElements.controls.innerHTML = selectControl("llmPrompt", "Prepared prompt", [["story", "the small …"], ["colour", "a red …"], ["animal", "the cat …"]]) + selectControl("llmDecode", "Decoding rule", [["greedy", "Greedy: choose highest"], ["sample", "Sample from distribution"]]) + rangeControl("llmTemperature", "Temperature", "τ", .2, 2, .1, .8) + rangeControl("llmTopK", "Top-k candidates", "k", 1, 6, 1, 4) + rangeControl("llmContext", "Context window", "C", 1, 6, 1, 4) + rangeControl("llmRepeat", "Repetition penalty", "r", 0, 1.5, .1, .4);
  conceptElements.actions.innerHTML = '<button class="button primary" id="llmNext" type="button">Generate one token</button><button class="button" id="llmAuto" type="button">Auto-generate</button><button class="button" id="llmUndo" type="button">Undo token</button><button class="button quiet" id="llmReset" type="button">Restart prompt</button>';
  legend([{label: "Active context", color: conceptPalette.blue}, {label: "Generated token", color: conceptPalette.green}, {label: "Probability", color: conceptPalette.orange}]);
  setExplanation('<p>A language model repeatedly predicts a probability distribution for the next token, chooses one token, appends it to the context, and repeats. Decoding settings change the choice without changing the model scores.</p><div class="try-card"><strong>Try this</strong><p>Set top-k to 1, then to 4. Raise temperature and use sampling to make lower-ranked choices more likely.</p></div>', '<div class="formula compact">P(next token | context) = softmax(logits / τ) &nbsp; → &nbsp; choose &nbsp; → &nbsp; append</div><p id="llmMath"></p>', '<ul class="plain-list"><li>Low temperature sharpens the distribution; high temperature flattens it.</li><li>Top-k removes all but the k highest-scoring candidates before sampling.</li><li>The context window limits which earlier tokens can influence the next prediction.</li><li>Generation is autoregressive: every chosen token changes later probabilities.</li><li>This is a tiny prepared token model, not an LLM. It exposes decoding without an API or pretrained weights.</li></ul>');
  conceptState = {prompt: "story", history: [...LLM_PROMPTS.story.tokens], promptLength: 2, decode: "greedy", temperature: .8, topK: 4, contextWindow: 4, repetition: .4, seed: 1501, barAreas: []};

  const reset = () => {stopConceptAnimation(); conceptState.history = [...LLM_PROMPTS[conceptState.prompt].tokens]; conceptState.promptLength = conceptState.history.length; conceptState.seed = 1501; conceptRender();};
  const choose = token => {if (conceptState.history.length >= 15) conceptState.history.splice(conceptState.promptLength, 1); conceptState.history.push(token); conceptRender();};
  const generate = () => {
    const distribution = llmDistribution(conceptState.history, conceptState.contextWindow, conceptState.temperature, conceptState.topK, conceptState.repetition);
    let selected = distribution[0].token;
    if (conceptState.decode === "sample") {
      const random = seededRandom(conceptState.seed++)(), target = random;
      let cumulative = 0;
      for (const candidate of distribution) {cumulative += candidate.probability; if (target <= cumulative) {selected = candidate.token; break;}}
    }
    choose(selected);
  };

  conceptRender = () => {
    const distribution = llmDistribution(conceptState.history, conceptState.contextWindow, conceptState.temperature, conceptState.topK, conceptState.repetition), size = conceptSize(), context = conceptContext, activeStart = Math.max(0, conceptState.history.length - conceptState.contextWindow);
    context.clearRect(0, 0, size.width, size.height); context.fillStyle = "#f8fbfd"; context.fillRect(0, 0, size.width, size.height);
    context.fillStyle = conceptPalette.ink; context.font = "800 11px system-ui"; context.fillText("TOKEN SEQUENCE · SHADED TOKENS ARE IN CONTEXT", 24, 34);
    let x = 24, y = 58;
    conceptState.history.forEach((token, index) => {
      const width = Math.max(35, context.measureText(token).width + 20);
      if (x + width > size.width - 24) {x = 24; y += 43;}
      context.fillStyle = index >= activeStart ? (index < conceptState.promptLength ? "#dcebf5" : "#dff1e8") : "#edf1f4";
      context.fillRect(x, y, width, 30); context.strokeStyle = index >= activeStart ? conceptPalette.blue : "#c7d2da"; context.lineWidth = 1.5; context.strokeRect(x, y, width, 30);
      context.fillStyle = index >= activeStart ? conceptPalette.ink : "#91a0ac"; context.font = "700 12px system-ui"; context.textAlign = "center"; context.fillText(token, x + width / 2, y + 20); x += width + 7;
    });
    const chartY = Math.max(175, y + 68), labelWidth = Math.min(115, size.width * .22), barX = 30 + labelWidth, barWidth = size.width - barX - 58, rowHeight = Math.min(44, (size.height - chartY - 45) / Math.max(1, distribution.length));
    context.textAlign = "left"; context.fillStyle = conceptPalette.ink; context.font = "800 11px system-ui"; context.fillText("NEXT-TOKEN DISTRIBUTION · CLICK A BAR TO CHOOSE", 24, chartY - 18);
    conceptState.barAreas = [];
    distribution.forEach((candidate, index) => {
      const rowY = chartY + index * rowHeight, height = Math.min(22, rowHeight - 7);
      context.fillStyle = conceptPalette.ink; context.font = "750 12px system-ui"; context.textAlign = "right"; context.fillText(candidate.token, barX - 10, rowY + height * .76);
      context.fillStyle = "#e5edf2"; context.fillRect(barX, rowY, barWidth, height); context.fillStyle = index === 0 ? conceptPalette.orange : conceptPalette.blue; context.fillRect(barX, rowY, barWidth * candidate.probability, height);
      context.fillStyle = conceptPalette.ink; context.textAlign = "left"; context.font = "700 11px system-ui"; context.fillText(`${Math.round(candidate.probability * 100)}%`, barX + barWidth * candidate.probability + 7, rowY + height * .76);
      conceptState.barAreas.push({x: barX, y: rowY, w: barWidth, h: height, token: candidate.token});
    });
    const top = distribution[0], entropy = foundationEntropy(distribution.map(candidate => candidate.probability)), contextTokens = conceptState.history.slice(-conceptState.contextWindow);
    metricCards([{label: "Most likely next", value: top.token, note: "Highest current logit"}, {label: "Top probability", value: `${Math.round(top.probability * 100)}%`, note: "After filtering"}, {label: "Distribution entropy", value: fmt(entropy, 2), note: "Bits of uncertainty"}, {label: "Context used", value: contextTokens.length, note: contextTokens.join(" ")}]);
    const math = $("#llmMath");
    if (math) math.innerHTML = `From context “<b>${contextTokens.join(" ")}</b>”, the toy model gives <b>${top.token}</b> logit ${fmt(top.logit, 2)} and probability <b>${fmt(top.probability, 3)}</b> after temperature and top-k.`;
  };

  $("#llmPrompt").addEventListener("change", event => {conceptState.prompt = event.target.value; reset();});
  $("#llmDecode").addEventListener("change", event => {conceptState.decode = event.target.value; conceptRender();});
  bindRange("llmTemperature", value => conceptState.temperature = value, 1);
  bindRange("llmTopK", value => conceptState.topK = value, 0);
  bindRange("llmContext", value => conceptState.contextWindow = value, 0);
  bindRange("llmRepeat", value => conceptState.repetition = value, 1);
  $("#llmNext").addEventListener("click", () => {stopConceptAnimation(); generate();});
  $("#llmAuto").addEventListener("click", () => {if (conceptTimer) {stopConceptAnimation(); return;} $("#llmAuto").textContent = "Pause"; conceptTimer = setInterval(generate, window.labDelay ? window.labDelay(680) : 680);});
  $("#llmUndo").addEventListener("click", () => {stopConceptAnimation(); if (conceptState.history.length > conceptState.promptLength) conceptState.history.pop(); conceptRender();});
  $("#llmReset").addEventListener("click", reset);
  conceptPointerDown = event => {const position = conceptPosition(event), area = conceptState.barAreas.find(item => position.x >= item.x && position.x <= item.x + item.w && position.y >= item.y && position.y <= item.y + item.h); if (area) {stopConceptAnimation(); choose(area.token);}};
  conceptCanvas.style.cursor = "pointer";
  conceptRender();
}

const VLM_CAPTIONS = [
  {text: "red circle on the left", vector: [1,0,0,0,1,0,-1,0,0]},
  {text: "blue square on the right", vector: [0,1,0,0,0,1,1,0,0]},
  {text: "green square near the top", vector: [0,0,1,0,0,1,0,-1,0]},
  {text: "yellow circle near the bottom", vector: [0,0,0,1,1,0,0,1,0]},
  {text: "alternating colour stripes", vector: [.5,.5,0,0,0,0,0,0,1]}
];

function vlmScene(name) {
  const size = 6, patches = Array.from({length: size * size}, (_, index) => ({row: Math.floor(index / size), column: index % size, color: "#edf2f5", vector: Array(9).fill(0)}));
  const paint = (row, column, color, type, stripe = 0) => {
    const patch = patches[row * size + column], x = column / (size - 1) * 2 - 1, y = row / (size - 1) * 2 - 1, colors = {red: [1,0,0,0], blue: [0,1,0,0], green: [0,0,1,0], yellow: [0,0,0,1]}, fills = {red: "#dd6b62", blue: "#4f8fc2", green: "#4fa47b", yellow: "#e1b94c"};
    patch.color = fills[color]; patch.vector = [...colors[color], type === "circle" ? 1 : 0, type === "square" ? 1 : 0, x, y, stripe];
  };
  if (name === "stacked") {
    [[1,2],[1,3],[2,2],[2,3]].forEach(([r,c]) => paint(r,c,"green","square"));
    [[4,2],[4,3],[5,2],[5,3]].forEach(([r,c]) => paint(r,c,"yellow","circle"));
  } else if (name === "stripes") {
    for (let row = 0; row < size; row++) for (let column = 0; column < size; column++) paint(row, column, column % 2 ? "blue" : "red", "square", 1);
  } else {
    [[2,0],[2,1],[3,0],[3,1]].forEach(([r,c]) => paint(r,c,"red","circle"));
    [[2,4],[2,5],[3,4],[3,5]].forEach(([r,c]) => paint(r,c,"blue","square"));
  }
  return patches;
}

function setupVlm() {
  setConcept(CONCEPT_META.vlm, "Cross-modal alignment", "Visual patches and candidate descriptions", "Click a visual patch to select it. Relevance borders show which patches best match the selected description.", "<strong>Experiment</strong><p>On the two-shape scene, compare full-image and selected-patch scope. Then mask a highly relevant patch.</p>");
  conceptElements.controls.innerHTML = selectControl("vlmScene", "Prepared visual", [["paired", "Two coloured shapes"], ["stacked", "Objects above and below"], ["stripes", "Alternating stripes"]]) + selectControl("vlmScope", "Visual representation", [["full", "Full image"], ["patch", "Selected patch"]]) + '<label class="field-label" for="vlmCaption">Target description</label><select id="vlmCaption">' + VLM_CAPTIONS.map((caption, index) => `<option value="${index}">${caption.text}</option>`).join("") + '</select>' + rangeControl("vlmTemperature", "Similarity temperature", "τ", .2, 2, .1, .7);
  conceptElements.actions.innerHTML = '<button class="button primary" id="vlmNext" type="button">Next patch</button><button class="button" id="vlmMask" type="button">Mask selected patch</button><button class="button quiet" id="vlmReset" type="button">Reset visual</button>';
  legend([{label: "Selected patch", color: conceptPalette.red}, {label: "Text relevance", color: conceptPalette.orange}, {label: "Masked patch", color: "#7f8c96"}]);
  setExplanation('<p>A vision–language model converts visual regions and text into comparable features. Similar image and text meanings should lie close together, so their cosine similarity is high.</p><div class="try-card"><strong>Try this</strong><p>Select “blue square on the right,” then click different patches. Watch patch relevance and description ranking change.</p></div>', '<div class="formula compact">similarity(image,text) = (v·t) / (‖v‖‖t‖) &nbsp; · &nbsp; match probability = softmax(similarity / τ)</div><p id="vlmMath"></p>', '<ul class="plain-list"><li>An image is divided into patches that become visual tokens.</li><li>A dual encoder compares pooled image and text embeddings efficiently.</li><li>Cross-attention models can let text tokens inspect individual visual tokens in more detail.</li><li>Masking a patch demonstrates how missing visual evidence changes alignment.</li><li>These handcrafted features are a transparent analogy; real VLMs learn high-dimensional representations.</li></ul>');
  conceptState = {scene: "paired", patches: vlmScene("paired"), selected: 18, scope: "full", caption: 0, temperature: .7, masked: new Set(), patchAreas: []};

  const patchScores = caption => conceptState.patches.map((patch, index) => conceptState.masked.has(index) ? -1 : foundationCosine(patch.vector, caption.vector));
  const captionScore = caption => {
    if (conceptState.scope === "patch") return conceptState.masked.has(conceptState.selected) ? -1 : foundationCosine(conceptState.patches[conceptState.selected].vector, caption.vector);
    const values = patchScores(caption).filter(value => value > -1).sort((a, b) => b - a), top = values.slice(0, 4);
    return top.length ? .7 * top[0] + .3 * top.reduce((sum, value) => sum + value, 0) / top.length : -1;
  };
  const moveSelection = amount => {conceptState.selected = (conceptState.selected + amount + conceptState.patches.length) % conceptState.patches.length; conceptRender();};

  conceptRender = () => {
    const rawScores = VLM_CAPTIONS.map(captionScore), probabilities = foundationSoftmax(rawScores, conceptState.temperature), relevance = patchScores(VLM_CAPTIONS[conceptState.caption]), size = conceptSize(), context = conceptContext, cell = Math.min(44, size.width * .36 / 6, size.height * .58 / 6), gridX = 24, gridY = 92, chartX = Math.max(gridX + cell * 6 + 42, size.width * .5), chartW = size.width - chartX - 24;
    context.clearRect(0, 0, size.width, size.height); context.fillStyle = "#f8fbfd"; context.fillRect(0, 0, size.width, size.height);
    context.fillStyle = conceptPalette.ink; context.font = "800 11px system-ui"; context.fillText("VISUAL PATCH TOKENS", gridX, gridY - 14);
    conceptState.patchAreas = [];
    conceptState.patches.forEach((patch, index) => {
      const x = gridX + patch.column * cell, y = gridY + patch.row * cell, masked = conceptState.masked.has(index), selected = index === conceptState.selected, score = Math.max(0, relevance[index]);
      context.fillStyle = masked ? "#7f8c96" : patch.color; context.fillRect(x, y, cell, cell);
      context.strokeStyle = selected ? conceptPalette.red : `rgba(226,120,53,${.18 + .8 * score})`; context.lineWidth = selected ? 4 : 1 + 5 * score; context.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
      if (masked) {context.beginPath(); context.moveTo(x + 5, y + 5); context.lineTo(x + cell - 5, y + cell - 5); context.moveTo(x + cell - 5, y + 5); context.lineTo(x + 5, y + cell - 5); context.strokeStyle = "#fff"; context.lineWidth = 2; context.stroke();}
      conceptState.patchAreas.push({x, y, w: cell, h: cell, index});
    });
    context.fillStyle = conceptPalette.ink; context.font = "800 11px system-ui"; context.fillText("TEXT CANDIDATES", chartX, gridY - 14);
    const rowHeight = Math.min(61, (size.height - gridY - 35) / VLM_CAPTIONS.length), barWidth = chartW;
    VLM_CAPTIONS.forEach((caption, index) => {
      const y = gridY + index * rowHeight, active = index === conceptState.caption;
      context.fillStyle = active ? "#fff0e5" : "#eef3f6"; context.fillRect(chartX, y, chartW, rowHeight - 9); context.strokeStyle = active ? conceptPalette.orange : "#d1dce4"; context.lineWidth = active ? 2 : 1; context.strokeRect(chartX, y, chartW, rowHeight - 9);
      context.fillStyle = conceptPalette.ink; context.font = "700 11px system-ui"; context.textAlign = "left"; const label = size.width < 600 ? caption.text.slice(0, 20) + (caption.text.length > 20 ? "…" : "") : caption.text; context.fillText(label, chartX + 8, y + 16);
      context.fillStyle = "#dce5eb"; context.fillRect(chartX + 8, y + 24, barWidth - 46, 9); context.fillStyle = active ? conceptPalette.orange : conceptPalette.blue; context.fillRect(chartX + 8, y + 24, (barWidth - 46) * probabilities[index], 9);
      context.fillStyle = conceptPalette.ink; context.font = "700 10px system-ui"; context.fillText(`${Math.round(probabilities[index] * 100)}%`, chartX + barWidth - 34, y + 32);
    });
    context.textAlign = "left";
    const best = probabilities.indexOf(Math.max(...probabilities)), selectedScore = rawScores[conceptState.caption], selectedPatchRelevance = relevance[conceptState.selected], sorted = [...probabilities].sort((a, b) => b - a), margin = sorted[0] - (sorted[1] || 0);
    metricCards([{label: "Target similarity", value: fmt(selectedScore, 3), note: VLM_CAPTIONS[conceptState.caption].text}, {label: "Best description", value: VLM_CAPTIONS[best].text, note: `${Math.round(probabilities[best] * 100)}% match probability`}, {label: "Selected-patch relevance", value: fmt(selectedPatchRelevance, 3), note: `Patch ${conceptState.selected + 1}`}, {label: "Ranking margin", value: fmt(margin, 3), note: "Best minus second"}]);
    const math = $("#vlmMath");
    if (math) math.innerHTML = `The current <b>${conceptState.scope === 'full' ? 'image' : 'patch'}</b> and “<b>${VLM_CAPTIONS[conceptState.caption].text}</b>” have cosine similarity <b>${fmt(selectedScore, 3)}</b>. After softmax, its match probability is <b>${fmt(probabilities[conceptState.caption], 3)}</b>.`;
  };

  $("#vlmScene").addEventListener("change", event => {conceptState.scene = event.target.value; conceptState.patches = vlmScene(conceptState.scene); conceptState.selected = conceptState.scene === "stacked" ? 8 : 18; conceptState.masked = new Set(); conceptRender();});
  $("#vlmScope").addEventListener("change", event => {conceptState.scope = event.target.value; conceptRender();});
  $("#vlmCaption").addEventListener("change", event => {conceptState.caption = Number(event.target.value); conceptRender();});
  bindRange("vlmTemperature", value => conceptState.temperature = value, 1);
  $("#vlmNext").addEventListener("click", () => moveSelection(1));
  $("#vlmMask").addEventListener("click", () => {if (conceptState.masked.has(conceptState.selected)) conceptState.masked.delete(conceptState.selected); else conceptState.masked.add(conceptState.selected); conceptRender();});
  $("#vlmReset").addEventListener("click", () => {conceptState.patches = vlmScene(conceptState.scene); conceptState.masked = new Set(); conceptRender();});
  conceptPointerDown = event => {const position = conceptPosition(event), area = conceptState.patchAreas.find(item => position.x >= item.x && position.x <= item.x + item.w && position.y >= item.y && position.y <= item.y + item.h); if (area) {conceptState.selected = area.index; conceptRender();}};
  conceptKeyDown = event => {if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return; event.preventDefault(); moveSelection(event.key === 'ArrowRight' ? 1 : -1);};
  conceptCanvas.style.cursor = "pointer";
  conceptRender();
}

Object.assign(CONCEPT_SETUPS, {
  diffusion: setupDiffusion,
  transformer: setupTransformer,
  llm: setupLlm,
  vlm: setupVlm
});

const requestedModule = new URLSearchParams(window.location.search).get("module");
if (requestedModule) {
  const requestedItem = MODULE_GROUPS.flatMap(group => group.items).find(item => item.id === requestedModule);
  if (requestedItem) openModule(requestedItem);
}

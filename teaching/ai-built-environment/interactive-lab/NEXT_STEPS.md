# Interactive ML & Deep Learning Lab — Next Steps

All 22 modules are live, including the four generative and foundation-model modules (diffusion, attention, LLMs, and vision–language models). Keep every module lightweight, browser-only, and free of paid or cloud APIs.

## Done (November 2026)

- Guided mode now uses one specific experiment per module (`PREDICTION_SCENARIOS` in `learning.js`). It records the named readout when the student reaches step 2 and reports the actual change against the prediction in step 3.
- The lab follows the operating-system reduce-motion setting until the student chooses otherwise.
- The redundant "Live" badges have been removed from the module list.

## Next

### 1. Built-environment context

Add civil-engineering presets alongside the abstract datasets:

- concrete strength vs. water–cement ratio for regression
- crack / no-crack features for logistic regression, KNN, and SVM
- bridge-deck condition ratings for decision trees
- an accelerometer record containing a damage event for the RNN
- crack and spalling pixel patterns for the CNN
- anomaly detection on vibration signatures for the autoencoder
- inspection captions such as "exposed rebar on pier cap" for the vision–language module

### 2. A module that links the two labs

Damage detection from natural-frequency shifts using the structural-dynamics models, or a small surrogate that learns spectral displacement.

### 3. Concept checks and progress

- Two or three auto-graded questions per module.
- Completion stored in the browser and shown as ticks in the module list.

### 4. Accessibility and responsive testing

- Complete keyboard-operation checks.
- Improve canvas text alternatives.
- On phones, keep the plot visible while a control is being adjusted.

### 5. Final validation

Re-run the scripted check of every guided experiment after any change to module controls or readout labels. A scenario fails quietly if its `metric` label no longer matches a metric card.

## Optional future modules

- Embeddings and similarity
- Generative adversarial networks
- Transfer learning
- Reinforcement learning

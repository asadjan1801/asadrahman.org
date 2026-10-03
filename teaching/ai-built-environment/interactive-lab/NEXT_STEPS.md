# Interactive ML & Deep Learning Lab — Next Steps

The current priority is to keep every module lightweight, browser-only, intuitive, and independent of paid or cloud APIs.

## Core remaining playgrounds

### 1. Diffusion Model Playground

- Visualize the forward noising process.
- Step through the reverse denoising process.
- Compare linear and cosine noise schedules.
- Compare predicted noise with the reconstructed sample.
- Support manual steps and lightweight animation.
- Demonstrate too few, sufficient, and excessive denoising steps.

### 2. Attention & Transformer Playground

- Provide an editable token sequence.
- Expose query, key, and value calculations.
- Visualize the attention-score matrix.
- Add softmax temperature and causal masking controls.
- Compare multiple attention heads.
- Introduce positional encoding.
- Show the simplified flow through a transformer block.

### 3. LLM Playground

- Demonstrate simple tokenization.
- Visualize context-window limitations.
- Show prepared next-token probabilities.
- Compare greedy, temperature, top-k, and top-p decoding.
- Generate text one token at a time.
- Demonstrate hallucination and loss of context.
- Keep the module fully local with no real LLM or external API.

### 4. Vision–Language Model Playground

- Divide an image into patches.
- Visualize visual and textual embeddings.
- Compare image–text similarity in a shared embedding space.
- Demonstrate image–caption matching.
- Visualize patch-to-token attention.
- Allow image regions to be masked.
- Include prepared zero-shot classification examples.
- Keep the module fully local with no real VLM inference or external API.

## Lab-wide refinement

### 5. Guided lesson improvements

- Add prediction prompts and short experiments.
- Add common-misconception checks.
- Maintain intuition, mathematics, and deeper-explanation tabs.
- Provide a recommended sequence through the modules.

### 6. Cross-module consistency

- Standardize controls, buttons, legends, and colour meanings.
- Keep reset, play, pause, and step behavior consistent.
- Review module descriptions and central questions.

### 7. Accessibility and responsive testing

- Complete keyboard-operation checks.
- Test phone, tablet, and desktop layouts.
- Preserve reduced-motion support.
- Improve canvas accessibility descriptions.
- Review text size, contrast, clipping, and horizontal overflow.

### 8. Performance pass

- Load modules only when useful.
- Stop animations when their module is inactive.
- Reduce unnecessary canvas calculations.
- Confirm that no large models, libraries, or media files slow the lab.
- Test on lower-powered devices.

### 9. Teaching-flow review

Recommended sequence:

1. Foundations
2. Classical machine learning
3. Neural networks
4. CNNs
5. RNNs and LSTMs
6. Autoencoders and latent spaces
7. Diffusion models
8. Attention and transformers
9. Large language models
10. Vision–language models

### 10. Final validation

- Test every control and preset.
- Review all guided-learning text.
- Verify desktop and mobile layouts.
- Confirm that no data leaves the browser.
- Update navigation and module descriptions.
- Publish the final revisions when they are ready.

## Optional future modules

- Embeddings and similarity
- Generative adversarial networks
- Transfer learning
- Reinforcement learning

Complete the core modules and refinement work before adding these optional topics.

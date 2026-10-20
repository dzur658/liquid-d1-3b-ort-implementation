## Minimal d1-3B-ONNX Implementation

Implements [LiquidAI's d1-3b model](https://huggingface.co/LiquidAI/d1-3B) in ONNX via [the ONNX community q4 quantization](https://huggingface.co/onnx-community/d1-3B-ONNX)

Meant as a reference for implementing this decision model in ORT.

NOTE: this implementation is text only and the vision encoder is purposefully not loaded.

#### Maximum Delta Observed

- Maximum Delta During 13 noul test cases: `0.0717592575385289`
- state: `The experiment was a complete success and the results were published. Unfortunately, the experiment failed to replicate.`
- instructions: `Did the experiment succeed?`
- PyTorch Float32 via LiquidAI's test code (P(yes)): `0.6713981924614711`
- Onnx-community Q4 via ORT (P(yes)): `0.74315745`
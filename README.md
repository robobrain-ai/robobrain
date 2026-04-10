# RoboBrain

**Open robot foundation model for manipulation.**

## LIBERO Benchmark Results

**Model:** pi0.5 fine-tuned (3.3B params, PaliGemma 2B + Gemma 300M action expert)  
**Training:** 30K steps on LIBERO demonstrations, 8×H100 FSDP  
**Evaluation:** 50 episodes per task, 10 tasks per suite, H100 JAX native serving

| Suite | Success Rate | Episodes |
|-------|-------------|----------|
| LIBERO-Spatial | **99.2%** | 500 |
| LIBERO-Object | **99.0%** | 500 |
| LIBERO-Goal | **96.0%** | 500 |
| LIBERO-Long-Horizon | **92.4%** | 500 |
| **Average** | **96.65%** | **2,000** |

### Cross-GPU Parity

| Platform | LIBERO-Spatial | Delta |
|----------|---------------|-------|
| H100 JAX (original) | 99.2% | — |
| A100 PyTorch (converted) | 99.2% | **0.0%** |

Zero performance loss across GPU architectures and frameworks.

## License

Apache 2.0

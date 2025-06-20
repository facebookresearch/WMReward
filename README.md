# Video Guidance

This project aims to leverage sampling guidance to utilize the prior knowledge learned in an SSL model (i.e., V-JEPA) to improve physics understanding and add additional control to a video diffusion model.


## Experiment

### exp 1
Vanilla sampling v.s. rejection sampling based on surprise score of V-JEPA

Model: Wan2.1
Benchmark: VideoPhy2

### exp 2
Add naive implementation of sampling guidance with suprise score signal
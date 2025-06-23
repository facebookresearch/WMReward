# Video Guidance

This project aims to leverage sampling guidance to utilize the prior knowledge learned in an SSL model (i.e., V-JEPA) to improve physics understanding and add additional control to a video diffusion model.


## Experiment

### 1
Vanilla sampling v.s. rejection sampling based on surprise score of V-JEPA2

Model: Wan2.1
Benchmark: VBench

| Model | subject_consistency | dynamic_degree | imaging_quality | aesthetic_quality | motion_smoothness | temporal_flickering | Semantic Adherence | Physics Commonsense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wan1.3b-rejection-v1 | 96.80 | 54.17 | 70.32 | 56.40 | 97.78 | 95.96 | 4.24 | 4.60 |
| wan1.3b-vanilla | 96.06 | 73.61 | 68.77 | 53.92 | 97.71 | 95.68 | 4.27 | 4.51 |
 
rejection sampling context window needs tuning

### 2
Add naive implementation of sampling guidance with suprise score signal
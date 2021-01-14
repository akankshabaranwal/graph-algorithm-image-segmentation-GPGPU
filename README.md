# Graph Based Image Segmentation on the GPU

Akanksha Baranwal, Amory Hoste, Gyorgy Rethy, Kouroche Bouchiat


## Installation Instructions
[Installation instructions](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/blob/master/installation.md)


## GPU Implementation branches

- Atomic Felzenszwalb Segmentation & Dynamic Parallelism [[CUDA]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/cuda-mst-naive)
- Ground Up DPP Segmentation Hierarchies [[CUDA]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/boruvka_fastMST_fixingSegments_v1)
- DPP Segmentation Hierarchies [[CUDA]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/fastmst_segment)
- DPP Superpixel Hierarchy [[CUDA]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/superpixel_gpu)


## Benchmarks

- Segmentation quality benchmark
  - [Dataset](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/benchmarking/dataset)
  - [ASA & UE score calculator](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/comparetool)
  - [Benchmark scripts](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/benchmarking)
  - [Data and plots](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/correct-benchmark-plots)
- Performance benchmark
  - [Dataset](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/performance_benchmark/dataset/jpg)
  - [Benchmark scripts](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/performance_benchmark/performance_benchmark)
  - [Benchmark data and plots](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/performance_benchmark/performance_plots)


## CPU Implementation branches

- Felzenszwalb [[Python]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/felzenszwalb_python) [[C++]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/felzenswlab_baseline)
- Felzenszwalb Boruvka [[Python]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/boruvka_sequential_python) [[C++]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/felzenszwalb_Boruvka_cpp)
- Fast Minimum Spanning Tree using Data Parallel Primitives [[Python]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/fastmst_python)
- DPP Segmentation Hierarchies [[Python]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/hierarchies_python)
- DPP Superpixel Hierarchy [[Python]](https://gitlab.ethz.ch/ahoste/graph-algorithm-image-segmentation/-/tree/superpixel_hierarchy)






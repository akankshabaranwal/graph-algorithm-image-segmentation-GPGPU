# Graph Based Image Segmentation on the GPU

Akanksha Baranwal, Amory Hoste, Gyorgy Rethy, Kouroche Bouchiat


## Installation Instructions
[Installation instructions](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/blob/master/installation.md)


## GPU Implementation branches

- Atomic Felzenszwalb Segmentation & Dynamic Parallelism [[CUDA]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/cuda-mst-naive)
- Ground Up DPP Segmentation Hierarchies [[CUDA]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/boruvka_fastMST_fixingSegments_v1)
- DPP Segmentation Hierarchies [[CUDA]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/fastmst_segment)
- DPP Superpixel Hierarchy [[CUDA]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/superpixel_gpu)


## Benchmarks

- Segmentation quality benchmark
  - [Dataset](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/benchmarking/dataset)
  - [ASA & UE score calculator](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/comparetool)
  - [Benchmark scripts](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/benchmarking)
  - [Benchmark data and plots](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/correct-benchmark-plots)
- Performance benchmark
  - [Dataset](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/performance_benchmark/dataset/jpg)
  - [Benchmark scripts](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/performance_benchmark/performance_benchmark)
  - [Benchmark data and plots](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/performance_benchmark/performance_plots)


## CPU Implementation branches

- Felzenszwalb [[Python]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/felzenszwalb_python) [[C++]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/felzenswlab_baseline)
- Felzenszwalb Boruvka [[Python]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/boruvka_sequential_python) [[C++]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/felzenszwalb_Boruvka_cpp)
- Fast Minimum Spanning Tree using Data Parallel Primitives [[Python]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/fastmst_python)
- DPP Segmentation Hierarchies [[Python]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/hierarchies_python)
- DPP Superpixel Hierarchy [[Python]](https://github.com/akankshabaranwal/graph-algorithm-image-segmentation-GPGPU/tree/superpixel_hierarchy)






#if WITH_CUDA
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <cstdio>

namespace reclib {
namespace opengl {
namespace cuda {

__global__ void fill_depth_nn_kernel(const float* depth_in, float* depth_out,
                                     int width, int height,
                                     float random_noise_scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = y * width + x;
  float val = depth_in[idx];

  if (val > 0) {
    depth_out[idx] = val;
    return;
  }

  // Simple nearest neighbor search in a local window
  float nearest_val = 0;
  int min_dist_sq = INT_MAX;
  int search_radius = 5;

  for (int dy = -search_radius; dy <= search_radius; dy++) {
    for (int dx = -search_radius; dx <= search_radius; dx++) {
      int nx = x + dx;
      int ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      float nval = depth_in[ny * width + nx];
      if (nval > 0) {
        int dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
          nearest_val = nval;
        }
      }
    }
  }

  depth_out[idx] = nearest_val;
}

void fill_depth_nearest_neighbor(const cv::cuda::GpuMat depth_in,
                                 cv::cuda::GpuMat depth_out,
                                 float random_noise_scale) {
  int width = depth_in.cols;
  int height = depth_in.rows;

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  fill_depth_nn_kernel<<<grid, block>>>(
      depth_in.ptr<float>(), depth_out.ptr<float>(), width, height,
      random_noise_scale);

  cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace opengl
}  // namespace reclib
#endif  // WITH_CUDA

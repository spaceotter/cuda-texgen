#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace cv;

struct BGR {
  uint8_t blue;
  uint8_t green;
  uint8_t red;
};

__device__ void pixel_func(int x, int y, BGR *out) {
  double t = pow(1.0 - fabs(sin((double)y / 20.0)), 1.5);
  uint8_t v = (uint8_t)(t * 255.0);
  out->blue = out->green = out->red = v;
}

__global__
void texture_kernel(int dim_x, int dim_y, BGR *out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  BGR *elt = out + (y * dim_x + x);
  pixel_func(x, y, elt);
}

int main(int argc, char** argv )
{
  if ( argc != 2 )
  {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }
  size_t dim = 512;
  Mat texture(dim, dim, CV_8UC3);

  for (int i = 0; i < texture.rows; i++) {
    for (int j = 0; j < texture.cols; j++) {
      BGR &p = texture.ptr<BGR>(i)[j];
      p.blue = p.green = p.red = 0;
    }
  }

  printf("total: %ld\n", texture.total());
  printf("elem: %ld\n", texture.elemSize());

  BGR *d_texture;
  size_t texture_bytes = texture.total() * texture.elemSize();
  cudaMalloc((void **)&d_texture, texture_bytes);
  // TODO this can probably be skipped
  cudaMemcpy(d_texture, texture.data, texture_bytes, cudaMemcpyHostToDevice);

  int thrd_x = 8;
  int thrd_y = 8;
  dim3 grid(ceil(dim/(double)thrd_x), ceil(dim/(double)thrd_y), 1);
  dim3 thrd(thrd_x, thrd_y, 1);

  texture_kernel<<<grid, thrd>>>(dim, dim, d_texture);

  cudaMemcpy(texture.data, d_texture, texture_bytes, cudaMemcpyDeviceToHost);

  // TODO write the final texture to texture memory and render directly with OpenGL
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", texture);
  imwrite(argv[1], texture);
  waitKey(0);
  return 0;
}

#pragma once

#include <stdint.h>

struct BGR {
  uint8_t blue;
  uint8_t green;
  uint8_t red;
};

template <class T>
struct v2 {
  T x;
  T y;

  __device__ v2(): x(0), y(0) {}
  __device__ v2(T x, T y): x(x), y(y) {}

  template <class U>
  __device__ v2<T> operator* (v2<U> other) {
    return {(T)(x * other.x), (T)(y * other.y)};
  }

  __device__ T dot(v2<T> other) {
    return x * other.x + y * other.y;
  }
};

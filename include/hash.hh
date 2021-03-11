// Copyright 2021 spaceotter

#pragma once

#include <stdint.h>
#include "util.hh"

__device__ uint64_t coord_hash(uint64_t seed, int x, int y);
__device__ v2<float> hash2unit(uint64_t hash);

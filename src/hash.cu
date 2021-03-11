// imitate python tuple hash
// https://github.com/python/cpython/blob/master/Objects/tupleobject.c#L348
// because I haven't found any good materials on writing hash functions yet.

#include <math.h>
#include "hash.hh"

static const uint64_t prime1 = 11400714785074694791UL;
static const uint64_t prime2 = 14029467366897019727UL;
static const uint64_t prime5 = 2870177450012600261UL;

#define ROT_HASH(x) (((x) << 31UL) | ((x) >> 33UL))

__device__
uint64_t coord_hash(uint64_t seed, int x, int y) {
  uint64_t r = prime5;
  r += seed * prime2;
  r = ROT_HASH(r);
  r *= prime1;

  r += x * prime2;
  r = ROT_HASH(r);
  r *= prime1;

  r += y * prime2;
  r = ROT_HASH(r);
  r *= prime1;

  r += 3 ^ prime5 ^ 3527539UL;
  return r;
}

__device__
v2<float> hash2unit(uint64_t hash) {
  float angle = (double)(hash >> 12UL) / 4096;
  return {cosf(angle), sinf(angle)};
}

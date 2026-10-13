#pragma once

/* The ABI form of a tensor view: plain C data shared by the program and its
 * plugins. A TensorView (C++) is this struct plus methods, with the same
 * memory layout, so views cross the plugin boundary without conversion. */

#include <stdint.h>

#define LITNICELM_MAX_TENSOR_RANK 6

typedef struct TensorViewStruct {
  void *ptr;          /* first element */
  uint32_t device_id; /* a Device value */
  uint32_t dtype_id;  /* a DType value */
  uint32_t ndim;      /* rank: number of valid entries in dims and strides */
  uint32_t reserved;  /* zero; keeps dims 8-byte aligned */
  int64_t dims[LITNICELM_MAX_TENSOR_RANK];
  int64_t strides_bytes[LITNICELM_MAX_TENSOR_RANK];
} TensorViewStruct;

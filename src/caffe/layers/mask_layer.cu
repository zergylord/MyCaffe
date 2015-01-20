// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdio.h>

using std::max;

namespace caffe {


//TODO:switch to -1 since its less likely to pop up naturally
template <typename Dtype>
__global__ void MaskForward(const int n, const Dtype* in,
    const Dtype* mask, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (mask[index] == -1.0) ? -1.0 : in[index] ;
  }
}
//takes two blobs, returns one. The second blob is the mask
//containing 1s for each element you want to keep and 0s elsewhere
template <typename Dtype>
Dtype MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    const Dtype* mask = bottom[1]->gpu_data();
    //caffe_gpu_rng_uniform(count, mask);
    // set thresholds (not used)
    // NOLINT_NEXT_LINE(whitespace/operators)
    //printf("pre mask");
    MaskForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, top_data);
    CUDA_POST_KERNEL_CHECK;
    //printf("post mask");
  } else {
    caffe_gpu_copy(count, bottom_data, top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count(); //should this be top?
    caffe_gpu_copy(count, top_diff, bottom_diff);//copy top grad to bottom
  }
}

INSTANTIATE_CLASS(MaskLayer);


}  // namespace caffe

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

template <typename Dtype>
Dtype QNetLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
    caffe_gpu_copy(count,bottom[1]->gpu_data(),difference_.mutable_gpu_data());
    caffe_gpu_add_scalar(count,Dtype(1),difference_.mutable_gpu_data());//turn -1.0 -> 0.0
    caffe_gpu_scal(count,Dtype(-1),difference_.mutable_gpu_data());//turn add->sub
    caffe_gpu_cond_add(count,
            bottom[0]->gpu_data(),//addend 1
            difference_.gpu_data(),//mask
            difference_.mutable_gpu_data());//addend 2 (and the storage place)
    caffe_gpu_cond_add_scalar(count,Dtype(1),difference_.gpu_data(),difference_.mutable_gpu_data());//undo the first +1
  //Dtype loss = 1337;
  Dtype loss = caffe_cpu_asum(count,difference_.cpu_data()) / 192;//32*6, so average per action
/*//  if(loss > 100000){
      printf("hello");
      for(int i = 0;i<192;i++){
        printf("%f ",bottom[0]->cpu_data()[i]);
        if (i % 6 == 0)
            printf("\n");
      }
//  }
  if(loss > 100000){
      printf("hello");
      for(int i = 0;i<192;i++){
        printf("%f ",difference_.cpu_data()[i]);
        if (difference_.cpu_data()[i] == 0)
            printf("yay!");
        if (i % 6 == 0)
            printf("\n");
      }
  }*/
 // printf("fuck %f %d",loss,num);
  return loss;
}

template <typename Dtype>
void QNetLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
/*  if(loss > 100000){
      printf("hello");
      for(int i = 0;i<192;i++){
        printf("%f ",difference_.cpu_data()[i]);
        if (i % 6 == 0)
            printf("\n");
      }
//  }*/
  //change to top count
  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
  if (propagate_down)
    caffe_gpu_copy(count,difference_.gpu_data(),(*bottom)[0]->mutable_gpu_diff());
  else
    printf("no propagate down!");
}

INSTANTIATE_CLASS(QNetLossLayer);


}  // namespace caffe

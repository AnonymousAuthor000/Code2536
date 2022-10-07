/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <cstring>
#include <memory>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace tcaqmi {

constexpr int kInputTensor = 0;
// constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteIntArray* GetOutputShape(TfLiteContext*, TfLiteNode*);

TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &input1));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteIntArray* output_size = nullptr;
  output_size = TfLiteIntArrayCopy(input1->dims);
  return context->ResizeTensor(context, output, output_size);
}

// inline TfLiteIntArray* GetOutputShapeFromTensor(TfLiteContext* context,
//                                                 TfLiteNode* node) {
//   const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
//   if (shape == nullptr) return nullptr;

//   TfLiteIntArray* output_shape = TfLiteIntArrayCreate(shape->dims->data[0]);
//   for (int i = 0; i < output_shape->size; ++i) {
//     output_shape->data[i] = shape->data.i32[i];
//   }

//   return output_shape;
// }

// inline TfLiteIntArray* GetOutputShapeFromParam(TfLiteContext* context,
//                                                TfLiteNode* node) {
//   auto* params = reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);

//   // The function is returned above this line if the shape tensor is usable.
//   // Now fallback to the shape parameter in `TfLiteReshapeParams`.
//   int num_dimensions = params->num_dimensions;
//   if (num_dimensions == 1 && params->shape[0] == 0) {
//     // Legacy tflite models use a shape parameter of [0] to indicate scalars,
//     // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
//     // toco conversion.
//     num_dimensions = 0;
//   }
//   TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
//   for (int i = 0; i < num_dimensions; ++i) {
//     output_shape->data[i] = params->shape[i];
//   }

//   return output_shape;
// }

// Check if the shape tensor is valid. Shapes should be int32 vectors.
// inline bool ShapeIsVector(TfLiteContext* context, TfLiteNode* node) {
//   const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
//   return (shape != nullptr && shape->dims->size == 1 &&
//           shape->type == kTfLiteInt32);
// }

// TfLiteIntArray* GetOutputShape(TfLiteContext* context, TfLiteNode* node) {
//   if (ShapeIsVector(context, node)) {
//     return GetOutputShapeFromTensor(context, node);
//   } else {
//     return GetOutputShapeFromParam(context, node);
//   }
// }

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
//   TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Always postpone sizing string tensors, even if we could in principle
  // calculate their shapes now. String tensors don't benefit from having their
  // shapes precalculated because the actual memory can only be allocated after
  // we know all the content.
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (output->type != kTfLiteString) {
    // if (IsConstantTensor(GetInput(context, node, kShapeTensor))) {
    //   TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
    // } else {
    //   SetTensorToDynamic(output);
    // }
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // There are two ways in which the 'output' can be made dynamic: it could be
  // a string tensor, or its shape cannot be calculated during Prepare(). In
  // either case, we now have all the information to calculate its shape.
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }

  // Note that string tensors are always "dynamic" in the sense that their size
  // is not known until we have all the content. This applies even when their
  // shape is known ahead of time. As a result, a string tensor is never given
  // any memory by ResizeOutput(), and we need to do it manually here. Since
  // reshape doesn't change the data, the output tensor needs exactly as many
  // bytes as the input tensor.
  if (output->type == kTfLiteString) {
    auto bytes_required = input->bytes;
    TfLiteTensorRealloc(bytes_required, output);
    output->bytes = bytes_required;
  }

  memcpy(output->data.raw, input->data.raw, input->bytes);

  return kTfLiteOk;
}

}  // namespace reshape

TfLiteRegistration* Register_tcaqmi() {
  static TfLiteRegistration r = {nullptr, nullptr, tcaqmi::Prepare,
                                 tcaqmi::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

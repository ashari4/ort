// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/framework/customregistry.h>
#include "ort_backends.h"
#include "ort_log.h"

#ifdef USE_MSNPU
  namespace onnxruntime {
    std::unique_ptr<onnxruntime::IExecutionProvider> CreateMSNPU_ExecutionProvider();
  }
  namespace MAIA
  {
    namespace OnnxRT
    {
      onnxruntime::OpKernel *CreateTransformerDecoder(const onnxruntime::OpKernelInfo& info);
    }
  }

  onnxruntime::KernelDefBuilder TransformerDecoderKernelDef() {
    onnxruntime::KernelDefBuilder def;
    def.SetName("TransformerDecoder")
        .SetDomain(onnxruntime::kMSDomain)
        .SinceVersion(1)
        .Provider("MAIA")
        .TypeConstraint("T", onnxruntime::DataTypeImpl::AllFixedSizeTensorTypes());
    return def;
  }
#endif

//use the environment from python module
namespace onnxruntime{
namespace python{
  onnxruntime::Environment& GetEnv();
}
}

namespace torch_ort {
namespace eager {

using namespace onnxruntime;

ORTBackendsManager& GetORTBackendsManager() {
  auto& env = onnxruntime::python::GetEnv();
  static ORTBackendsManager instance {env.GetLoggingManager()->DefaultLogger()};
  return instance;
}

onnxruntime::ORTInvoker& GetORTInvoker(const at::Device device) {
  return GetORTBackendsManager().GetInvoker(device);
}

onnxruntime::ORTInvoker& ORTBackendsManager::GetInvoker(const at::Device device) {
  ORT_LOG_FN(device);

  auto device_index = 0;
  if (device.has_index()) {
    device_index = device.index();
  }

  TORCH_CHECK(device.type() == at::DeviceType::ORT, "must be an ORT device");
  TORCH_CHECK(device_index >= 0, "must have a valid index");

  auto lookup = backends_.find(device.index());
  if (lookup != backends_.end()) {
    return *lookup->second;
  }

#ifdef USE_MSNPU
  auto ep = onnxruntime::CreateMSNPU_ExecutionProvider();

  // TODO: 0000 Register custom kernel 
  onnxruntime::CustomRegistry customRegistry;
  // auto def = TransformerDecoderKernelDef();
  // KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) -> OpKernel* { 
  //   return MAIA::OnnxRT::CreateTransformerDecoder(info);
  // };
  // auto status = customRegistry.RegisterCustomKernel(def, kernel_create_fn);
  // if(!status.IsOK())
  // {
  //     throw std::runtime_error("ORT return failure status:" + status.ErrorMessage());
  // }

  using namespace ONNX_NAMESPACE;
  std::vector<ONNX_NAMESPACE::OpSchema> schemas; 

  onnx::OpSchema schema("TransformerDecoder", "", 0);
  schema.SetDomain(onnxruntime::kMSDomain);
  schema.SinceVersion(1);
  schema.SetDoc("Single Layer of Custom Transformer TNLG for MAIA");
  schema.Input(0, "X_0", "input embeddings", "T");
  schema.Input(1, "X_1", "normalization weight", "T");
  schema.Input(2, "X_2", "normalization bias", "T");
  schema.Input(3, "X_3", "query weight", "T");
  schema.Input(4, "X_4", "query bias", "T");
  schema.Input(5, "X_5", "key weight", "T");
  schema.Input(6, "X_6", "key bias", "T");
  schema.Input(7, "X_7", "value weight", "T");
  schema.Input(8, "X_8", "value bias", "T");
  schema.Input(9, "X_9", "attention mask", "T");
  schema.Input(10, "X_10", "project weight", "T");
  schema.Input(11, "X_11", "project bias", "T");
  schema.Input(12, "X_12", "feedforward network weight", "T");
  schema.Input(13, "X_13", "feedforward network bias", "T");
  schema.Input(14, "X_14", "feedforward network weight", "T");
  schema.Input(15, "X_15", "feedforward network bias", "T");
  schema.Input(16, "X_16", "normalization weight", "T");
  schema.Input(17, "X_17", "normalizaton bias", "T");
  schema.Input(18, "X_18", "pad values", "T");
  schema.Output(0, "Y_0", "layer norm", "T");
  schema.Output(1, "Y_1", "std inv DMemAddr", "T");
  schema.Output(2, "Y_2", "shift DMemAddr", "T");
  schema.Output(3, "Y_3", "query DMem", "T");
  schema.Output(4, "Y_4", "key DMem", "T");
  schema.Output(5, "Y_5", "value DMem", "T");
  schema.Output(6, "Y_6", "softmax DMem", "T");
  schema.Output(7, "Y_7", "soft dropout DMem", "T");
  schema.Output(8, "Y_8", "soft dropout mask DMem", "T");
  schema.Output(9, "Y_9", "soft DMem", "T");
  schema.Output(10, "Y_10", "dense dropout mask DMem", "T");
  schema.Output(11, "Y_11", "post selfattention layer norm", "T");
  schema.Output(12, "Y_12", "normalization std inv DMemAddr", "T");
  schema.Output(13, "Y_13", "normalization shift DMemAddr", "T");
  schema.Output(14, "Y_14", "multilayer perceptron dense", "T");
  schema.Output(15, "Y_15", "multilayer perceptron gelu", "T");
  schema.Output(16, "Y_16", "multilayer perceptron dropout mask", "T");
  schema.Output(17, "Y_17", "residual connection", "T");
  schema.Attr("softDropoutProb", "softmax dropout probability", AttributeProto::FLOAT, 0.0f);
  schema.Attr("denseDropoutProb", "dense dropout probability", AttributeProto::FLOAT, 0.0f);
  schema.Attr("mlpDropoutProb", "MLP dropout probability", AttributeProto::FLOAT, 0.0f);
  schema.Attr("paddedHiddenSize", "padded hidden size", AttributeProto::INT, static_cast<int64_t>(1024));
  schema.Attr("headSize", "head size", AttributeProto::INT, static_cast<int64_t>(64));
  schema.Attr("softDropoutSeed", "softmax dropout seed", AttributeProto::INT, static_cast<int64_t>(111));
  schema.Attr("denseDropoutSeed", "dense layer dropout seed", AttributeProto::INT, static_cast<int64_t>(222));
  schema.Attr("mlpDropoutSeed", "mlp dropout seed", AttributeProto::INT, static_cast<int64_t>(333));
  schema.Attr("epsilon", "epsilon value", AttributeProto::FLOAT, 1e-5f);
  schema.Attr("numHeads", "number of heads", AttributeProto::INT, static_cast<int64_t>(8));
  schema.Attr("hiddenSize", "hidden size", AttributeProto::INT, static_cast<int64_t>(512));
  schema.TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float or bflat16 tensors.");

  // ONNX_OPERATOR_SCHEMA(TransformerDecoder)
  //     .SetDomain(kMSDomain)
  //     .SinceVersion(1)
  //     .SetDoc("Single Layer of Custom Transformer TNLG for MAIA")
  //     .Input(0, "X_0", "input embeddings", "T")
  //     .Input(1, "X_1", "normalization weight", "T")
  //     .Input(2, "X_2", "normalization bias", "T")
  //     .Input(3, "X_3", "query weight", "T")
  //     .Input(4, "X_4", "query bias", "T")
  //     .Input(5, "X_5", "key weight", "T")
  //     .Input(6, "X_6", "key bias", "T")
  //     .Input(7, "X_7", "value weight", "T")
  //     .Input(8, "X_8", "value bias", "T")
  //     .Input(9, "X_9", "attention mask", "T")
  //     .Input(10, "X_10", "project weight", "T")
  //     .Input(11, "X_11", "project bias", "T")
  //     .Input(12, "X_12", "feedforward network weight", "T")
  //     .Input(13, "X_13", "feedforward network bias", "T")
  //     .Input(14, "X_14", "feedforward network weight", "T")
  //     .Input(15, "X_15", "feedforward network bias", "T")
  //     .Input(16, "X_16", "normalization weight", "T")
  //     .Input(17, "X_17", "normalizaton bias", "T")
  //     .Input(18, "X_18", "pad values", "T")
  //     .Output(0, "Y_0", "layer norm", "T")
  //     .Output(1, "Y_1", "std inv DMemAddr", "T")
  //     .Output(2, "Y_2", "shift DMemAddr", "T")
  //     .Output(3, "Y_3", "query DMem", "T")
  //     .Output(4, "Y_4", "key DMem", "T")
  //     .Output(5, "Y_5", "value DMem", "T")
  //     .Output(6, "Y_6", "softmax DMem", "T")
  //     .Output(7, "Y_7", "soft dropout DMem", "T")
  //     .Output(8, "Y_8", "soft dropout mask DMem", "T")
  //     .Output(9, "Y_9", "soft DMem", "T")
  //     .Output(10, "Y_10", "dense dropout mask DMem", "T")
  //     .Output(11, "Y_11", "post selfattention layer norm", "T")
  //     .Output(12, "Y_12", "normalization std inv DMemAddr", "T")
  //     .Output(13, "Y_13", "normalization shift DMemAddr", "T")
  //     .Output(14, "Y_14", "multilayer perceptron dense", "T")
  //     .Output(15, "Y_15", "multilayer perceptron gelu", "T")
  //     .Output(16, "Y_16", "multilayer perceptron dropout mask", "T")
  //     .Output(17, "Y_17", "residual connection", "T")
  //     .Attr("softDropoutProb", "softmax dropout probability", AttributeProto::FLOAT, 0.0f)
  //     .Attr("denseDropoutProb", "dense dropout probability", AttributeProto::FLOAT, 0.0f)
  //     .Attr("mlpDropoutProb", "MLP dropout probability", AttributeProto::FLOAT, 0.0f)
  //     .Attr("paddedHiddenSize", "padded hidden size", AttributeProto::INT, static_cast<int64_t>(1024))
  //     .Attr("headSize", "head size", AttributeProto::INT, static_cast<int64_t>(64))
  //     .Attr("softDropoutSeed", "softmax dropout seed", AttributeProto::INT, static_cast<int64_t>(111))
  //     .Attr("denseDropoutSeed", "dense layer dropout seed", AttributeProto::INT, static_cast<int64_t>(222))
  //     .Attr("mlpDropoutSeed", "mlp dropout seed", AttributeProto::INT, static_cast<int64_t>(333))
  //     .Attr("epsilon", "epsilon value", AttributeProto::FLOAT, 1e-5f)
  //     .Attr("numHeads", "number of heads", AttributeProto::INT, static_cast<int64_t>(8))
  //     .Attr("hiddenSize", "hidden size", AttributeProto::INT, static_cast<int64_t>(512))
  //     .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float or bflat16 tensors.");

  schemas.push_back(schema);
  auto status = customRegistry.RegisterOpSet(schemas, onnxruntime::kMSDomain, 1, 2);
  if(!status.IsOK())
  {
      throw std::runtime_error("ORT return failure status:" + status.ErrorMessage());
  }
  custom_op_schema_.push_back(customRegistry.GetOpschemaRegistry());
#else
  auto ep = std::make_unique<onnxruntime::CPUExecutionProvider>(
    onnxruntime::CPUExecutionProviderInfo(false));
#endif
  
  auto invoker = 
    std::make_unique<onnxruntime::ORTInvoker>(
      std::move(ep),
      logger_,
      custom_op_schema_);

  backends_[device_index] = std::move(invoker);
  return *backends_[device_index];
}

} // namespace eager
} // namespace torch_ort
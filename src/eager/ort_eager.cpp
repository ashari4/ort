// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/extension.h>

#include "ort_backends.h"
#include "ort_log.h"
#include "msnpu_ops.h"

namespace torch_ort {
namespace eager {

PYBIND11_MODULE(torch_ort, torch_ort_module) {
  ORT_LOG_DEBUG << "pybind11 module init";

  torch_ort_module.def(
    "device",
    [](int device_index) {
      return py::cast<py::object>(
        THPDevice_New(at::Device(at::DeviceType::ORT, device_index)));
    },
    py::arg("device_index") = 0);

    auto msnpu_module = torch_ort_module.def_submodule("msnpu");
    msnpu_module.def("transformer_decoder", &torch_ort::eager::msnpu::transformer_decoder);
    msnpu_module.def("transformer_decoder_grad", &torch_ort::eager::msnpu::transformer_decoder_grad);
}

} // namespace eager
} // namespace torch_ort
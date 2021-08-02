#include <torch/extension.h>
#include <vector>

namespace torch_ort {
namespace eager {
namespace msnpu {

static const char * const TransformerDecoderName = "transformerdecoder";
static const char * const TransformerDecoderGradName = "transformerdecodergrad";

std::vector<at::Tensor> transformerdecoder(
    int64_t padded_hidden_size, int64_t head_size, float soft_dropout_prob,
    int64_t soft_dropout_seed, float dense_dropout_prob,
    int64_t dense_dropout_seed, float mlp_dropout_prob,
    int64_t mlp_dropout_seed, float epsilon,
    const torch::Tensor& embeddings_post_dropout,
    const torch::Tensor& normalization_1_w,
    const torch::Tensor& normalization_1_b, const torch::Tensor &query_w,
    const torch::Tensor& query_b, const torch::Tensor &key_w,
    const torch::Tensor& key_b, const torch::Tensor &value_w,
    const torch::Tensor& value_b, const torch::Tensor &attention_mask,
    const torch::Tensor& project_w, const torch::Tensor &project_b,
    const torch::Tensor& FFN1_w, const torch::Tensor &FFN1_b,
    const torch::Tensor& FFN2_w, const torch::Tensor &FFN2_b,
    const torch::Tensor& normalization_2_w,
    const torch::Tensor& normalization_2_b, const torch::Tensor &pad_values);

std::vector<at::Tensor> transformerdecodergrad(
    int64_t padded_hidden_size, int64_t head_size, int64_t num_heads, const torch::Tensor& norm_1_weight,
                                     const torch::Tensor& q_weight,
                                     const torch::Tensor& k_weight,
                                     const torch::Tensor& v_weight,
                                     const torch::Tensor& attention_mask,
                                     const torch::Tensor& dense_weight,
                                     const torch::Tensor& mlp_weight_0,
                                     const torch::Tensor& mlp_weight_1,
                                     const torch::Tensor& norm_2_weight,
                                     const torch::Tensor& pad_values,
                                     const torch::Tensor& grad_input,
                                     const torch::Tensor& norm_1_dmem,
                                     const torch::Tensor& norm_1_std_inv_dmem,
                                     const torch::Tensor& norm_1_shift_dmem,
                                     const torch::Tensor& q_dmem,
                                     const torch::Tensor& k_dmem,
                                     const torch::Tensor& v_dmem,
                                     const torch::Tensor& softmax_dmem,
                                     const torch::Tensor& soft_dropout_dmem,
                                     const torch::Tensor& soft_dropout_mask_dmem,
                                     const torch::Tensor& v_soft_dmem,
                                     const torch::Tensor& dense_dropout_mask_dmem,
                                     const torch::Tensor& norm_2_dmem,
                                     const torch::Tensor& norm_2_std_inv_dmem,
                                     const torch::Tensor& norm_2_shift_dmem,
                                     const torch::Tensor& mlp_hidden_1_dmem,
                                     const torch::Tensor& mlp_gelu_dmem,
                                     const torch::Tensor& mlp_dropout_mask_dmem);
}
} // namespace eager
} // namespace torch_ort
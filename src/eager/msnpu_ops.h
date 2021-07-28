#include <torch/extension.h>
#include <vector>

namespace torch_ort {
namespace eager {
namespace msnpu {

static const char * const TransformerDecoderName = "transformerdecoder";
static const char * const TransformerDecoderName = "transformerdecodergrad";

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
    int64_t padded_hidden_size, int64_t head_size, int64_t num_heads, torch::Tensor norm_1_weight,
                                     torch::Tensor q_weight,
                                     Tensor k_weight,
                                     Tensor v_weight,
                                     Tensor attention_mask,
                                     Tensor dense_weight,
                                     Tensor mlp_weight_0,
                                     Tensor mlp_weight_1,
                                     Tensor norm_2_weight,
                                     Tensor pad_values,
                                     Tensor grad_input,
                                     Tensor norm_1_dmem,
                                     Tensor norm_1_std_inv_dmem,
                                     Tensor norm_1_shift_dmem,
                                     Tensor q_dmem,
                                     Tensor k_dmem,
                                     Tensor v_dmem,
                                     Tensor softmax_dmem,
                                     Tensor soft_dropout_dmem,
                                     Tensor soft_dropout_mask_dmem,
                                     Tensor v_soft_dmem,
                                     Tensor dense_dropout_mask_dmem,
                                     Tensor norm_2_dmem,
                                     Tensor norm_2_std_inv_dmem,
                                     Tensor norm_2_shift_dmem,
                                     Tensor mlp_hidden_1_dmem,
                                     Tensor mlp_gelu_dmem,
                                     Tensor mlp_dropout_mask_dmem);
}
} // namespace eager
} // namespace torch_ort
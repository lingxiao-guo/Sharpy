#if HAS_DNN_MODULE
#include <reclib/dnn/dnn_utils.h>

namespace torch {
// In PyTorch 2.6+, None and Slice are already defined in at::indexing
// Re-export them into the torch namespace for backward compatibility
using at::indexing::None;
const torch::indexing::Slice All;

}  // namespace torch
#endif  // HAS_DNN_MODULE
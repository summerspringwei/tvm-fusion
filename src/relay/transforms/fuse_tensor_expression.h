#ifndef FUSE_TENSOR_EXPRESSION_H
#define FUSE_TENSOR_EXPRESSION_H

#include <tvm/te/tensor.h>
#include <tvm/relay/expr.h>
namespace tvm {
namespace relay {
namespace transform{
  Array<te::Tensor> FFusionTensorExpression(Array<te::Tensor> outputs, 
    std::unordered_map<te::Tensor, Expr, ObjectPtrHash, ObjectPtrEqual> tensor_constant_map);
}
}
}

#endif
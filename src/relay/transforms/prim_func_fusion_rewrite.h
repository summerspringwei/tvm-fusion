#ifndef PRIM_FUNC_FUSION_REWRITE_H
#define PRIM_FUNC_FUSION_REWRITE_H

#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {
  using ExprTEMap = std::unordered_map<relay::Expr, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual>;
  using TEExprMap = std::unordered_map<te::Tensor, relay::Expr, ObjectPtrHash, ObjectPtrEqual>;
  
  std::pair<Expr, Array<te::Tensor>> RewriteFusedPrimFunc(const Expr prim_func, ExprTEMap expr_te_map,
  TEExprMap te_expr_map, int32_t num_branch);
  void PrintTEGraph(te::Tensor tensor);
  Array<te::Tensor> FFusionTensorExpression(Array<te::Tensor> outputs);
}
}

#endif
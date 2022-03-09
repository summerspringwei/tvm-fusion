#ifndef REPLACE_TENSORS_IN_EXPR_STMT_H
#define REPLACE_TENSORS_IN_EXPR_STMT_H

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>
#include <tvm/te/operation.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/transform.h>

namespace tvm{
namespace tir{
namespace transforms{
  void FVerifyTensorConnect(const te::Tensor& tensor);
  std::unordered_set<te::Tensor> FGetTensorsFromStmtExpr(const tir::Stmt& stmt);
  Stmt FReplaceDataProducer(Stmt& stmt, std::unordered_map<te::Tensor, te::Tensor>& replace_map);
}
}
}

#endif
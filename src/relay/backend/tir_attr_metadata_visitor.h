
#ifndef TIR_ATTR_METADATA_VISITOR
#define TIR_ATTR_METADATA_VISITOR

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/debug.h>

namespace tvm {
namespace relay {

  Array<te::Tensor> GetOutputTensorsFromRelayFunc(const Expr& body);

  Map<GlobalVar, Array<te::Tensor>> GetPerVarTensorsFromIRModule(const IRModule& mod);

  void PrintVarTensorMap(const Map<GlobalVar, Array<te::Tensor>>& var_tensor_map);

}
}

#endif

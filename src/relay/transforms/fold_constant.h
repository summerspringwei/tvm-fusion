#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {
  Expr FoldConstant(const Expr& expr, const IRModule& mod);
}
}
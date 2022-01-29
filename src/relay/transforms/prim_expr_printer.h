

#ifndef TVM_RELAY_TRANSFORMS_PRIM_EXPR_PRINTER_H
#define TVM_RELAY_TRANSFORMS_PRIM_EXPR_PRINTER_H

#include <tvm/tir/expr.h>

namespace tvm {
namespace relay {

void PrintPrimExpr(const PrimExpr& expr);

}
}

#endif
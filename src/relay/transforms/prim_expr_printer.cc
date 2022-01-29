
/*!
 * \file prim_expr_printer.cc
 * \brief Utility to print the PrimExpr after lower and before schedule
 */
#include "prim_expr_printer.h"

#include <tvm/runtime/registry.h>
#include <tvm/tir/expr_functor.h>

#include <set>

namespace tvm {
namespace relay {
using namespace tir;

class PrimExprPrinter : public ExprVisitor{
  public:
    PrimExprPrinter(): depth_{0} {}

    void VisitExpr(const PrimExpr& node) final {
      if(visited_.count(node.get()) != 0) return;
      visited_.insert(node.get());

      depth_++;
      ExprVisitor::VisitExpr(node);
      
      depth_--;
    }

    // void VisitExpr_(const AddNode* op) final {
    //   os_ << "tir.add(" << op->a << ", " << op->b << ")";
    //   VisitExpr(op->a);
    //   VisitExpr(op->b);
    // }

    void VisitExpr_(const AddNode* op) final {
      os_ << "\ntir.add(\n";
      VisitExpr(op->a);
      os_ << ", \n";
      VisitExpr(op->b);
      os_ << ")\n";
    }

    void VisitExpr_(const MulNode* op) final {
      os_ << "\ntir.mul(\n";
      VisitExpr(op->a);
      os_ << ", \n";
      VisitExpr(op->b);
      os_ << ")\n";
    }

    void VisitExpr_(const SelectNode* op) final {
      os_ << "\ntir.select(condition\n";
      VisitExpr(op->condition);
      os_ << "true_value:\n";
      VisitExpr(op->true_value);
      os_ << "false_value:\n";
      VisitExpr(op->false_value);
      os_ << ")\n";
    }

    // void VisitExpr_(const MulNode* op) final {
    //   os_ << "tir.mul(" << op->a << ", " << op->b << ")";
    //   VisitExpr(op->a);
    //   VisitExpr(op->b);
    // }

    void VisitExpr_(const ReduceNode* op) final {
      os_ << "tir.reduce( reduce_axis: ";
      for(const auto& var: op->axis) {
        os_ << var;
      } os_ << " source: " << op->source << ", combiner: " << op->combiner << ", condition " 
        << op->condition << ", value_index: " << op->value_index << ")";
      for(auto& expr: op->source){
        VisitExpr(expr);
      }
    }

    void VisitExpr_(const VarNode* op) final {
      os_ << "tir.Var(" << op->name_hint << ")";
    }

    void VisitExpr_(const LoadNode* op) final {
      os_ << "tir.Load(" << op->buffer_var << "[" << op->index << "]" << ")";
    }

    void VisitExpr_(const BufferLoadNode* op) final {
      os_ << "tir.BufferLoad(" << op->buffer << "[\n";
      for(auto& expr: op->indices){
        VisitExpr(expr);
      }
      os_<<"]\n";
    }

    void VisitExpr_(const ProducerLoadNode* op) final {
      os_ << "tir.ProducerLoad(" << op->producer << "[\n";
      for(auto& expr: op->indices){
        VisitExpr(expr);
      }
      os_<<"]\n";
    }

    void VisitExpr_(const IntImmNode* op) final {
      os_ << "tir.IntImm(" << op->value << ")";
    }

    std::unordered_set<const Object*> visited_;
    std::ostringstream os_;
    size_t depth_;
};

void PrintPrimExpr(const PrimExpr& expr){
  auto printer = PrimExprPrinter();
  printer.VisitExpr(expr);
  LOG(INFO) << printer.os_.str();
}

// TVM_REGISTER_GLOBAL("transform.PrintPrimExpr")
//     .set_body_typed([](const PrimExpr& v) {
//       return PrintPrimExpr(v);
//     });

}
}
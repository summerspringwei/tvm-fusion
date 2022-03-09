#include "fuse_tensor_expression.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>
#include <tvm/te/operation.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/transform.h>
#include <tvm/tir/stmt.h>

#include <queue>
#include <vector>
#include <unordered_map>
#include <set>
#include <utility>

#include "../../support/arena.h"
#include "expr_subst.h"
#include "pattern_utils.h"
#include "fold_constant.h"

namespace tvm {
namespace relay {
using namespace tvm::tir;
using namespace tvm::te;

using TensorMap = std::unordered_map<Tensor, Tensor, ObjectPtrHash, ObjectPtrEqual>;

extern Expr FoldConstant(const Expr& expr, const IRModule& mod);

bool check_only_one_compute(const te::Tensor& output_tensor, 
  std::vector<te::Tensor>& compute_branch, std::vector<te::Tensor>& placeholder_branch) {
  size_t count = 0;
  te::Tensor input_tensor_compute;
  for(auto tensor: output_tensor->op->InputTensors()){
    if(tensor->op.as<te::ComputeOpNode>()){
      count++;
      input_tensor_compute = tensor;
      compute_branch.push_back(tensor);
    }else if(tensor->op.as<PlaceholderOpNode>()){
      placeholder_branch.push_back(tensor);
    }
  }
  if(count==1){
    return check_only_one_compute(input_tensor_compute, compute_branch, placeholder_branch);
  }else if(count==0){
    return true;
  }else{
    return false;
  }
};


// Modify a ComputeOp's ProducerLoad all related with specific tensor
class ProduceLoadInseartIndiceRewriter : public StmtExprMutator {
  public:
  
  ProduceLoadInseartIndiceRewriter(te::Var index_var, TensorMap& replace_map)
    : index_var_(index_var), replace_map_(replace_map) {}

  PrimExpr Rewrite(PrimExpr expr) { 
    VLOG(2) << "Start rewrite: " << expr;
    return this->VisitExpr(expr); 
  }

  // Modify the ProducerLoad and it's producer at the same time
  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<ProducerLoadNode>();
    te::Tensor t = Downcast<te::Tensor>(op->producer);
    if(this->replace_map_.count(t)) {
      Array<PrimExpr> new_args = {this->index_var_};
      for(auto arg: op->indices){
        new_args.push_back(arg);
      }
      auto new_expr = ProducerLoad(replace_map_[t], new_args);
      VLOG(2) << "NewProducerLoad: " << new_expr << " replace old " << t 
        << " " << ObjectPtrHash()(t) << " with " << replace_map_[t] << " " <<ObjectPtrHash()(replace_map_[t]);
      return new_expr;
    }
    return expr;
  }

  private:
  te::Var index_var_;
  TensorMap replace_map_;
};

Tensor RecursiveTensorRewriter(const Tensor& tensor, TensorMap& placeholder_map, const size_t num_branches){
  if(auto compute_op = tensor->op.as<ComputeOpNode>()){
    TensorMap replace_map;
    auto input_tensors = compute_op->InputTensors();
    // Get new body
    for(auto& it: input_tensors){
      if(it->op.as<ComputeOpNode>()){
        auto new_it = RecursiveTensorRewriter(it, placeholder_map, num_branches);
        ICHECK(new_it != NullValue<Tensor>());
        replace_map.insert({it, new_it});
        VLOG(2) << "replace_map.insert({it, new_it}); " << it->op << " -> " << new_it->op;
      }else if(it->op.as<PlaceholderOpNode>()){
        auto new_it = placeholder_map[it];
        replace_map.insert({it, new_it});
      }else{
        LOG_FATAL << "Unknow tensor op: " << it->op;
      }
    }

    // Rewrite new body
    auto new_var = tir::Var("g");
    auto rewriter = ProduceLoadInseartIndiceRewriter(new_var, replace_map);
    Array<PrimExpr> new_body;
    for(auto prim_expr: compute_op->body){
      auto new_prim_expr = rewriter.Rewrite(prim_expr);
      VLOG(2) << "new_prim_expr: " << new_prim_expr;
      new_body.push_back(new_prim_expr);
    }
    // Rewrite new axis and attrs
    Array<IterVar> new_axis = {tir::IterVar(tvm::Range(0, (int32_t)num_branches), new_var, IterVarType::kDataPar)};
    for(auto iter_var: compute_op->axis) {
      new_axis.push_back(iter_var);
    }
    Map<String, ObjectRef> new_attrs;
    if (compute_op->tag=="conv2d_nchw") {
      new_attrs = {};
    } else {
      new_attrs = compute_op->attrs;
    }
    auto new_compute = te::ComputeOp(compute_op->name, compute_op->tag, new_attrs, new_axis, new_body);
    // Rewrite new shape
    auto new_shape = Array<PrimExpr>({(int32_t)num_branches});
    for(auto s: tensor->shape){
      new_shape.push_back(s);
    }
    // Return new Tensor
    return te::Tensor(new_shape, tensor->dtype, new_compute, tensor->value_index);
  }else{
    return NullValue<Tensor>();
  }
}


class PrimFuncFusionRewriteV3 {
public:
  PrimFuncFusionRewriteV3(std::unordered_map<te::Tensor, Expr, ObjectPtrHash, ObjectPtrEqual> tensor_constant_map) 
  : tensor_constant_map_(tensor_constant_map){};
  Array<te::Tensor> Transform(Array<te::Tensor> outputs) {
    Array<te::Tensor> to_be_combined_tensors = outputs;
    // size_t num_of_tensor_produced_by_compute_op = 0;
    // size_t num_of_tensor_produced_by_placeholder_op = 0;
    // If the output is a sink op, we trace it's inputs to find the branches
    while(to_be_combined_tensors.size() == 1) {
      auto tmp_tensors = to_be_combined_tensors[0]->op->InputTensors();
      to_be_combined_tensors.clear();
      for(auto tensor: tmp_tensors){
        if(tensor->op.as<te::ComputeOpNode>()){
          to_be_combined_tensors.push_back(tensor);
        }
      }
    }
    // We assume in each branch each ComputeOp's input tensors only contains one tensor
    // that is produced by computeOp
    // We check here
    const size_t num_of_branches = to_be_combined_tensors.size();
    std::vector<std::vector<te::Tensor>> compute_branches(num_of_branches);
    std::vector<std::vector<te::Tensor>> placeholder_branches(num_of_branches);

    bool all_branch_only_one = true;
    size_t i = 0;
    for(auto& tensor: to_be_combined_tensors){
      all_branch_only_one &= check_only_one_compute(tensor, compute_branches[i], placeholder_branches[i]);
      i++;
    }
    ICHECK(all_branch_only_one);
    // TODO(Chunwei Xia) Structural Equal can not help to compare two PrimExprs
    // BFS to Check Structure equal
    bool equal = true;
    StructuralEqual seq;
    for(size_t j=0; j<compute_branches[0].size(); ++j){
      for(i=0; i<num_of_branches; ++i){
        VLOG(2) << "Compare " << (compute_branches[0][j]->op) << " and " << compute_branches[i][j]->op;
        equal = (equal && seq(Downcast<te::ComputeOp>(compute_branches[0][j]->op)->body,
          Downcast<te::ComputeOp>(compute_branches[i][j]->op)->body));
        if(!equal){
          LOG_WARNING << compute_branches[0][j]->op
            << " and " << compute_branches[i][j]->op << "Structure not equal";
        }
      }
    }
    // ICHECK(equal);
    // Now concate all placeholder and insert in placeholder->concated_placeholder map
    // We need to fold constant weights here
    TensorMap placeholder_replace_map;
    for(size_t j=0; j<compute_branches[0].size(); ++j){
      Array<Tensor> tuple;
      Array<Expr> expr_fields;
      bool find_constant = false;
      for(i=0; i<num_of_branches; ++i){
        tuple.push_back(placeholder_branches[i][j]);
        if(this->tensor_constant_map_.count(placeholder_branches[i][j])){
          expr_fields.push_back(this->tensor_constant_map_[placeholder_branches[i][j]]);
          find_constant = true;
        }
      }
      if(find_constant){
        auto expr_concated = relay::MakeConcatenate(relay::Tuple(expr_fields), 0);
        auto expr_new_shape = Array<tvm::Integer>({tvm::Integer((int32_t)num_of_branches)});
        for(auto prim_expr: tuple[0]->shape){
          expr_new_shape.push_back(Downcast<tvm::Integer>(prim_expr));
        }
        auto expr_reshaped = relay::MakeReshape(expr_concated, expr_new_shape);
        IRModule module;
        auto result_constant = FoldConstant(expr_reshaped, module);
        VLOG(2) << result_constant;
        auto expr_constant = Downcast<Constant>(result_constant);
        tvm::te::Tensor concated_tensor = tvm::te::placeholder(expr_constant->tensor_type()->shape, expr_constant->tensor_type()->dtype);
        placeholder_replace_map.insert({placeholder_branches[0][j], concated_tensor});
      }else{
        auto concated = topi::concatenate(tuple, 0);
        auto new_shape = Array<PrimExpr>({(int32_t)num_of_branches});
        for(auto prim_expr: tuple[0]->shape){
          new_shape.push_back(prim_expr);
        }
        placeholder_replace_map.insert({placeholder_branches[0][j], topi::reshape(concated, new_shape)});
      }
    }
    // Next we visit the first branch in PostOrder
    // First rewrite the computeOp's body that consume placeholder, 
    // then return a new computeOp with a new tensor, then rewrite
    Array<Tensor> new_outputs;
    // We only need to rewrite one branch
    new_outputs.push_back(RecursiveTensorRewriter(outputs[0], placeholder_replace_map, num_of_branches));
    return new_outputs;
  }

  private:
  std::unordered_map<te::Tensor, Expr, ObjectPtrHash, ObjectPtrEqual> tensor_constant_map_;
};


namespace transform {
  Array<te::Tensor> FFusionTensorExpression(Array<te::Tensor> outputs, 
    std::unordered_map<te::Tensor, Expr, ObjectPtrHash, ObjectPtrEqual> tensor_constant_map){
    auto fusion = PrimFuncFusionRewriteV3(tensor_constant_map);
    return fusion.Transform(outputs);
  }
}
}
}
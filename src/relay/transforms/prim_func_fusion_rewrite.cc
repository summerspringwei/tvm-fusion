
#include "prim_func_fusion_rewrite.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>
#include <tvm/te/operation.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/transform.h>

#include <queue>
#include <vector>
#include <unordered_map>
#include <set>
#include <utility>

#include "../../support/arena.h"
#include "expr_subst.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

using namespace tvm::tir;

// 1. Build relationship between ProducerLoad and Tensor (Tensor is the DataProducer of the ProducerLoad)
// We can also obtain the PlaceholderOp throught the TE compute graph connected by tensors
// ComputeOps connectted through input/output tensors
// PrimExpr in ComputeOp's body also connected through tensors by ProducerLoad's producers
// Therefore we can find which ProducerLoadOps consume a specific placeholder op
class LoadTensorRelaionBuilder : public StmtExprVisitor {
  public:
  void Build(const PrimExpr& e) {
    ExprVisitor::VisitExpr(e);
  }

  void PrintRelation() {
    VLOG(2) << "PrimExpr -> Tensor:\n";
    for(const auto& ele: this->load_tensor_map_){
      VLOG(2) << ele.first << " -> " << ele.second ;
    }
    VLOG(2) << "Tensor -> PrimExpr:\n";
    for(const auto& ele: this->tensor_load_map_){
      std::stringstream os;
      os << ele.first << " -> " << "[";
      for(auto& load: ele.second){
        os << load << ", ";
      }
      os << "]" ;
      VLOG(2) << os.str();
    }
  }
  
  void VisitExpr_(const ProducerLoadNode* op) {
    te::Tensor t = Downcast<te::Tensor>(op->producer);
    // ICHECK(load_tensor_map_.count(GetRef<PrimExpr>(op)) == 0);
    load_tensor_map_.insert({GetRef<PrimExpr>(op), t});
    if(tensor_load_map_.count(t) == 0){
      tensor_load_map_.insert({t, Array<PrimExpr>({GetRef<PrimExpr>(op)})});
    }else{
      tensor_load_map_[t].push_back(GetRef<PrimExpr>(op));
    }
  }

  void VisitExpr_(const AddNode* op) {
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }

  void VisitExpr_(const SubNode* op) {
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }

  void VisitExpr_(const MulNode* op) {
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }

  void VisitExpr_(const ReduceNode* op) {
    for(auto prim_expr: op->source){
      this->VisitExpr(prim_expr);
    }
  }

  void VisitExpr_(const SelectNode* op) {
    VLOG(2) << "Visit SelectNode";
    this->VisitExpr(op->condition);
    this->VisitExpr(op->true_value);
    this->VisitExpr(op->false_value);
  }

  std::unordered_map<PrimExpr, te::Tensor, ObjectPtrHash, ObjectPtrEqual> load_tensor_map_;
  std::unordered_map<te::Tensor, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual> tensor_load_map_;
};


// 2. Build relationship between input tensor and output tensor
class TERelationBuilder {
  public:

  void UpdateTensorMap_(const Array<te::Tensor>& inputs, const te::Tensor output_tensor){
    for(const auto& input_tensor: inputs){
      if(this->tinput_toutput_map.count(input_tensor) == 0){
        this->tinput_toutput_map.insert({input_tensor, Array<te::Tensor>({output_tensor})});
      }else{
        this->tinput_toutput_map[input_tensor].push_back(output_tensor);
      }
      if(this->toutput_tinput_map.count(output_tensor) == 0){
        this->toutput_tinput_map.insert({output_tensor, Array<te::Tensor>({input_tensor})});
      }else{
        this->toutput_tinput_map[output_tensor].push_back(input_tensor);
      }
    }
  }

  // Build relationship between the ComputeOp's input and output tensors of the whole TE graph
  void Build(const Array<te::Tensor>& outputs) {
    std::set<te::Operation> op_visited;
    std::function<void(const Array<te::Tensor>&)> recursive_visitor = [&](const Array<te::Tensor>& outputs) {
      for(auto& tensor: outputs){
        if(op_visited.count(tensor->op)){
          continue;
        }
        if(auto compute = tensor->op.as<te::ComputeOpNode>()){
          for(auto prim_expr: compute->body){
            this->load_tensor_builder.Build(prim_expr);
            this->prim_expr_op_map.insert({prim_expr, tensor->op});
          }
          auto inputs = compute->InputTensors();
          this->UpdateTensorMap_(inputs, tensor);
          recursive_visitor(inputs);
        }
        // TODO(Chunwei Xia) May consider the PlaceholderOp
      }
    };
    recursive_visitor(outputs);
  }

  // Get the rewrite tensor set, must first call Build
  void GetRewriteTensorSet(const Array<te::Tensor>& inputs) {
    // TODO(Chunwei Xia) May apply more precise dataflow analysis
    for(auto input_tensor: inputs) {
      if(this->rewrite_tensor_set_.count(input_tensor)){
        continue;
      }
      this->rewrite_tensor_set_.insert(input_tensor);
      VLOG(2) << "rewrite_tensor_set_ add" << input_tensor;
      auto output_tensors = this->tinput_toutput_map[input_tensor];
      GetRewriteTensorSet(output_tensors);
    }
  }

  void PrintRelations() {
    VLOG(2) << "PrimExpr -> ComputeOp:";
    for(const auto& ele: this->prim_expr_op_map){
      VLOG(2) << ele.first << " -> " << ele.second;
    }
    VLOG(2) << "Output Tensor -> Input Tensor";
    for(const auto& ele: this->toutput_tinput_map){
      VLOG(2) << ele.first << " -> " << ele.second;
    }
    VLOG(2) << "Input Tensor -> Output Tensor";
    for(const auto& ele: this->tinput_toutput_map){
      VLOG(2) << ele.first << " -> " << ele.second;
    }
    VLOG(2) << "Rewrite Tensor Set:";
    for(const auto& tensor: this->rewrite_tensor_set_){
      VLOG(2) << tensor ;
    }
    this->load_tensor_builder.PrintRelation();
  }

  LoadTensorRelaionBuilder load_tensor_builder;
  // ProducerLoad to its computeOp
  std::unordered_map<PrimExpr, te::Operation, ObjectPtrHash, ObjectPtrEqual> prim_expr_op_map;
  std::unordered_map<te::Tensor, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> tinput_toutput_map;
  std::unordered_map<te::Tensor, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> toutput_tinput_map;
  // Set of tensors with relate ProducerLoadExpr need to be rewrite
  std::set<te::Tensor> rewrite_tensor_set_;
};


// Modify a ComputeOp's ProducerLoad all related with specific tensor
class ProduceLoadInseartIndiceRewriter : public StmtExprMutator {
  public:
  
  ProduceLoadInseartIndiceRewriter(te::Var& var, const std::set<te::Tensor>& rewrite_tensor_set,
    const std::unordered_map<te::Tensor, te::Tensor, ObjectPtrHash, ObjectPtrEqual>& replace_map)
    : var_(var), rewrite_tensor_set_(rewrite_tensor_set),  replace_map_(replace_map) {
      VLOG(2) << "ReplaceMap:";
      for(const auto& ele: replace_map_){
        VLOG(2) << ele.first << " -> " << ele.second;
      }
    }

  PrimExpr Mutate(PrimExpr expr) {
    VLOG(2) << "Start rewrite: " << expr;
    return this->VisitExpr(expr);
  }

  // Modify the ProducerLoad and it's producer at the same time
  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    te::Tensor t = Downcast<te::Tensor>(op->producer);
    if(this->rewrite_tensor_set_.count(t)) {
      Array<PrimExpr> new_args = {this->var_};
      for(auto arg: op->indices){
        new_args.push_back(arg);
      }
      VLOG(2) << "DataProduce: "<< t;
      ICHECK(replace_map_.count(t));
      auto new_expr = ProducerLoad(replace_map_[t], new_args);
      VLOG(2) << "NewProducerLoad: " << new_expr;
      return new_expr;
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const AddNode* op) {
    return tvm::tir::Add(this->VisitExpr(op->a), this->VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const SubNode* op) {
    return tvm::tir::Sub(this->VisitExpr(op->a), this->VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const MulNode* op) {
    return tvm::tir::Mul(this->VisitExpr(op->a), this->VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const ReduceNode* op) {
    Array<PrimExpr> new_source;
    for(auto prim_expr: op->source){
      new_source.push_back(this->VisitExpr(prim_expr));
    }
    return tvm::tir::Reduce(op->combiner, new_source, op->axis, op->condition, op->value_index, op->init);
  }

  PrimExpr VisitExpr_(const SelectNode* op) {
    return tvm::tir::Select(this->VisitExpr(op->condition), 
      this->VisitExpr(op->true_value), this->VisitExpr(op->false_value));
  }

  private:
  tir::Var var_;
  std::set<te::Tensor> rewrite_tensor_set_;
  std::unordered_map<te::Tensor, te::Tensor, ObjectPtrHash, ObjectPtrEqual> replace_map_;
};


// 3. Rewrite the relay graph and TE graph
class PrimFuncFusionRewriteV2 : private ExprMutator {
  public:
  PrimFuncFusionRewriteV2(const Expr prim_func, ExprTEMap expr_te_map, TEExprMap te_expr_map, int32_t num_branch): 
    prim_func_(prim_func),  num_branch_(num_branch), expr_te_map_(expr_te_map), te_expr_map_(te_expr_map) {};

  std::pair<Expr, Array<te::Tensor>> Transform() {
    // 1. Rewrite the original relay graph and get entry and exit tensors
    auto expr = this->Mutate(prim_func_);
    // 2.1 Build the input and output tensor relations from the exit tensors
    relation_builder.Build(this->tensors_from_a_branch_);
    // 2.2 Now we get all the tensors need to be rewrite from the entry tensors
    relation_builder.GetRewriteTensorSet(this->tensors_from_split_);
    relation_builder.PrintRelations();
    // 3. Rewrite
    this->PrintRelation();
    auto output_tensors = this->RewriteTensors();
    // Array<te::Tensor> output_tensors;
    return std::make_pair(expr, output_tensors);
  }

  // Rewrite recursively from the output tensor of the relay graph
  te::Tensor RewriteComputeOp(te::Tensor output_tensor, std::string iter_var_name, int extent) {
    if(auto compute = output_tensor->op.as<te::ComputeOpNode>()){
      auto new_var = tir::Var(iter_var_name);
      // (1) Rewrite ProducerLoadNode's indices in compute->body
      auto pl_rewriter = ProduceLoadInseartIndiceRewriter(new_var, 
        this->relation_builder.rewrite_tensor_set_, this->replace_map_);
      Array<PrimExpr> new_body;
      for(auto expr: compute->body){
        VLOG(2) << "Rewrite: " << expr;
        auto new_expr = pl_rewriter.Mutate(expr);
        VLOG(2) << " to-> " << new_expr;
        new_body.push_back(new_expr);
      }
      // (2) Rewrite ComputeOp's axis
      Array<IterVar> new_axis = {tir::IterVar(tvm::Range(0, extent), new_var, IterVarType::kDataPar)};
      for(auto iter_var: compute->axis) {
        new_axis.push_back(iter_var);
      }
      auto new_compute = te::ComputeOp(compute->name, compute->tag, compute->attrs, new_axis, new_body);
      // (3) Rewrite Tensor's shape and op
      auto new_shape = Array<PrimExpr>({extent});
      for(auto s: output_tensor->shape){
        new_shape.push_back(s);
      }
      auto new_tensor = te::Tensor(new_shape, output_tensor->dtype, new_compute, output_tensor->value_index);
      return new_tensor;
    }
    return output_tensor;
  }

  // Rewrite all the tensors, it's op and op's primExpr
  Array<te::Tensor> RewriteTensors(){
    // Using BFS to rewrite from input to output
    std::queue<te::Tensor> queue_tensor;
    std::set<te::Tensor> added;
    auto funique_push_to_queue = [&added, &queue_tensor](const auto& tensor) {
      if(added.count(tensor) == 0){
        queue_tensor.push(tensor);
        added.insert(tensor);
      }
    };
    for(auto& tensor: this->tensors_from_split_) {
      auto new_shape = Array<PrimExpr>({this->num_branch_});
      for(auto s: tensor->shape) {
        new_shape.push_back(s);
      }
      auto new_tensor = te::Tensor(new_shape, tensor->dtype, tensor->op, tensor->value_index);
      this->replace_map_.insert({tensor, new_tensor});
      ICHECK(relation_builder.tinput_toutput_map.count(tensor));
      for(auto& output: relation_builder.tinput_toutput_map[tensor]){
        VLOG(2) << "relation_builder.tinput_toutput_map[tensor]" << tensor << " -> " << output;
        funique_push_to_queue(output);
      }
    }
    
    std::set<te::Tensor> visisted;
    while(!queue_tensor.empty()) {
      auto output_tensor = queue_tensor.front();
      queue_tensor.pop();
      if(visisted.count(output_tensor)) {
        continue;
      }
      visisted.insert(output_tensor);
      /**
       * 
       * output_tensor (first compute op after split)
       */
      // Get ouput_tensor's parent, if it's in spilt, then replace the input tensor with
      for(auto split_tensor: relation_builder.toutput_tinput_map[output_tensor]){
        if(this->split_to_var_tensor_map_.count(split_tensor)){
          auto& placeholder_tensor = this->split_to_var_tensor_map_[split_tensor];
          VLOG(2) << "placeholder_tensor: " << placeholder_tensor;
          Array<PrimExpr> new_shape = {PrimExpr(num_branch_), 
          // tir::Div(placeholder_tensor->shape[0], PrimExpr(num_branch_))};
          PrimExpr((int32_t)placeholder_tensor->shape[0].as<IntImmNode>()->value / num_branch_)};
          for(size_t i = 1; i<placeholder_tensor->shape.size(); ++i){
            new_shape.push_back(placeholder_tensor->shape[i]);
          }
          this->replace_map_[split_tensor] = tvm::topi::reshape(placeholder_tensor, new_shape);
          VLOG(2) << "Connected with PlaceHolder " << placeholder_tensor << " new_output_tensor " << this->replace_map_[split_tensor] ;
        }
      }
      
      VLOG(2) << "rewirte_output_tensor" << output_tensor;
      auto new_output_tensor = this->RewriteComputeOp(output_tensor, "g", num_branch_);
      this->replace_map_[output_tensor] = new_output_tensor;
      // If we rewrite split tensor, 
      // we will insert topi::reshape between the new split_tensor and Placeholder
      
      if(relation_builder.tinput_toutput_map.count(output_tensor)) {
        for(auto& child_tensor: relation_builder.tinput_toutput_map[output_tensor]) {
          funique_push_to_queue(child_tensor);
        }
      }
    }
    Array<te::Tensor> new_output_tensor;
    for(auto output: this->tensors_from_a_branch_){
      new_output_tensor.push_back(this->replace_map_[output]);
    }

    return new_output_tensor;
  }

  void PrintMemo(){
    VLOG(2) << "Mutate memo:";
    for(auto ele: this->memo_) {
      VLOG(2) << ele.first << " -> " << ele.second;
    }
  }

  // TODO (Chunwei Xia) Deal with concat
  Expr VisitExpr_(const CallNode* op) {
    // Is the return expr of the Function
    if(GetRef<Expr>(op) == this->func_return_expr_ && op->op == Op::Get("concatenate")) {
      VLOG(2) << "Get func_return_expr && Get concatenate";
      ICHECK(op->args.size()==1);
      auto result = this->VisitExpr(op->args[0]);
      VLOG(2) << "result: " << result;
      return result;
    }
    for(auto arg: op->args) {
      this->VisitExpr(arg);
    }
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const FunctionNode* op) {
    VLOG(2) << "FunctionNode: " << GetRef<Expr>(op);
    this->func_return_expr_ = op->body;
    return Function(op->params, this->VisitExpr(op->body), op->ret_type, op->type_params, op->attrs, op->span);
  }

  // TODO(Chunwei Xia) Add reshape op
  Expr VisitExpr_(const TupleGetItemNode* op) {
    VLOG(2) << "TupleGetItemNode: " << GetRef<Expr>(op);
    if(auto call_op = op->tuple.as<CallNode>()){
      if(call_op->op == Op::Get("split") 
        && call_op->args.size() == 1 
        && call_op->args[0].as<VarNode>()){
        VLOG(2) << "Find branch entry";
        for(auto split_tensor: expr_te_map_[GetRef<Expr>(op)]) {
          tensors_from_split_.push_back(split_tensor);
          // Placeholder tensor
          for(auto placeholder_tensor: expr_te_map_[call_op->args[0]]){
            split_to_var_tensor_map_.insert({split_tensor, placeholder_tensor});
          }
        }
      }
    }
    return GetRef<Expr>(op);
  }

  // TODO(Chunwei Xia) For Now we assume all the branches are equal
  Expr VisitExpr_(const TupleNode* op) {
    VLOG(2) << "TupleNode: " << GetRef<Expr>(op);
    if (op->fields.size() == 1) {
      return GetRef<Expr>(op);
    }
    auto f0 = op->fields[0];
    bool all_equal = true;
    // TODO(Chunwei Xia) Find how to compare relay graph equal
    for(auto f: op->fields) {
      all_equal = all_equal && StructuralEqual()(f0, f);
    }
    tensors_from_a_branch_ = expr_te_map_[f0];
    // ICHECK(all_equal);
    return this->VisitExpr(f0);
  }

  void PrintRelation() {
    VLOG(2) << "tensors from split:";
    for(auto t: this->tensors_from_split_){
      VLOG(2) << t;
    }
    VLOG(2) << "Tensors from a branch:";
    for(auto t: this->tensors_from_a_branch_){
      VLOG(2) << t;
    }
    VLOG(2) << "split_to_var_tensor_map_:";
    for(auto ele: this->split_to_var_tensor_map_){
      VLOG(2) << ele.first << " -> " << ele.second;
    }
  }

  public:
  // The Function marked with kFusion to be rewrite
  Expr prim_func_;
  // The number of branch
  int32_t num_branch_;
  // Tensors produced by the split op
  Array<te::Tensor> tensors_from_split_;
  // Record the tensor produced by split node to it's parent placeholder,
  // then at the mutate stage we inseart reshape op between them
  std::unordered_map<te::Tensor, te::Tensor, ObjectPtrHash, ObjectPtrEqual> split_to_var_tensor_map_;
  // Tensors produced by the end op of a branch
  Array<te::Tensor> tensors_from_a_branch_;
  // Tensor replace map
  std::unordered_map<te::Tensor, te::Tensor, ObjectPtrHash, ObjectPtrEqual> replace_map_;
  // The return expr of the prim_func_
  Expr func_return_expr_;
  TERelationBuilder relation_builder;
  ExprTEMap expr_te_map_;
  TEExprMap te_expr_map_;
};

std::pair<Expr, Array<te::Tensor>> RewriteFusedPrimFunc(const Expr prim_func, ExprTEMap expr_te_map,
  TEExprMap te_expr_map, int32_t num_branch) {
  // return PrimFuncFusionRewrite(prim_func, expr_te_map, te_expr_map).Transform();
  return PrimFuncFusionRewriteV2(prim_func, expr_te_map, te_expr_map, num_branch).Transform();
}

// Recursively visit computeOp's input tensors
void PrintTEGraph(te::Tensor tensor){
  if(auto compute = tensor->op.as<te::ComputeOpNode>()){
    VLOG(2) << tensor->op ;
    for(auto t: compute->InputTensors()){
      PrintTEGraph(t);
    }
  }else if(auto placeholder = tensor->op.as<te::PlaceholderOpNode>()) {
    VLOG(2) << placeholder->name << " " << tensor;
  }
}

}
}
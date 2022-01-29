
/**
 * \file horizontal_fuse_op.cc
 *
 *
 * \brief Fuse ops horizontally
 *
 * Firstly we try to visit the packedFunction created by te::compute
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include <queue>
#include <vector>
#include <unordered_map>

#include "../../support/arena.h"
#include "expr_subst.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

using support::LinkedList;
using support::LinkNode;

template<typename X>
using ExprMap = std::unordered_map<Expr, X, ObjectPtrHash, ObjectPtrEqual>;
using Branch = std::vector<const CallNode*>;
using Group = std::vector<Branch>;
using ExprSubstMap = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>;
// template<typename Y>
// using NodePtrMap = std::unordered_map<const relay::ExprNode*, Y, ObjectHash, ObjectEqual>;
// using NodePtrMap = std::unordered_map<const relay::ExprNode*, Y, ObjectHash, ObjectEqual>;
/*!
 * \brief Indexed data flow graph in forward direction.
 *  This is a temporary data structure used for operator fusion analysis.
 *
 *  This data structure only captures the dataflow fragment and
 *  could ignore blocks like let by simply ordering each dataflow block
 *  and mark the output node as extern_ref;
 */
class UpwardRankGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const ExprNode* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief The upward rank of the node in topological order. */
    size_t rank{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;
  };
  /*! \brief The node map that maps node to graph */
  ExprMap<Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> forward_bfs_order;
  /*! \brief uprank to array of exprs with same uprank */
  std::unordered_map<size_t, std::vector<Expr>> uprank_array_map;
  /*! \brief From child to parents */
  ExprMap<std::vector<const ExprNode*>> parent_map_;  
  /*! \brief From parents to child */
  ExprMap<std::vector<const ExprNode*>> child_map_;
  /*! \brief input variables of the relay graph */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> input_vars_;
  /*! \brief constant weights of the relay graph */
  std::unordered_set<Constant, ObjectPtrHash, ObjectPtrEqual> ops_weights_;
  /*! \brief Operators that can be fused */
  std::vector<Group> groups;
  /*! \brief Dump the graph into string. */
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < forward_bfs_order.size(); ++i) {
      Node* node = forward_bfs_order[i];
      os << "node[" << i << "], " << GetRef<Expr>(node->ref) << " uprank " << node->rank;
      // for (auto* link = node->outputs.head; link != nullptr; link = link->next) {
      //   os << link->value.node->index << ", ";
      // }
      os << "]\n";
    }
    LOG(INFO) << os.str();
  }
  /*!
   * \brief create a indexed forward graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static UpwardRankGraph Create(support::Arena* arena, const Expr& body);

  std::vector<const ExprNode*> getExprChildren(Expr call){
    if(this->child_map_.count(call)){
      return this->child_map_[call];
    }else{
      return std::vector<const ExprNode*>();
    }
  }
 public:
  class Creator;
};



// class HoriFusionMutator : private ExprVisitor {
class UpwardRankGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}

  UpwardRankGraph Transform(const Expr& body) {
    this->VisitExpr(body);
    graph_.groups = this->FindFusedConv2DOps_();
    return std::move(graph_);
  }

 private:
  
  /*! \brief Internal arena. */
  support::Arena* arena_;
  UpwardRankGraph graph_;

  void VisitExpr_(const CallNode* c) {
    VLOG(2) << "********* VisitExpr CallNode: ***********\n";
    this->VisitExpr(c->op);
    UpdateRelationMap_(GetRef<Call>(c), c->args);
    for(auto& arg: c->args) {
      this->VisitExpr(arg);
    }
    
    if(auto* param = c->attrs.as<Conv2DAttrs>()){
      auto tweight = c->args[1]->type_as<TensorTypeNode>();
      auto oc = tir::as_const_int(tweight->shape[0]);
      auto oi = tir::as_const_int(tweight->shape[1]);
      auto kh = tir::as_const_int(tweight->shape[2]);
      auto kw = tir::as_const_int(tweight->shape[3]);
      VLOG(2) <<"strides: "<<param->strides;
      VLOG(2) << "Conv2d weight shape: " << *oc << "," << *oi << "," << *kh <<"," << *kw;
    }
  }
  
  // For now we assume that FunctionNode is at the top level
  void VisitExpr_(const FunctionNode* f) {
    VLOG(2) << "********* VisitExpr FunctionNode: ***********\n";
    this->VisitExpr(f->body);
    FindInputVars_(f->body);
    DumpDebugRelationMap(graph_.parent_map_);
    DebugDumpInputVars_();
    BuildUprankGraph_();
    graph_.DebugDump();
  }

  // TODO(Chunwei Xia) Try to lower op here
  void VisitExpr_(const OpNode* op) {
    std::ostringstream os;
    auto op_ref = GetRef<ObjectRef>(op);
    
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    auto op_pattern = static_cast<OpPatternKind>(fpattern[GetRef<Op>(op)]);
    static auto ftvmcompute = Op::GetAttrMap<FTVMCompute>("FTVMCompute");
    
    //auto op_fcompute = static_cast<FTVMCompute>(ftvmcompute[GetRef<Op>(op)]);
    os << "name " << op->name << " ";
    VLOG(2) << os.str() << "pattern " << op_pattern
              << " ********* VisitExpr OpNode: ***********\n";
  }

  void VisitExpr_(const VarNode* op) {
    VLOG(2) << "********* name: " << op->name_hint() << " VisitExpr VarNode: ***********\n";
  }

  void VisitExpr_(const TupleNode* op) { UpdateRelationMap_(GetRef<Tuple>(op), op->fields); }

  // child consume data produced by parents
  // parent_map_ is child->parents
  // child_map_ is parents->child
  void UpdateRelationMap_(Expr child, Array<Expr> parents) {
    for (size_t i = 0; i < parents.size(); ++i) {
      auto arg = parents[i];
      this->VisitExpr(arg);
      if (graph_.parent_map_.count(child) == 0) {
        auto new_arr = std::vector<const ExprNode*>();
        new_arr.push_back(arg.as<ExprNode>());
        graph_.parent_map_[child] = new_arr;
      } else {
        graph_.parent_map_[child].push_back(arg.as<ExprNode>());
      }
      auto child_op_node = child.as<ExprNode>();
      if (graph_.child_map_.count(arg) == 0) {
        auto new_arr = std::vector<const ExprNode*>();
        new_arr.push_back(child_op_node);
        graph_.child_map_[arg] = new_arr;
      } else {
        graph_.child_map_[arg].push_back(child_op_node);
      }
    }
  }

  std::string GetExprNodeTypeStr(Expr relay_expr){
    if(relay_expr.as<ConstantNode>()){
      return "Constant";
    }else if(relay_expr.as<VarNode>()){
      return "Var";
    }else if(relay_expr.as<TupleNode>()){
      return "Tuple";
    }else if(relay_expr.as<CallNode>()){
      return "Call";
    }else if(relay_expr.as<LetNode>()){
      return "Let";
    }else if(relay_expr.as<LetNode>()){
      return "Let";
    }else if(relay_expr.as<TupleGetItemNode>()){
      return "TupleGetItem";
    }else if(relay_expr.as<RefCreateNode>()){
      return "RefCreate";
    }else if(relay_expr.as<RefReadNode>()){
      return "RefRead";
    }else{
      return "Maybe RefWriteNode or TempExprNode";
    }
  }

  void DumpDebugRelationMap(ExprMap<std::vector<const ExprNode*>>& r_map) {
    for (auto it = r_map.begin(); it != r_map.end(); ++it) {
      std::string child_op_type = GetExprNodeTypeStr(it->first);
      for (size_t i = 0; i < it->second.size(); ++i) {
        std::string parent_op_type = GetExprNodeTypeStr(GetRef<Expr>(it->second[i]));
        VLOG(2) << child_op_type << " to " << parent_op_type << " " << it->first << "->" << it->second[i] << "\n";
      }
    }
  }

  void FindInputVars_(Expr body) {
    auto expr_queue = std::queue<Expr>();
    expr_queue.push(body);
    while (!expr_queue.empty()) {
      auto current_expr = expr_queue.front();
      expr_queue.pop();
      if (current_expr.as<VarNode>()) {
        ICHECK(graph_.parent_map_.count(current_expr) == 0);
        graph_.input_vars_.insert(Downcast<Var>(current_expr));
      } else if (current_expr.as<ConstantNode>()) {
        // Save constant node as weights we need further
        ICHECK(graph_.parent_map_.count(current_expr) == 0);
        graph_.ops_weights_.insert(Downcast<Constant>(current_expr));
      } else if (graph_.parent_map_.count(current_expr)) {
        for (const ExprNode* parent_expr : graph_.parent_map_[current_expr]) {
          expr_queue.push(GetRef<Expr>(parent_expr));
        }
      } else {
        VLOG(2) << "Special case need to be handled " << current_expr << "\n";
      }
    }
  }
  
  void AddNode(const ExprNode* op){
    auto op_expr = GetRef<Expr>(op);
    if(graph_.node_map.count(op_expr) == 0){
      UpwardRankGraph::Node* op_node = arena_->make<UpwardRankGraph::Node>();
      op_node->ref = op;
      op_node->index = graph_.forward_bfs_order.size();
      graph_.forward_bfs_order.push_back(op_node);
      graph_.node_map[op_expr] = op_node;
      if(graph_.parent_map_[op_expr].size() == 0){
        op_node->rank = 0;
      }else{
        for(const ExprNode* p: graph_.parent_map_[op_expr]){
          ICHECK(graph_.node_map.count(GetRef<Expr>(p)));
          graph_.node_map[op_expr]->rank = std::max(graph_.node_map[op_expr]->rank, graph_.node_map[GetRef<Expr>(p)]->rank + 1);
        }
      }
      auto it = graph_.uprank_array_map.find(op_node->rank);
      if(it==graph_.uprank_array_map.end()){
        std::vector<Expr> expr_arr;
        graph_.uprank_array_map.insert({op_node->rank, {GetRef<Expr>(op_node->ref)}});
      }else{
        (*it).second.push_back(GetRef<Expr>(op_node->ref));
      }
      std::ostringstream os;
      os << "Add " << obj << " up rank " << op_node->rank << "\n";
      VLOG(2) << os.str();
    }
  }

  void BuildUprankGraph_() {
    // First add weights and vars to graph
    for (auto it = graph_.ops_weights_.begin(); it != graph_.ops_weights_.end(); ++it) {
      this->AddNode(it->as<ConstantNode>());
    }
    std::queue<Expr> tmp_queue;
    for (auto it = graph_.input_vars_.begin(); it != graph_.input_vars_.end(); ++it) {
      this->AddNode(it->as<VarNode>());
      tmp_queue.push(*it);
    }
    // We start from vars and use bfs to traverse the whole expr
    while(!tmp_queue.empty()){
      auto top = tmp_queue.front();
      tmp_queue.pop();
      if(graph_.child_map_.count(top)==0){
        LOG_WARNING << "Can not find " << top << "\n";
      }
      for(const ExprNode* child: graph_.child_map_[top]){
        this->AddNode(child);
        tmp_queue.push(GetRef<Expr>(child));
      }
    }
  }

  bool CanConv2DOpsBeCombined(const CallNode* a, const CallNode* b) {
    StructuralEqual eq;
    const Layout kOIHW("OIHW");
    const auto* attrs_a = a->attrs.as<Conv2DAttrs>();
    const auto* attrs_b = b->attrs.as<Conv2DAttrs>();
    ICHECK(attrs_a);
    ICHECK(attrs_b);
    const auto* tweight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* tweight_b = b->args[1]->type_as<TensorTypeNode>();
    const auto shape_a =
        tir::BijectiveLayout(Layout(attrs_a->kernel_layout), kOIHW).ForwardShape(tweight_a->shape);
    const auto shape_b =
        tir::BijectiveLayout(Layout(attrs_b->kernel_layout), kOIHW).ForwardShape(tweight_b->shape);

    return eq(attrs_a->strides, attrs_b->strides) && eq(attrs_a->padding, attrs_b->padding) &&
           eq(attrs_a->dilation, attrs_b->dilation) && eq(attrs_a->groups, attrs_b->groups) &&
           eq(attrs_a->data_layout, attrs_b->data_layout) &&
           eq(attrs_a->kernel_layout, attrs_b->kernel_layout) &&
           eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
           eq(attrs_a->out_layout, attrs_b->out_layout) && eq(shape_a[2], shape_b[2]) &&
           eq(shape_a[3], shape_b[3]);
  }

  //TODO(Chunwei Xia) For now we assume all the fuseable ops are with the same uprank
  std::vector<Group> FindFusedConv2DOps_() {
    std::vector<Group> can_be_fused_arr;
    for(auto it=graph_.uprank_array_map.begin(); it!=graph_.uprank_array_map.end(); ++it){
      if(it->second.size() < 2){
        continue;
      }
      auto first_op = it->second[0];
      // For now we only fuse conv2d
      auto call_node = first_op.as<CallNode>();
      if(!call_node){
        continue;
      }
      auto cond = call_node->attrs.as<Conv2DAttrs>();
      if(!cond){
        continue;
      }
      Group g;
      for(size_t i=1; i<it->second.size(); ++i){
        auto op = it->second[i];
        if(op.as<CallNode>()->attrs.as<Conv2DAttrs>()){
          if(CanConv2DOpsBeCombined(first_op.as<CallNode>(), op.as<CallNode>())){
            g.push_back(Branch({op.as<CallNode>()}));
          }
        }
      }
      g.push_back(Branch({first_op.as<CallNode>()}));
      VLOG(2) << "Find group with size " << g.size();
      can_be_fused_arr.push_back(g);
    }
    return can_be_fused_arr;
  }

  Expr ConcatInputs(const Group& branches, size_t arg_index) {
    Array<Expr> inputs;
    for(auto branch: branches) {
      auto conv2d = branch[0];
      ICHECK(arg_index<conv2d->args.size());
      inputs.push_back(conv2d->args[arg_index]);
    }
    return MakeConcatenate(Tuple(inputs), 0);
  }

  void DebugDumpInputVars_() {
    std::ostringstream os;
    for(auto var: graph_.input_vars_) {
      os << var << " ";
    }
    VLOG(2) << "input vars: " << os.str() << "\n";
  }
};

class HorizontalFuseMutator : private MixedModeMutator {
  public:
  
  Expr Transform(const Expr body) {
    support::Arena arena;
    graph_ = UpwardRankGraph::Create(&arena, body);
    for(auto& g: graph_.groups){
      if(g.size()<2){
        continue;
      }
      MakeCombinedOp(g);
    }
    this->Mutate(body);
    return ExprSubst(body, std::move(subst_map_));
  }
  //TODO(Chunwei Xia)
  Expr UpdateArg(){

  }

  // Transform calls.
  Expr Rewrite_(const CallNode* call, const Expr& post) {
    if(call->op.as<OpNode>()){
      static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
      if(fnoncomputational.get(Downcast<Op>(call->op), false)){
        return ExprMutator::VisitExpr_(call);
      }else if(op2item.count(call)){
        // Modify all children's argument
        std::vector<const ExprNode*> children = graph_.getExprChildren(GetRef<Call>(call));
        for(auto child: children){
          if(child->IsInstance<CallNode>()){
            auto child_call = Downcast<Call>(GetRef<Expr>(child));
            Array<Expr> new_args;
            for(const auto arg: child_call->args){
              if(arg.as<CallNode>() && arg.as<CallNode>()==call){
                new_args.push_back(op2item[call]);
              }else{
                new_args.push_back(arg);
              }
            }
            auto new_child = Call(child_call->op, new_args, child_call->attrs, child_call->type_args, child_call->span);
            VLOG(2) << "Make new_child " << new_child;
            subst_map_.insert({GetRef<Expr>(child), new_child});
          }else{
            LOG(FATAL) << GetRef<Call>(call) << " Child is not call\n";
            return ExprMutator::VisitExpr_(call);
          }
        }
      }else{
        return ExprMutator::VisitExpr_(call);
      }
    }
    return ExprMutator::VisitExpr_(call);
  }

  Call MakeCombinedOp(const Group& branches) {
    ICHECK(branches.size()>=2);
    Array<Expr> inputs, weights, fields, pre_ops;
    Array<IndexExpr> input_split_indices;
    Array<IndexExpr> output_split_indices;
    int32_t sum_of_input_channel = 0, sum_of_output_channel = 0;
    // Get conv2d params
    auto conv2d = branches[0][0];
    auto batch = (int32_t)GetConv2DInputBatchDim(conv2d);
    auto channels = (int32_t)GetConv2DInputChannelsDim(conv2d);
    auto height = (int32_t)GetConv2DInputHeightDim(conv2d);
    auto width = (int32_t)GetConv2DInputWidthDim(conv2d);
    auto kernel_height = (int32_t)GetConv2DWeightKernelHeightDim(conv2d);
    auto kernel_width = (int32_t)GetConv2DWeightKernelWidthDim(conv2d);
    auto input_dtype = GetConv2DInputDataType(conv2d);
    auto weight_dtype = GetConv2DInputDataType(conv2d);
    // Create arguments for function
    for(auto branch: branches){
      auto conv2d = branch[0];
      ICHECK(conv2d->args.size()>=2);
      inputs.push_back(conv2d->args[0]);
      weights.push_back(conv2d->args[1]);
      auto num_input_channel = (int32_t)GetConv2DInputChannelsDim(conv2d);
      auto num_output_channel = (int32_t)GetConv2DSuperChannelsDim(conv2d);
      sum_of_input_channel += num_input_channel;
      sum_of_output_channel += num_output_channel;
      if(input_split_indices.empty()){
        input_split_indices.push_back(IndexExpr(num_input_channel));
        output_split_indices.push_back(IndexExpr(num_output_channel));
      }else{
        input_split_indices.push_back(IndexExpr(num_input_channel) + input_split_indices.back());
        output_split_indices.push_back(IndexExpr(num_output_channel) + output_split_indices.back());
      }
      pre_ops.push_back(GetRef<Call>(conv2d));
    }
    // Does not need last indices
    input_split_indices.pop_back();
    output_split_indices.pop_back();
    //[1,3,5,5]*[2,3,1,1] ; [1,3,5,5]*[32,3,1,1]
    Array<PrimExpr> fn_input_shape = {
      PrimExpr(batch), 
      PrimExpr(sum_of_input_channel), 
      PrimExpr(height), 
      PrimExpr(width)
    };
    Array<PrimExpr> fn_weight_shape = {
      PrimExpr(sum_of_output_channel), 
      PrimExpr(channels), 
      PrimExpr(kernel_height), 
      PrimExpr(kernel_width)
    };
    Array<PrimExpr> fn_output_shape = {
      PrimExpr(batch), 
      PrimExpr(sum_of_output_channel), 
      PrimExpr(height), 
      PrimExpr(width)
    };
    Array<Var> params = {
      Var("hfused_inputs", tvm::TensorType(fn_input_shape, input_dtype)), 
      Var("hfused_weights", tvm::TensorType(fn_weight_shape, weight_dtype))
    };
    auto new_inputs = MakeConcatenate(Tuple(inputs), 1);
    auto new_weights = MakeConcatenate(Tuple(weights), 0);
    auto split_inputs = MakeSplit(params[0], Integer(2), 1);
    auto split_weights = MakeSplit(params[1], Integer(2), 0);
    // Modify ops in the body of function
    int i = 0;
    for(auto branch: branches){
      auto conv2d = branch[0];
      Array<Expr> call_new_args = {TupleGetItem(split_inputs, (int32_t)i), TupleGetItem(split_weights, (int32_t)i)};
      auto new_conv2d = Call(conv2d->op, call_new_args, conv2d->attrs, conv2d->type_args, conv2d->span);
      fields.push_back(new_conv2d);
      i++;
    }
    Array<Expr> arguments = {
      new_inputs, 
      new_weights
    };
    auto fn_ret = MakeConcatenate(Tuple(fields), 1);
    auto func = Function(params, fn_ret, tvm::TensorType(fn_output_shape, input_dtype), {});
    func = WithAttr(std::move(func), attr::kFusion, tvm::Integer(1));
    
    auto new_call = Call(func, arguments, Attrs());
    // Add following split op
    const int channel_dim = 1;
    auto new_split = MakeSplit(new_call, Integer(2), channel_dim);
    for(size_t i=0; i<fields.size(); ++i){
      auto conv2d = pre_ops[i];
      op2split.insert({conv2d.as<ExprNode>(), new_split});
      op2item.insert({conv2d.as<ExprNode>(), TupleGetItem(new_split, (int32_t)i)});
      subst_map_.insert({conv2d, fields[i]});
    }
    return Downcast<Call>(new_split);
  }

  private:
  using MixedModeMutator::VisitExpr_;
  UpwardRankGraph graph_;
  std::unordered_map<const ExprNode*, Expr> op2split;
  std::unordered_map<const ExprNode*, Expr> op2item;
  /* \brief map of Expr to Expr to substitute it with after running pass */
  ExprSubstMap subst_map_;
};

UpwardRankGraph UpwardRankGraph::Create(support::Arena* arena, const Expr& body){
  return Creator(arena).Transform(body);
}

// Expr HorizontalFusion(const Expr& e) { return HoriFusionMutator().Transform(e); }
Expr HorizontalFusion(const Expr e)  { 
  return HorizontalFuseMutator().Transform(e);
}

namespace transform {

Pass HorizontalFusion(int fuse_opt_level) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(HorizontalFusion(f));
        // if(fuse_opt_level < pc->opt_level){
        //   return Downcast<Function>(f);
        // }else{
        //   return Downcast<Function>(HorizontalFusion(f));
        // }
      };
  return CreateFunctionPass(pass_func, 0, "HorizontalFusion", {});
}

TVM_REGISTER_GLOBAL("relay._transform.HorizontalFusion").set_body_typed(HorizontalFusion);
}  // namespace transform

}  // namespace relay
}  // namespace tvm


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
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using ConstSet = std::unordered_set<Constant, ObjectPtrHash, ObjectPtrEqual>;
using Branch = std::vector<Expr>;
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
  // Data flow from input to output, thus the input is parent and output is child
  /*! \brief From child to parents */
  ExprMap<ExprSet> child_to_parent_map_;
  /*! \brief From parents to child */
  ExprMap<ExprSet> parent_to_children_map_;
  /*! \brief input variables of the relay graph */
  VarSet input_vars_;
  /*! \brief constant weights of the relay graph */
  ConstSet ops_weights_;
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

  ExprSet getExprChildren(Expr call){
    if(this->parent_to_children_map_.count(call)){
      return this->parent_to_children_map_[call];
    }else{
      return ExprSet();
    }
  }

  std::pair<bool, Expr> IsChildTuple(const Branch& branch){
    auto last_op_in_fist_branch = branch.back();
    if(!this->parent_to_children_map_.count(last_op_in_fist_branch)){
      return std::make_pair(false, Expr());
    }
    auto children_set = this->parent_to_children_map_[last_op_in_fist_branch];
    VLOG(1) << "last_op_in_branch: " << last_op_in_fist_branch << " child_0: " << *(children_set.begin());
    if(children_set.size() == 1){
      auto it = children_set.begin();
      if((*it).as<TupleNode>()){
        return std::make_pair(true, *it);
      }
    }
    return std::make_pair(false, Expr());
  }

  // Whether all the branches in a group sink to a concate op
  std::pair<bool, Expr> IsGroupSinkToTuple_(const Group& g){
    if(g.size()==0){
      return std::make_pair(false, Expr());
    }
    auto result_a = IsChildTuple(g[0]);
    if(!result_a.first){
      return std::make_pair(false, Expr());
    }
    for(auto branch: g){
      auto result_b = IsChildTuple(branch);
      if(result_a != result_b){
        return std::make_pair(false, Expr());
      }
    }
    return std::make_pair(true, result_a.second);
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
    // graph_.groups = this->FindFusedConv2DOps_();
    graph_.groups = this->FindFusedConv2DBlocksV2_();
    return std::move(graph_);
  }

 private:
  
  /*! \brief Internal arena. */
  support::Arena* arena_;
  UpwardRankGraph graph_;

  // For now we assume that FunctionNode is at the top level
  void VisitExpr_(const FunctionNode* f) {
    VLOG(2) << "Visit: " << GetRef<Function>(f);
    this->VisitExpr(f->body);
    // FindInputVars_(f->body);
    DumpDebugRelationMap(graph_.parent_to_children_map_);
    DebugDumpInputVars_();
    BuildUprankGraph_();
    graph_.DebugDump();
  }


  void VisitExpr_(const CallNode* c) {
    VLOG(2) << "Visit: " << GetRef<Call>(c);
    this->VisitExpr(c->op);
    UpdateRelationMap_(GetRef<Call>(c), c->args);
    for(auto& arg: c->args) {
      this->VisitExpr(arg);
    }
  }
  
  void VisitExpr_(const TupleNode* op){
    VLOG(2) << "Visit: " << GetRef<Tuple>(op);
    UpdateRelationMap_(GetRef<Tuple>(op), op->fields);
    for(auto& f: op->fields){
      this->VisitExpr(f);
    }
  }

  void VisitExpr_(const TupleGetItemNode* op){
    VLOG(2) << "Visit: " << GetRef<TupleGetItem>(op);
    UpdateRelationMap_(GetRef<TupleGetItem>(op), Array<Expr>({op->tuple}));
    this->VisitExpr(op->tuple);
  }

  void VisitExpr_(const OpNode* op) {
    std::ostringstream os;
    auto op_ref = GetRef<ObjectRef>(op);
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    if(fpattern.count(GetRef<Op>(op))){
      auto op_pattern = static_cast<OpPatternKind>(fpattern[GetRef<Op>(op)]);
      os << "name " << op->name << " ";
      VLOG(2) << os.str() << "pattern " << op_pattern << " VisitExpr OpNode";
    }
  }

  void VisitExpr_(const VarNode* op) {
    VLOG(2) << "Visit: Var: " << op->name_hint();
    graph_.input_vars_.insert(GetRef<Var>(op));
  }

  void VisitExpr_(const ConstantNode* op) {
    VLOG(2) << "Visit: Constant: " << op->data;
    graph_.ops_weights_.insert(GetRef<Constant>(op));
  }

  // child consume data produced by parents
  // parent_map_ is child->parents
  // child_map_ is parents->child
  void UpdateRelationMap_(Expr child, Array<Expr> parents) {
    for (size_t i = 0; i < parents.size(); ++i) {
      // Update parent->children
      auto p = parents[i];
      if (graph_.parent_to_children_map_.count(p) == 0) {
        auto children_set = ExprSet({child});
        graph_.parent_to_children_map_.insert({p, children_set});
      } else {
        graph_.parent_to_children_map_[p].insert(child);
      }

      // Update child->parents
      if (graph_.child_to_parent_map_.count(child) == 0) {
        graph_.child_to_parent_map_.insert({child, ExprSet({p})});
      } else {
        graph_.child_to_parent_map_[child].insert(p);
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

  void DumpDebugRelationMap(ExprMap<ExprSet>& r_map) {
    for (auto it = r_map.begin(); it != r_map.end(); ++it) {
      std::string child_op_type = GetExprNodeTypeStr(it->first);
      for (auto& to_expr: it->second) {
        std::string parent_op_type = GetExprNodeTypeStr(to_expr);
        VLOG(2) << child_op_type << " to " << parent_op_type << " " << it->first << "a->b" << to_expr;
      }
    }
  }

  // void FindInputVars_(Expr body) {
  //   auto expr_queue = std::queue<Expr>();
  //   expr_queue.push(body);
  //   while (!expr_queue.empty()) {
  //     auto current_expr = expr_queue.front();
  //     expr_queue.pop();
  //     if (current_expr.as<VarNode>()) {
  //       ICHECK(graph_.parent_map_.count(current_expr) == 0);
  //       graph_.input_vars_.insert(Downcast<Var>(current_expr));
  //     } else if (current_expr.as<ConstantNode>()) {
  //       // Save constant node as weights we need further
  //       ICHECK(graph_.parent_map_.count(current_expr) == 0);
  //       graph_.ops_weights_.insert(Downcast<Constant>(current_expr));
  //     } else if (graph_.parent_map_.count(current_expr)) {
  //       for (auto& parent_expr : graph_.parent_map_[current_expr]) {
  //         expr_queue.push(parent_expr);
  //       }
  //     } else {
  //       VLOG(2) << "Special case need to be handled " << current_expr << "\n";
  //     }
  //   }
  // }
  
  void AddNode(Expr op_expr){
    if(graph_.node_map.count(op_expr) == 0){
      UpwardRankGraph::Node* op_node = arena_->make<UpwardRankGraph::Node>();
      op_node->ref = op_expr.as<ExprNode>();
      op_node->index = graph_.forward_bfs_order.size();
      graph_.forward_bfs_order.push_back(op_node);
      graph_.node_map[op_expr] = op_node;
      if(!graph_.child_to_parent_map_.count(op_expr)){
        op_node->rank = 0;
      }else{
        for(auto& p: graph_.child_to_parent_map_[op_expr]){
          if(!graph_.node_map.count(p)){
            LOG_FATAL << "!graph_.node_map.count(p) " << op_expr << " parent: " << p;
          }
          graph_.node_map[op_expr]->rank = std::max(graph_.node_map[op_expr]->rank, graph_.node_map[p]->rank + 1);
        }
      }
      auto it = graph_.uprank_array_map.find(op_node->rank);
      if(it==graph_.uprank_array_map.end()){
        std::vector<Expr> expr_arr;
        graph_.uprank_array_map.insert({op_node->rank, {op_expr}});
      }else{
        (*it).second.push_back(op_expr);
      }
      std::ostringstream os;
      os << "Add " << GetRef<Expr>(op_node->ref) << " up rank " << op_node->rank << "\n";
      VLOG(2) << os.str();
    }
  }

  void BuildUprankGraph_() {
    // First add vars to graph
    ExprSet visited;
    std::queue<Expr> tmp_queue;
    for (auto op_var: graph_.input_vars_) {
      this->AddNode(op_var);
      tmp_queue.push(op_var);
    }
    for (auto op_const: graph_.ops_weights_) {
      this->AddNode(op_const);
    }
    // We start from vars and use bfs to traverse the whole expr
    while(!tmp_queue.empty()){
      auto top = tmp_queue.front();
      tmp_queue.pop();
      if(graph_.parent_to_children_map_.count(top)){
        for(auto child: graph_.parent_to_children_map_[top]){
          if(!visited.count(child)){
            this->AddNode(child);
            visited.insert(child);
            tmp_queue.push(child);
          }
        }        
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

  bool CanBatchNorOpsBeCombined(const CallNode* a, const CallNode* b){
    StructuralEqual eq;
    const auto* attrs_a = a->attrs.as<BatchNormAttrs>();
    const auto* attrs_b = b->attrs.as<BatchNormAttrs>();
    ICHECK(attrs_a);
    ICHECK(attrs_b);
    return eq(attrs_a->axis, attrs_b->axis) && eq(attrs_a->epsilon, attrs_b->epsilon) && 
      eq(attrs_a->scale, attrs_b->scale) && eq(attrs_a->center, attrs_b->center);
  }

  // For now we assume all ops with same uprank having same number of ops like resnext
  std::pair<bool, TOpPattern> AllTheSameOps(const std::vector<Expr>& arr_expr){
    if(arr_expr.size() == 0) {
      return std::make_pair(false, OpPatternKind::kOpaque);
    }

    bool all_same = true;
    TOpPattern pattern;
    if(arr_expr.front().as<CallNode>()) {
      auto first_op = Downcast<Call>(arr_expr.front());
      auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
      if(fpattern.count(Downcast<Op>(first_op->op))){
        pattern = fpattern[Downcast<Op>(first_op->op)];
        VLOG(1) << "GetOp " << first_op << " pattern: " << pattern;
      }else{
        VLOG(1) << "GetOp " << first_op << " cannot find pattern ";
      }

      for(size_t i=1; i<arr_expr.size(); ++i){
        auto op = arr_expr[i];
        if(first_op->attrs.as<Conv2DAttrs>()){
          if(!op.as<CallNode>()->attrs.as<Conv2DAttrs>() || 
            !CanConv2DOpsBeCombined(first_op.as<CallNode>(), op.as<CallNode>())){
            all_same = false;
            VLOG(1) << "Conv2d not same";
            break;
          }
        }else if(Downcast<Op>(first_op->op)==Op::Get("nn.batch_norm")){
          if(!op.as<CallNode>()->attrs.as<BatchNormAttrs>() || 
            !CanBatchNorOpsBeCombined(first_op.as<CallNode>(), op.as<CallNode>())){
            all_same = false;
            pattern = OpPatternKind::kOpaque;
            VLOG(1) << "batchnorm not same";
            break;
          }else{
            pattern = OpPatternKind::kInjective;
          }
        }else if(fpattern[Downcast<Op>(op.as<CallNode>()->op)] != pattern ||
          pattern > OpPatternKind::kInjective){
            VLOG(1) << "call not same";
          all_same = false;
          break;
        }
      }
    }else if(arr_expr.front().as<TupleGetItemNode>()){
      for(size_t i=1; i<arr_expr.size(); ++i){
        auto op = arr_expr[i];
        pattern = OpPatternKind::kBroadcast;
        if(!op.as<TupleGetItemNode>()){
          all_same = false;
          pattern = OpPatternKind::kOpaque;
          VLOG(1) << "TupleGetItem not same";
          break;
        }
      }
    }
    else {
      LOG(INFO) << "Not implement for " << arr_expr.front();
      all_same = false;
      pattern = OpPatternKind::kOpaque;
    }
    return std::make_pair(all_same, pattern);
  }

  /**
   * @brief Now we add batch_norm and relu, process layer by layer
   * Call(conv2d)    Call(conv2d)
   *    |               |
   * Call(bn)        Call(bn)
   *    |               |
   * TupleGetItem(0) TupleGetItem(0)
   *    |               |
   * Call(relu)      Call(relu)
   *    \               /
   *         Tuple()
   *           |
   *     Call(concatenate)
   * @return std::vector<Group> 
   */
  //TODO(Chunwei Xia) For now we assume all the fuseable ops having the same uprank
  std::vector<Group> FindFusedConv2DBlocksV2_() {
    std::vector<Group> can_be_fused_arr;
    size_t last_uprank_size = 0;
    Group g;
    std::function<void()> handle_group = [&](){
      if(!g.empty()){
        can_be_fused_arr.push_back(g);
        LOG(INFO) << "Create a group with branches: " << g.size();
        g.clear(); 
      }
    };
    // The we should sort the uprank_array_map based on the uprank
    // First get the max uprank
    size_t max_uprank = 0;
    for(auto kv: graph_.uprank_array_map){
      max_uprank = std::max(max_uprank, kv.first);
    }
    std::vector<std::vector<Expr>> arr_arr_expr(max_uprank+1);
    for(auto kv: graph_.uprank_array_map){
      arr_arr_expr[kv.first] = kv.second;
    }
    for(auto expr_with_same_uprank: arr_arr_expr){
      if(expr_with_same_uprank.size() > 1) {
        auto result = AllTheSameOps(expr_with_same_uprank);
        VLOG(2) << result.first << " " << result.second;
        // All the ops are the same
        if(result.first){
          // Conv2d like ops, try to fuse following ops to them
          if(result.second > OpPatternKind::kInjective){ 
            handle_group();
            for(auto expr: expr_with_same_uprank){
              g.push_back(Branch({expr}));
            }
            last_uprank_size = expr_with_same_uprank.size();
          }else if(result.second <= OpPatternKind::kInjective){
            // Fuse with last uprank
            if(!g.empty()){
              if(last_uprank_size == expr_with_same_uprank.size()){
                size_t i=0;
                for(auto expr: expr_with_same_uprank){
                  g[i].push_back(expr);
                  i++;
                }
                last_uprank_size = expr_with_same_uprank.size();
              }else{
                handle_group();
                // TODO(Chunwei Xia) Add implementation to process more ops
              }
            }else{
              LOG_FATAL << "Not support fusion starting from \
                ops with OpPatternKind less than kBroadcast yet";
            }
          }else{
            LOG_FATAL << "Not implement for this case";
          }
        }else{
          //TODO(Chunwei Xia) Add implementation
        }
      }
    }
    handle_group();
    LOG(INFO) << "Find " << can_be_fused_arr.size() << " groups for horizontal fusion";
    return can_be_fused_arr;
  }

  //TODO(Chunwei Xia) For now we assume all the fuseable ops having the same uprank
  // Now we add batch_norm and relu, process layer by layer
  std::vector<Group> FindFusedConv2DOps_() {
    std::vector<Group> can_be_fused_arr;
    Group g;
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
      
      for(size_t i=1; i<it->second.size(); ++i){
        auto op = it->second[i];
        if(op.as<CallNode>()->attrs.as<Conv2DAttrs>()){
          if(CanConv2DOpsBeCombined(first_op.as<CallNode>(), op.as<CallNode>())){
            g.push_back(Branch({op}));
          }
        }
      }
      g.push_back(Branch({first_op}));
      VLOG(2) << "Find group with size " << g.size();
      can_be_fused_arr.push_back(g);
    }
    return can_be_fused_arr;
  }

  Expr ConcatInputs(const Group& branches, size_t arg_index) {
    Array<Expr> inputs;
    for(auto branch: branches) {
      auto conv2d = branch[0];
      ICHECK(arg_index < Downcast<Call>(conv2d)->args.size());
      inputs.push_back(Downcast<Call>(conv2d)->args[arg_index]);
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

class GetArgsToBeReplaced : public ExprVisitor{
public:
  ExprSet GetArgs(){
    return this->args_;
  }

  void VisitExpr_(const VarNode* op) {
    this->args_.insert(GetRef<Var>(op));
    ExprVisitor::VisitExpr_(op);
  }
  
  void VisitExpr_(const ConstantNode* op) {
    this->args_.insert(GetRef<Constant>(op));
    ExprVisitor::VisitExpr_(op);
  }

  private:
  ExprSet args_;
};

class HorizontalFuseMutator : private MixedModeMutator {
  public:
  
  Expr Transform(const Expr body) {
    support::Arena arena;
    graph_ = UpwardRankGraph::Create(&arena, body);
    // return Expr();
    for(auto& g: graph_.groups){
      if(g.size()<2){
        continue;
      }
      // MakeCombinedOp(g);
      VLOG(2) << "Branch start:";
      for(auto expr: g[0]){
        VLOG(2) << expr;
      }
      VLOG(2) << "Branch end";
      MakeBlockCombinedOp(g);
    }
    for(auto kv: subst_map_){
      LOG(INFO) << "replace: " << kv.first << " with: " << kv.second;
    }
    return this->Mutate(body);
    // return ExprSubst(body, std::move(subst_map_));
  }

  Expr VisitExpr_(const FunctionNode* op) {
    if(op->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Expr>(op);
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) {
    // First check whether it self needs to be replace
    if(subst_map_.count(GetRef<Call>(call))){
      return subst_map_[GetRef<Call>(call)];
    }
    // Second check whether his parents needs to be replace
    bool changed = false;
    Array<Expr> new_args;
    for(auto& arg: call->args){
      if(subst_map_.count(arg)){
        new_args.push_back(subst_map_[arg]);
        changed = true;
      }else{
        new_args.push_back(arg);
      }
    }
    if(changed){
      return Call(call->op, new_args, call->attrs);
    }else{
      return ExprMutator::VisitExpr_(call);
    }
  }

  Expr Rewrite_(const TupleNode* tuple, const Expr& post) {
    if(subst_map_.count(GetRef<Tuple>(tuple))){
      return subst_map_[GetRef<Tuple>(tuple)];
    }
    Array<Expr> new_fields;
    bool changed = false;
    for(auto f: tuple->fields){
      if(subst_map_.count(f)){
        new_fields.push_back(subst_map_[f]);
        changed = true;
      }else{
        new_fields.push_back(f);
      }
    }
    if(changed){
      return Tuple(new_fields);
    }else{
      return ExprMutator::VisitExpr_(tuple);
    }
  }

  Expr Rewrite_(const TupleGetItemNode* tuple_get, const Expr& post) {
    if(subst_map_.count(GetRef<TupleGetItem>(tuple_get))){
      return subst_map_[GetRef<TupleGetItem>(tuple_get)];
    }else{
      return ExprMutator::VisitExpr_(tuple_get);
    }
  }
  
  //TODO(Chunwei Xia)
  // Expr UpdateArg(){}

  // Transform calls.
  // Expr Rewrite_(const CallNode* call, const Expr& post) {
  //   if(call->op.as<OpNode>()){
  //     static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
  //     if(fnoncomputational.get(Downcast<Op>(call->op), false)){
  //       return ExprMutator::VisitExpr_(call);
  //     }else if(op2item.count(call)){
  //       // Modify all children's argument
  //       std::vector<const ExprNode*> children = graph_.getExprChildren(GetRef<Call>(call));
  //       for(auto child: children){
  //         if(child->IsInstance<CallNode>()){
  //           auto child_call = Downcast<Call>(GetRef<Expr>(child));
  //           Array<Expr> new_args;
  //           for(const auto arg: child_call->args){
  //             if(arg.as<CallNode>() && arg.as<CallNode>()==call){
  //               new_args.push_back(op2item[call]);
  //             }else{
  //               new_args.push_back(arg);
  //             }
  //           }
  //           auto new_child = Call(child_call->op, new_args, child_call->attrs, child_call->type_args, child_call->span);
  //           VLOG(2) << "Make new_child " << new_child;
  //           subst_map_.insert({GetRef<Expr>(child), new_child});
  //         }else{
  //           LOG(FATAL) << GetRef<Call>(call) << " Child is not call\n";
  //           return ExprMutator::VisitExpr_(call);
  //         }
  //       }
  //     }else{
  //       return ExprMutator::VisitExpr_(call);
  //     }
  //   }
  //   return ExprMutator::VisitExpr_(call);
  // }

  // Original we first concat inputs and weights so that
  // at the TE stage we do not need to concat the weight
  // and we only need to modify the tensor expression.
  // But we plan to do more transformation at the te level,
  // so in this pass we only put the fused ops in one function
  // and does not change the original inputs
  // TODO(Chunwei Xia) we need to find all the input and output ops of the block
  Call MakeBlockCombinedOp(const Group& branches){
    // 1. Make params and arguments of function (input op of the block) 
    
    ExprMap<Expr> var_constant_map;
    auto getter = GetArgsToBeReplaced();
    for(auto branch: branches){
      getter.VisitExpr(branch.back());
    }
    Array<Var> params;
    Array<Expr> arguments;
    int index_param=0;
    ExprMap<Expr> args_subst_map;
    for(auto& expr: getter.GetArgs()){
      VLOG(2) << "Expr to replace: " << expr;
      arguments.push_back(expr);
      auto new_var = Var("p_" + std::to_string(index_param), 
            GetRef<tvm::TensorType>(expr->type_as<tvm::TensorTypeNode>()));
      if(expr.as<ConstantNode>()){
        var_constant_map.insert({new_var, expr});
      }
      params.push_back(new_var);
      args_subst_map.insert({expr, new_var});
      index_param++;
    }
    // 2. Make body of function (only need the output of the block)
    Expr body;
    tvm::Type ret_type;
    auto result = graph_.IsGroupSinkToTuple_(branches);
    if(result.first){
      body = result.second;
      VLOG(2) << body;
    }else{
      // We make a tuple node
      Array<Expr> fields;
      for(auto branch: branches){
        fields.push_back(branch.back());
      }
      body = Tuple(fields);
    }
    auto new_body = ExprSubst(body, args_subst_map);
    VLOG(2) << "new_body: " << new_body;
    // TODO(Chunwei Xia) Check whether the tuple's child is concat
    ret_type = GetRef<tvm::TupleType>(body->type_as<tvm::TupleTypeNode>());

    // 3. Make the function and warp it to a call op
    auto func = Function(params, new_body, ret_type, {});
    func = WithAttr(std::move(func), attr::kFusion, tvm::Integer(1));
    func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(1));
    // For fusing constant in lowerTE
    Map<Expr, Expr> ref_var_constant_map(var_constant_map);
    func = WithAttr(std::move(func), std::string("var_constant_map"), ref_var_constant_map);
    auto new_call = Call(func, arguments, Attrs());

    // 4. Feed the subst_map re-setup the relationship with the blocks's children
    if(result.first){
      if(graph_.parent_to_children_map_.count(body)){
        subst_map_.insert({body, new_call});
      }
      // The concate is the output op, do nothing
    }else{
      // Create TupleGetItem op to get returned result from function
      size_t i = 0;
      for(auto& branch: branches){
        auto last_op = branch.back();
        subst_map_.insert({last_op, TupleGetItem(new_call, i)});
      }
    }

    return new_call;
  }

  
  Call MakeCombinedOp(const Group& branches) {
    ICHECK(branches.size()>=2);
    Array<Expr> inputs, weights, fields, pre_ops;
    Array<IndexExpr> input_split_indices;
    Array<IndexExpr> output_split_indices;
    int32_t sum_of_input_channel = 0, sum_of_output_channel = 0;
    // Get conv2d params
    auto conv2d = branches[0][0].as<CallNode>();
    auto batch = (int32_t)GetConv2DInputBatchDim(conv2d);
    auto input_channels = (int32_t)GetConv2DInputChannelsDim(conv2d);
    auto height = (int32_t)GetConv2DInputHeightDim(conv2d);
    auto width = (int32_t)GetConv2DInputWidthDim(conv2d);
    auto kernel_height = (int32_t)GetConv2DWeightKernelHeightDim(conv2d);
    auto kernel_width = (int32_t)GetConv2DWeightKernelWidthDim(conv2d);
    auto output_channels = (int32_t)GetConv2DSuperChannelsDim(conv2d);
    auto input_dtype = GetConv2DInputDataType(conv2d);
    auto weight_dtype = GetConv2DInputDataType(conv2d);
    // Create arguments for function
    for(auto branch: branches){
      auto conv2d = branch[0].as<CallNode>();
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
    // TODO(Chunwei Xia) Whether we concat at the batch axis or the channel axis
    // Organize as group convolution
    Array<PrimExpr> fn_input_shape = {
      PrimExpr((int32_t)(batch * branches.size())), 
      PrimExpr(input_channels), 
      PrimExpr(height), 
      PrimExpr(width)
    };
    Array<PrimExpr> fn_weight_shape = {
      PrimExpr(sum_of_output_channel), 
      PrimExpr(input_channels), 
      PrimExpr(kernel_height), 
      PrimExpr(kernel_width)
    };
    Array<PrimExpr> fn_output_shape = {
      PrimExpr((int32_t)(batch * branches.size())), 
      PrimExpr(output_channels), 
      PrimExpr(height), 
      PrimExpr(width)
    };
    Array<Var> params = {
      Var("hfused_inputs", tvm::TensorType(fn_input_shape, input_dtype)), 
      Var("hfused_weights", tvm::TensorType(fn_weight_shape, weight_dtype))
    };
    auto new_inputs = MakeConcatenate(Tuple(inputs), 0);
    auto new_weights = MakeConcatenate(Tuple(weights), 0);
    auto split_inputs = MakeSplit(params[0], Integer(branches.size()), 0);
    auto split_weights = MakeSplit(params[1], Integer(branches.size()), 0);
    // Modify ops in the body of function
    int i = 0;
    for(auto branch: branches) {
      auto conv2d = branch[0].as<CallNode>();
      Array<Expr> call_new_args = {TupleGetItem(split_inputs, (int32_t)i), TupleGetItem(split_weights, (int32_t)i)};
      auto new_conv2d = Call(conv2d->op, call_new_args, conv2d->attrs, conv2d->type_args, conv2d->span);
      fields.push_back(new_conv2d);
      i++;
    }
    Array<Expr> arguments = {
      new_inputs, 
      new_weights
    };
    auto fn_ret = MakeConcatenate(Tuple(fields), 0);
    auto func = Function(params, fn_ret, tvm::TensorType(fn_output_shape, input_dtype), {});
    func = WithAttr(std::move(func), attr::kFusion, tvm::Integer(1));
    func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(1));
    // Wrap function into CallNode
    auto new_call = Call(func, arguments, Attrs());
    // Add following split op
    const int channel_dim = 0;
    auto new_split = MakeSplit(new_call, Integer(branches.size()), channel_dim);
    for(size_t i=0; i<fields.size(); ++i){
      auto conv2d = pre_ops[i];
      op2split.insert({conv2d.as<ExprNode>(), new_split});
      op2item.insert({conv2d.as<ExprNode>(), TupleGetItem(new_split, (int32_t)i)});
      subst_map_.insert({conv2d, fields[i]});
    }
    return Downcast<Call>(new_split);
  }

  private:
  
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
        // return Downcast<Function>(HorizontalFusion(f));
        VLOG(2) << "hfuse_opt_level: " << fuse_opt_level << ", pc->opt_level: " << pc->opt_level;
        if(fuse_opt_level > pc->opt_level){
          return Downcast<Function>(f);
        }else{
          return Downcast<Function>(HorizontalFusion(f));
        }
      };
  return CreateFunctionPass(pass_func, 2, "HorizontalFusion", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.HorizontalFusion").set_body_typed(HorizontalFusion);
}  // namespace transform

}  // namespace relay
}  // namespace tvm

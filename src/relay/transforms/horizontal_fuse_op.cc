
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
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include <queue>

#include "../../support/arena.h"

namespace tvm {
namespace relay {

using support::LinkedList;
using support::LinkNode;
template <typename X>
using VarMap = std::unordered_map<Expr, X, ObjectPtrHash, ObjectPtrEqual>;
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
    const Expr* ref{nullptr};
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
  VarMap<Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> forward_bfs_order;

  /*! \brief Dump the graph into string. */
  void DebugDump() {
    std::ostringstream os;
    for (size_t i = 0; i < forward_bfs_order.size(); ++i) {
      Node* node = forward_bfs_order[i];
      os << "node[" << i << "], " << *(node->ref) << " uprank " << node->rank;
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

 public:
  class Creator;
};



// class HoriFusionMutator : private ExprVisitor {
class UpwardRankGraph::Creator : private ExprVisitor {
 public:
  explicit Creator(support::Arena* arena) : arena_(arena) {}

  Expr Transform(const Expr& body) {
    this->VisitExpr(body);
    return body;
  }

 private:
  VarMap<Array<Expr>> parent_map_;  // From child to parents
  VarMap<Array<Expr>> child_map_;   // From parents to child
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> input_vars_;
  std::unordered_set<Constant, ObjectPtrHash, ObjectPtrEqual> ops_weights_;
  /*! \brief Internal arena. */
  support::Arena* arena_;
  UpwardRankGraph graph_;
  void VisitExpr_(const CallNode* c) {
    LOG(INFO) << "********* VisitExpr CallNode: ***********\n";
    this->VisitExpr(c->op);
    UpdateRelationMap_(GetRef<Call>(c), c->args);
  }
  
  // For now we assume that FunctionNode is at the top level
  void VisitExpr_(const FunctionNode* f) {
    LOG(INFO) << "********* VisitExpr FunctionNode: ***********\n" << GetRef<Function>(f);
    this->VisitExpr(f->body);
    FindInputVars_(f->body);
    DumpDebugRelationMap(parent_map_);
    DebugDumpInputVars_();
    graph_.DebugDump();
    BuildUprankGraph_();
  }

  void VisitExpr_(const OpNode* op) {
    std::ostringstream os;
    auto op_ref = GetRef<ObjectRef>(op);
    os << "name " << op->name << " ";
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    auto op_pattern = static_cast<OpPatternKind>(fpattern[GetRef<Op>(op)]);
    static auto ftvmcompute = Op::GetAttrMap<FTVMCompute>("FTVMCompute");
    //   auto op_fcompute = static_cast<FTVMCompute>(ftvmcompute[GetRef<Op>(op)]);
    LOG(INFO) << os.str() << "pattern " << op_pattern
              << " ********* VisitExpr OpNode: ***********\n";
  }

  void VisitExpr_(const VarNode* op) {
    LOG(INFO) << "********* name: " << op->name_hint() << " VisitExpr VarNode: ***********\n";
  }

  void VisitExpr_(const TupleNode* op) { UpdateRelationMap_(GetRef<Tuple>(op), op->fields); }

  // child consume data produced by parents
  // parent_map_ is child->parents
  // child_map_ is parents->child
  void UpdateRelationMap_(Expr child, Array<Expr> parents) {
    for (size_t i = 0; i < parents.size(); ++i) {
      auto arg = parents[i];
      this->VisitExpr(arg);
      if (parent_map_.count(child) == 0) {
        auto new_arr = Array<Expr>();
        new_arr.push_back(arg);
        parent_map_[child] = new_arr;
      } else {
        parent_map_[child].push_back(arg);
      }
      if (child_map_.count(arg) == 0) {
        auto new_arr = Array<Expr>();
        new_arr.push_back(child);
        child_map_[arg] = new_arr;
      } else {
        child_map_[arg].push_back(child);
      }
    }
  }

  void DumpDebugRelationMap(VarMap<Array<Expr>>& r_map) {
    for (auto it = r_map.begin(); it != r_map.end(); ++it) {
      for (size_t i = 0; i < it->second.size(); ++i) {
        std::string op_type;
        if (it->second[i].as<ConstantNode>()) {
          op_type = "constant ";
        } else if (it->second[i].as<VarNode>()) {
          op_type = "variable ";
        } else if (it->second[i].as<TupleNode>()) {
          op_type = "tuple ";
        } else {
          op_type = "call ";
        }
        LOG(INFO) << op_type << " " << it->first << "->" << it->second[i] << "\n";
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
        ICHECK(parent_map_.count(current_expr) == 0);
        input_vars_.insert(Downcast<Var>(current_expr));
      } else if (current_expr.as<ConstantNode>()) {
        // Save constant node as weights we need further
        ICHECK(parent_map_.count(current_expr) == 0);
        ops_weights_.insert(Downcast<Constant>(current_expr));
      } else if (parent_map_.count(current_expr) == 1) {
        for (auto parent_expr : parent_map_[current_expr]) {
          expr_queue.push(parent_expr);
        }
      } else {
        LOG(INFO) << "Special case need to be handled " << current_expr << "\n";
      }
    }
  }
  
  void AddNode(Expr obj){
    if(graph_.node_map.count(obj) == 0){
      UpwardRankGraph::Node* op_node = arena_->make<UpwardRankGraph::Node>();
      op_node->ref = &obj;
      op_node->index = graph_.forward_bfs_order.size();
      graph_.forward_bfs_order.push_back(op_node);
      graph_.node_map[obj] = op_node;
      if(parent_map_[obj].size() == 0){
        op_node->rank = 0;
      }else{
        for(auto p: parent_map_[obj]){
          ICHECK(graph_.node_map.count(p));
          graph_.node_map[obj]->rank = std::max(graph_.node_map[obj]->rank, graph_.node_map[p]->rank + 1);
        }
      }
      // std::ostringstream os;
      // os << "Add " << obj << " up rank " << op_node->rank << "\n";
      // LOG(INFO) << os.str();
    }
  }

  void BuildUprankGraph_() {
    // First add weights and vars to graph
    for (auto it = ops_weights_.begin(); it != ops_weights_.end(); ++it) {
      this->AddNode(*it);
    }
    std::queue<Expr> tmp_queue;
    for (auto it = input_vars_.begin(); it != input_vars_.end(); ++it) {
      this->AddNode(*it);
      tmp_queue.push(*it);
    }
    // We start from vars and use bfs to traverse the hole expr
    while(!tmp_queue.empty()){
      auto top = tmp_queue.front();
      tmp_queue.pop();
      if(child_map_.count(top)==0){
        LOG_WARNING << "Can not find " << top << "\n";
      }
      for(auto child: child_map_[top]){
        this->AddNode(child);
        tmp_queue.push(child);
      }
    }
  }

  void ConcatWeights_(Array<Expr>& weights){
    for(auto w: weights){
      ICHECK(w.as<ConstantNode>());
    }
    
  }

  void DebugDumpInputVars_() {
    std::ostringstream os;
    for(auto var: input_vars_) {
    // for (auto it = input_vars_.begin(); it != input_vars_.end(); ++it) {
      os << var << " ";
    }
    LOG(INFO) << "input vars: " << os.str() << "\n";
  }
};

// Expr HorizontalFusion(const Expr& e) { return HoriFusionMutator().Transform(e); }
Expr HorizontalFusion(const Expr& e) { 
  support::Arena arena;
  return UpwardRankGraph::Creator(&arena).Transform(e);
}

namespace transform {
Pass HorizontalFusion() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(HorizontalFusion(f));
      };
  return CreateFunctionPass(pass_func, 0, "HorizontalFusion", {});
}

TVM_REGISTER_GLOBAL("relay._transform.HorizontalFusion").set_body_typed(HorizontalFusion);
}  // namespace transform

}  // namespace relay
}  // namespace tvm

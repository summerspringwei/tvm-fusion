#include "replace_tensors_in_expr_stmt.h"

#include <set>

namespace tvm{
namespace tir{
namespace transforms{
using namespace tvm::tir;


// TODO(Chunwei Xia) For now we only condier ProducerLoad and ProducerStore
class GetTensorsFromStmtExpr : public StmtExprVisitor {
  public:

  void VisitExpr_(const ProducerLoadNode* op) final {
    auto tensor = Downcast<te::Tensor>(op->producer);
    VLOG(1) << GetRef<ProducerLoad>(op) << " tensor: " << tensor << " hash: " << ObjectPtrHash()(tensor);
    used_tensors_.insert(tensor);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ProducerStoreNode* op) final {
    auto tensor = Downcast<te::Tensor>(op->producer);
    used_tensors_.insert(tensor);
    VLOG(1) << tensor << " hash: " << ObjectPtrHash()(tensor);
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_set<te::Tensor> GetTensors(){
    return this->used_tensors_;
  }
  private:
  std::unordered_set<te::Tensor> used_tensors_;
};

std::unordered_set<te::Tensor> FGetTensorsFromStmtExpr(const tir::Stmt& stmt){
  auto getter = GetTensorsFromStmtExpr();
  getter(stmt);
  return getter.GetTensors();
}

// Replace all Tensors in StmtExpr
class ReplaceTensorsInStmtExpr : public StmtExprMutator {
  public:
  ReplaceTensorsInStmtExpr(std::unordered_map<te::Tensor, te::Tensor> replace_map)
    : replace_map_(replace_map){};

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    auto ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<ProducerLoadNode>();
    auto tensor = Downcast<te::Tensor>(op->producer);
    if(this->replace_map_.count(tensor)){
      VLOG(1) << "replace " << tensor << " hash: " << ObjectPtrHash()(tensor) << " with "
        << this->replace_map_[tensor] << " hash: " << ObjectPtrHash()(this->replace_map_[tensor]);
      return ProducerLoad(this->replace_map_[tensor], op->indices);
    }else{
      return GetRef<ProducerLoad>(op);
    }
  }
  
  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    auto tensor = Downcast<te::Tensor>(op->producer);
    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<ProducerStoreNode>();
    if(this->replace_map_.count(tensor)){
      VLOG(1) << "replace " << tensor << " hash: " << ObjectPtrHash()(tensor) << " with "
        << this->replace_map_[tensor] << " hash: " << ObjectPtrHash()(this->replace_map_[tensor]);
      return ProducerStore(this->replace_map_[tensor], op->value, op->indices, op->span);
    }else{
      return GetRef<ProducerStore>(op);
    }
  }

  Stmt VisitStmt_(const ProducerRealizeNode* op) final {
    auto tensor = Downcast<te::Tensor>(op->producer);;
    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<ProducerRealizeNode>();
    if(this->replace_map_.count(tensor)){
      VLOG(1) << "replace " << tensor << " hash: " << ObjectPtrHash()(tensor) << " with "
        << this->replace_map_[tensor] << " hash: " << ObjectPtrHash()(this->replace_map_[tensor]);
      return ProducerRealize(this->replace_map_[tensor], op->bounds, op->condition, op->body, op->storage_scope);
    }else{
      return GetRef<ProducerRealize>(op);
    }
  }

  private:
  std::unordered_map<te::Tensor, te::Tensor> replace_map_;
};

Stmt FReplaceDataProducer(Stmt& stmt, std::unordered_map<te::Tensor, te::Tensor>& replace_map){
  return ReplaceTensorsInStmtExpr(replace_map)(stmt);
}

// Print all tensors in StmtExpr using tir::PostOderVisit api
void FVerifyTensorConnect(const te::Tensor& tensor){
  Array<te::Tensor> ret;
  std::unordered_set<te::Tensor> visited;
  std::function<void(te::Tensor)> f_verify = [&](te::Tensor tensor) {
    if(auto compute_op = tensor->op.as<te::ComputeOpNode>()){
      for (auto& e : compute_op->body) {
        tir::PostOrderVisit(e, [&ret, &visited, &f_verify](const ObjectRef& n) {
          if (auto* pload = n.as<tir::ProducerLoadNode>()) {
            te::Tensor t = Downcast<te::Tensor>(pload->producer);
            VLOG(1) << t << " hash: " << ObjectPtrHash()(t);
            if (!visited.count(t)) {
              ret.push_back(t);
              visited.insert(t);
            }
            f_verify(t);
          }
        });
      }
    }else if(auto placeholder_op = tensor->op.as<te::PlaceholderOpNode>()) {
      VLOG(1) << placeholder_op;
    }else{
      LOG_DFATAL << "Error";
    }
  };
  VLOG(1) << tensor << " hash: " << ObjectPtrHash()(tensor);
  f_verify(tensor);
}

}
}
}

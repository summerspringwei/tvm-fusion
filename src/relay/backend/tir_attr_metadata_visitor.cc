
#include "tir_attr_metadata_visitor.h"

namespace tvm {
namespace relay {
  class TIRAttrMetaDataVisitor: public ExprVisitor {
    public:
    Array<te::Tensor> GetOutputTensors(const Expr& body){
      // Only Visit FunctionNode
      if(body.as<relay::FunctionNode>()){
        this->VisitExpr(body);
      }
      return this->output_tensors_;
    }
    void VisitExpr_(const CallNode* call_node){
      if (const GlobalVarNode* gvn = call_node->op.as<GlobalVarNode>()) {
      if (const TIRCallAttrs* attrs = call_node->attrs.as<TIRCallAttrs>()) {
        VLOG(2) << "call_node->attrs.as<TIRCallAttrs>()";
        if(attrs->metadata.count("kFusion")){
          for(auto t: Downcast<Array<te::Tensor>>(attrs->metadata["output_tensors"])){
            this->output_tensors_.push_back(t);
          }
        }
      }
      }
      for(auto arg: call_node->args){
        this->VisitExpr(arg);
      }
    }

  private:
    Array<te::Tensor> output_tensors_;
  };

  Array<te::Tensor> GetOutputTensorsFromRelayFunc(const Expr& body){
    return TIRAttrMetaDataVisitor().GetOutputTensors(body);
  }

  Map<GlobalVar, Array<te::Tensor>> GetPerVarTensorsFromIRModule(const IRModule& mod){
    auto var_tensors_map = Map<GlobalVar, Array<te::Tensor>>();
    
    // Get kFusion output_tensors
    for(auto ele: mod->functions){
      VLOG(2) << ele.first << ele.second;
      auto output_tensors = GetOutputTensorsFromRelayFunc(ele.second);
      if(output_tensors.size()==0){
        continue;
      }
      var_tensors_map.Set(ele.first, output_tensors);
      VLOG(2) << "GetOutputTensorsFromRelayFunc:";
      for(auto t: output_tensors){
        VLOG(2) << t;
      }
    }

    return var_tensors_map;
  }

  void PrintVarTensorMap(const Map<GlobalVar, Array<te::Tensor>>& var_tensor_map) {
    for(auto ele: var_tensor_map) {
      VLOG(2) << ele.first;
      for(auto t: ele.second) {
        VLOG(2) << t;
      }
    }
  }
}
}
#include <vector>
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/engine.h>
#include <ATen/core/functional.h>
#include <iostream>

using namespace std;
void print_node(std::shared_ptr<torch::autograd::Node>& node,std::vector<torch::autograd::Variable>& variable_list);

void
print_edge_list(torch::autograd::edge_list& roots,std::vector<torch::autograd::Variable>& variable_list){
  for (auto edge : roots) {
    print_node(edge.function,variable_list);
  }
}

void
print_node(std::shared_ptr<torch::autograd::Node>& node,std::vector<torch::autograd::Variable>& variable_list){
  int id=-1;
  for(int i=0;i<variable_list.size();i++){
    if(variable_list[i].try_get_grad_accumulator()== node){
      id=i;
      break;
    }
  }
  cout << node->name() << " id:" << id << endl;
  print_edge_list(node->next_edges(),variable_list);
}

int
main(){
  auto x = torch::ones({2,2}, torch::requires_grad());
  auto y = torch::ones({2,2}, torch::requires_grad());
  std::vector<torch::autograd::Variable> vec = {x,y};
  auto z = vec[0]+vec[1];
  torch::autograd::Variable vz = z;
  torch::autograd::edge_list roots { vz.gradient_edge() };
  if (!roots[0].function) {
    throw std::runtime_error("Differentiated tensor not require grad");
  }
  print_edge_list(roots,vec);
  
  return 0;
}


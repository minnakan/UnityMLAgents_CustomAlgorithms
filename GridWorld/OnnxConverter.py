import torch
from torch.nn import Parameter
from stable_baselines3 import DQN

# Wrap the model in a forward pass function
class ModelWrapper(torch.nn.Module):
    def __init__(self, qnet,discrete_output_sizes):
        super(ModelWrapper, self).__init__()
        self.qnet = qnet

        version_number = torch.Tensor([3])
        self.version_number = Parameter(version_number, requires_grad=False)

        memory_size = torch.Tensor([0])
        self.memory_size = Parameter(memory_size, requires_grad=False)

        output_shape=torch.Tensor([discrete_output_sizes])
        self.discrete_shape = Parameter(output_shape, requires_grad=False)
    
    def forward(self,visual_obs: torch.tensor,mask: torch.tensor ):
        qnet_result = self.qnet(visual_obs)
        qnet_result = torch.mul(qnet_result, mask)
        action = torch.argmax(qnet_result, dim=1, keepdim=True)
        return [action], self.discrete_shape, self.version_number, self.memory_size
    
model = DQN.load("unity_model.zip", device="cpu")
onnxable_model = model.policy.q_net
num_actions = 5
input_shape = model.observation_space.shape
dummy_visual_obs = torch.randn(1, *input_shape)
dummy_action_masks = torch.ones(1, num_actions)

onnx_path = "dqn_model.onnx"
torch.onnx.export(ModelWrapper(onnxable_model,[num_actions]), 
                  (dummy_visual_obs, dummy_action_masks), 
                  onnx_path, 
                  opset_version=9, 
                  input_names=["obs_0", "action_masks"],
                  output_names=["discrete_actions", "discrete_action_output_shape",
                  "version_number", "memory_size"],
                  dynamic_axes={'obs_0': {0: 'batch'},
                  'action_masks': {0: 'batch'},
                  'discrete_actions': {0: 'batch'},
                  'discrete_action_output_shape': {0: 'batch'}
                 }
                   )

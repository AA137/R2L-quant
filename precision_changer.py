import torch
from pytorch_quantization import tensor_quant

in_path = 'R2L_Blender_Models/lego.tar'
tar = torch.load(in_path)
sd = tar['network_fn_state_dict']
new_sd = {k:tensor_quant.fake_tensor_quant(v, v.abs().max()) for k,v in sd.items()}
sd['network_fn_state_dict'] = new_sd
torch.save(sd, 'lego_quant8.tar')

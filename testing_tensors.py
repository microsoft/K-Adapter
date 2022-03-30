import torch
import numpy as np 

start_ids = torch.Tensor(8, 128)
start_ids_unsqueezed = start_ids.unsqueeze(1)
print(start_ids_unsqueezed)

a = torch.FloatTensor(8,128,1024)

entity_output = torch.bmm(start_ids_unsqueezed, a)

entity_output_squeezed = entity_output.squeeze(1)

print()
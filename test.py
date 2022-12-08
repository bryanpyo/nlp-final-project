import torch
from config import opt
from utils import get_dataloader
from tqdm import tqdm

print('Print whether torch cuda is available')
print(torch.cuda.is_available())
print(opt.model)

train_dataloader,valid_dataloader,test_dataloader = get_dataloader(opt)

for ii,batch in tqdm(enumerate(train_dataloader)):
    print(batch)
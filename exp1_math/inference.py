import torch
import torch.nn as nn
from pathlib import Path
import sys
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from exp1_math.model import Classifier_Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cpu': print('using cpu')


def load_hs(hs_path:str|Path):
    obj = torch.load(hs_path, weight_only=True, map_location='cpu', mmap=True)
    assert isinstance(obj, dict)
    hs = {int(k):v for k,v in obj.items()} # layer_idx: (N, hidden_dim)
    return hs

def reload_model(model_path: str | Path):
    model = Classifier_Linear(input_dim=768).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def inference(hs:torch.Tensor, model:Classifier_Linear):
    model.eval()
    with torch.inference_mode():
        hs = hs.to(device)
        logits = model(hs)
        probs = torch.sigmoid(logits)
    return probs

def set_args():
    p = argparse.ArgumentParser()
    p.add_argument('--hs_path', type=str, required=True)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--output_path', type=str, required=True)
    return p.parse_args()

def main():
    args = set_args()
    hs = load_hs(args.hs_path)
    model = reload_model(args.model_path)
    probs = inference(hs, model)
    print(probs)


if __name__ == '__main__':
    main()








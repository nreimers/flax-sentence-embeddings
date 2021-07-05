import torch
from transformers import AutoModel, AutoTokenizer
import sys
import os

tok = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
model = AutoModel.from_pretrained(sys.argv[1], from_flax=True)
model.save_pretrained(os.path.join(sys.argv[1], "pt"))
tok.save_pretrained(os.path.join(sys.argv[1], "pt"))
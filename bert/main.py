
from options import options

options = options()
opts = options.parse()

import data
import model

print("creating Bert model")
bert = model.bertModel(opts)

print("creating dataloader")
dataset = data.dataloader(bert, opts)

#print(dataset.train_input[0])

print("creating trainer")
trainer = model.bertTrainer(dataset, bert, opts)
trainer.train_model()
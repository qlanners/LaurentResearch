# Import all necessary packages

import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import prepare_data_quinn
import model_quinn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
print('cudnn benchmark')

"""~~~~~~~~~~~~~~~~~~
Set the parameters
~~~~~~~~~~~~~~~~~~"""

batch_size = 20
eval_batch_size = 20
dim_emb = 765
dim_hid = 765
t0 = 35
seed = 33
data_folder = './wikitext-2' #check this
h0_val = 0
c0_val = 0
max_epoch = 100
lr0 = 4
decay_trigger = 0.99
decay = 1.1

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

"""~~~~~~~~~~~~~~~~~
Load data
~~~~~~~~~~~~~~~~~~"""

print('preparing the data...')

corpus = prepare_data_quinn.Corpus(data_folder)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.cuda()
    return data

train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, batch_size)
test_data = batchify(corpus.test, batch_size)

print('Done')

"""~~~~~~~~~~~~~~
Build the model
~~~~~~~~~~~~~~"""

num_tokens = len(corpus.dictionary)

network = model_quinn.RNNModel(num_tokens, dim_emb, dim_hid, dropout=0, tie_weights=False)
network.cuda()

crit = nn.CrossEntropyLoss()

"""~~~~~~~~~~~~~
Training Code
~~~~~~~~~~~~~"""

def get_step_size_per_epoch(source):
	if source.size(0)%t0 == 0:
		return source.size(0)//t0-1
	else:
		return source.size(0)//t0

def normalize_gradient():
	grad_norm_squared = 0

	for p in network.parameters():
		grad_norm_squared += p.grad.data.norm()**2

	grad_norm = math.sqrt(grad_norm_squared)

	if grad_norm < 1e-4:
		net.zero_grad()
		print('grad norm close to zero. gradients of all model parameters set to 0')
	else:
		for p in network.parameters():
			p.grad.data.div_(grad_norm)

	return grad_norm


def forward_pass(data, h_0, c_0):

	loss = torch.Tensor(1).cuda()
	loss[0] = 0

	loss = Variable(loss, requires_grad=True)
	h = Variable(h_0, requires_grad=True)
	c = Variable(c_0, requires_grad=True)

	for t in range(t0):
		x = Variable(data[t], requires_grad=False)
		target = Variable(data[t+1], requires_grad=False)
		scores, h, c = network(x, h, c) #why not network.forward()
		this_loss = crit(scores, target)
		loss = loss + this_loss

	h_final = h.data
	c_final = c.data
	loss_final = loss.data[0]/t0

	return loss, loss_final, h_final, c_final

def evaluate(source):

	network.eval()

	h_0 = torch.Tensor(batch_size, dim_hid).cuda().fill_(h0_val)
	c_0 = torch.Tensor(batch_size, dim_hid).cuda().fill_(c0_val)
	total_loss = 0
	position = 0
	step_per_epoch = get_step_size_per_epoch(source)

	for step in range(step_per_epoch):
		data = source[position:position+t0+1]
		loss, loss_final, h_new, c_new = forward_pass(data, h_0, c_0)
		h_0.copy_(h_new)
		c_0.copy_(c_new)
		total_loss += loss_final
		position = position + t0

	network.train()

	return total_loss / step_per_epoch

def train():

	network.train()
	h_0 = torch.Tensor(batch_size, dim_hid).cuda()
	c_0 = torch.Tensor(batch_size, dim_hid).cuda()

	learning_rate = lr0

	step_per_epoch = get_step_size_per_epoch(train_data)
	perp_val = 1e6

	for epoch in range(max_epoch):

		start_time = time.time()

		h_0.fill_(h0_val)
		c_0.fill_(c0_val)
		total_loss = 0
		position = 0

		for step in range(step_per_epoch):

			network.zero_grad()
			data = train_data[position:position+t0+1]
			loss, loss_final, h_new, c_new = forward_pass(data, h_0, c_0)
			loss.backward()

			grad_norm = normalize_gradient()
			for p in network.parameters():
				p.data.add_(-learning_rate, p.grad.data)

			h_0.copy_(h_new)
			c_0.copy_(c_new)
			total_loss += loss_final
			position = position + t0

		loss_train = total_loss / step_per_epoch
		loss_valid = evaluate(val_data)
		perp_val_old = perp_val
		perp_val = math.exp(loss_valid)
		perp_train = math.exp(loss_train)

		elapsed_time = time.time() - start_time

		print('epoch='+str(epoch)+'\n'+'time elapsed='+str(elapsed_time)+'\n'+
			'learning rate='+str(learning_rate)+'\n'+'gradnorm='+str(grad_norm)+'\n'+
			'loss train/validation= '+str(perp_train)+'/'+str(perp_val))

		if perp_val > perp_val_old*decay_trigger:
			learning_rate = learning_rate / decay

train()




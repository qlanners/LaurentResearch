
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LTSMcell(nn.Module):
	
	def __init__(self, input_size, hidden_size):
		super(LTSMcell, self).__init__()
		
		self.data_size = input_size
		self.hidden_size = hidden_size

		self.dropgate = nn.Dropout(0.45)
		self.dropmain = nn.Dropout(0.65)

		#Forget gate
		self.i2h_f = nn.Linear(input_size, hidden_size, bias=False)
		self.old_h2h_f = nn.Linear(hidden_size, hidden_size, bias=False)
		self.b_f = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)

		#Input gate
		self.i2h_i = nn.Linear(input_size, hidden_size, bias=False)
		self.old_h2h_i = nn.Linear(input_size, hidden_size, bias=False)
		self.b_i = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)

		#Ouput gate
		self.i2h_o = nn.Linear(input_size, hidden_size, bias=False)
		self.old_h2h_o = nn.Linear(input_size, hidden_size, bias=False)
		self.b_o = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)

		#Main gate
		self.i2h_m = nn.Linear(input_size, hidden_size, bias=False)
		self.old_h2h_m = nn.Linear(input_size, hidden_size, bias=False)
		self.b_m = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)

	def init_weights(self): 
		initrange = 0.07
		for p in self.parameters():
			p.data.uniform(-initrange, initrange)

		self.b_f.data.fill_(1)
		self.b_i.data.fill_(-1)

	def forward(self, x, old_state):
		h0, c0 = old_state

		x_dr = self.dropgate(x)
		h0_dr = self.dropgate(h0)
		x_dr_m = self.dropmain(x)
		h0_dr_m = self.dropmain(h0)

		main_gate = F.tanh(self.i2h_m(x_dr_m) + self.old_h2h_m(h0_dr_m) + self.b_m)
		forget_gate = F.sigmoid(self.i2h_f(x_dr) + self.old_h2h_f(h0_dr) + self.b_f)
		input_gate = F.sigmoid(self.i2h_i(x_dr) + self.old_h2h_i(h0_dr) + self.b_i)
		output_gate = F.sigmoid(self.i2h_o(x_dr) + self.old_h2h_o(h0_dr) + self.b_o)

		c = forget_gate*c0 + input_gate*main_gate

		h = output_gate * (F.tanh(c))

		return (h,c)

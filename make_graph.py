from graphviz import Digraph
from torch.autograd import Variable

def save(fname, creator):
	dot = Digraph(comment='LRP', node_attr={'style': 'filled', 'shape': 'box'})
	seen = set()

	def add_nodes(var):
		if var not in seen:
			if isinstance(var, Variable):
				dot.node(str(id(var)), str(var.size()), fillcolor='lightbule')
			else:
				dot.node(str(id(var)), type(var).__name__)
			seen.add(var)
			if hasattr(var, 'previous_functions'):
				for u in var.previous_functions:
					dot.edge(str(id(u[0])), str(id(var)))
					add_nodes(u[0])

	add_nodes(creator)
	dot.save(fname)
	
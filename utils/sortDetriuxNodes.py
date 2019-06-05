import json
import numpy as np
from args import Args

def sort_matrix(matrix=None, circular=False):
	pt_name = Args.root_directory + "/utils/parcellationTable_Ordered.json" # parcellation table to edit VisuOrder

	# Read parcellation table to edit VisuOrder
	with open(pt_name) as f:
		pt = json.load(f)
	f.close()
	order = np.argsort([pt[i]["VisuOrder"] for i in range(0, len(pt))])
	l = len(order)
	if not circular:
		order[int(l/2):l] = np.flip(order[int(l/2):l])

	if matrix is None:
		return order
	else:
		sorted_matrix = matrix[order, :]
		sorted_matrix = sorted_matrix[:, order]
		return sorted_matrix, order


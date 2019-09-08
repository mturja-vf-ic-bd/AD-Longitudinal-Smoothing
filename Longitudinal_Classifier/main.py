from Longitudinal_Classifier.read_file import *
from Longitudinal_Classifier.model import LongGAT, GATConvPool, LongGNN
from Longitudinal_Classifier.helper import convert_to_geom_all

if __name__ == '__main__':
    data = read_subject_data('005_S_5038', conv_to_tensor=False)
    G = convert_to_geom_all(data["node_feature"], data["adjacency_matrix"], data["dx_label"])
    print("Data read finished !!!")

    model = LongGNN(in_feat=[1, 3, 3], dropout=0.5, concat=False,
                    alpha=0.2, n_heads=3, n_layer=2, n_class=3, pooling_ratio=0.3)
    g = model(G)
    print(g.shape)
    print(g.data)

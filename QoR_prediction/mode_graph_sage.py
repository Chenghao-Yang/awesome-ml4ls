import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, degree

allowable_synthesis_features = {
    'synth_type': [0, 1, 2, 3, 4, 5, 6]
}


def get_synth_feature_dims():
    return list(map(len, [
        allowable_synthesis_features['synth_type']
    ]))


full_synthesis_feature_dims = get_synth_feature_dims()

allowable_features = {
    'node_type': [0, 1, 2],
    'num_inverted_predecessors': [0, 1, 2]
}


def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))


full_node_feature_dims = get_node_feature_dims()


class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()

        self.node_type_embedding = torch.nn.Embedding(full_node_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        # First feature is node type, second feature is inverted predecessor
        x_embedding = self.node_type_embedding(x[:, 0])
        x_embedding = torch.cat((x_embedding, x[:, 1].reshape(-1, 1)), dim=1)
        return x_embedding


class GCNConv(MessagePassing):
    def __init__(self, in_emb_dim, out_emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(in_emb_dim, out_emb_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.linear(x)
        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, node_encoder, num_layer, input_dim, emb_dim, gnn_type='gcn'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder

        ###List of GNNs
        """
        self.convs = torch.nn.ModuleList()

        self.batch_norms_gat = torch.nn.BatchNorm1d(emb_dim)

        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(input_dim, emb_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(1, num_layer):
            self.convs.append(GCNConv(emb_dim, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        """
        self.sage = SAGENet(input_dim, emb_dim)


    def forward(self, batched_data):

        # gate_type, node_type, edge_index = batched_data.gate_type, batched_data.node_type, batched_data.edge_index
        edge_index = batched_data.edge_index

        x = torch.cat([batched_data.node_type.reshape(-1, 1), batched_data.num_inverted_predecessors.reshape(-1, 1)],
                      dim=1)
        h = self.node_encoder(x)
        """
        for layer in range(self.num_layer):

            h = self.convs[layer](h, edge_index)
            h = self.batch_norms[layer](h)

            if layer != self.num_layer - 1:
                h = F.relu(h)
        """
        h = self.sage(h, edge_index)

        return h


class GNN(torch.nn.Module):

    def __init__(self, node_encoder, input_dim, num_layer=2, emb_dim=64, gnn_type='gcn', graph_pooling="mean"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling

        self.gnn_node = GNN_node(node_encoder, num_layer, input_dim, emb_dim, gnn_type=gnn_type)
        self.pool1 = global_mean_pool
        self.pool2 = global_max_pool

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph1 = self.pool1(h_node, batched_data.batch)
        h_graph2 = self.pool2(h_node, batched_data.batch)
        return torch.cat([h_graph1,h_graph2],dim=1)


class SynthFlowEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(SynthFlowEncoder, self).__init__()
        self.synth_emb = torch.nn.Embedding(full_synthesis_feature_dims[0], emb_dim)
        torch.nn.init.xavier_uniform_(self.synth_emb.weight.data)

    def forward(self, x):
        """
        x_embedding = self.synth_emb(x[:, 0])
        for i in range(1, x.shape[1]):
            x_embedding = torch.cat((x_embedding, self.synth_emb(x[:, i])), dim=1)
        """
        x_embedding = self.synth_emb(x)

        return x_embedding


class SynthConv(torch.nn.Module):
    def __init__(self, inp_channel=1, out_channel=3, ksize=6, stride_len=1):
        super(SynthConv, self).__init__()
        self.conv1d = torch.nn.Conv1d(inp_channel, out_channel, kernel_size=(ksize,), stride=(stride_len,))

    def forward(self, x):
        x = x.reshape(-1, 1, x.size(1))  # Convert [4,60] to [4,1,60]
        x = self.conv1d(x)
        return x.reshape(x.size(0), -1)  # Convert [4,3,55] to [4,165]


class SynthNet(torch.nn.Module):

    def __init__(self, node_encoder, synth_encoder, n_classes, synth_input_dim, node_input_dim, gnn_embed_dim=128,
                 num_fc_layer=4, hidden_dim=512, channels=4, dropout=0.2):
        super(SynthNet, self).__init__()
        self.num_layers = num_fc_layer
        self.hidden_dim = hidden_dim
        self.node_encoder = node_encoder
        self.synth_encoder = synth_encoder
        self.node_enc_outdim = node_input_dim
        self.synth_enc_outdim = synth_input_dim
        self.gnn_emb_dim = gnn_embed_dim
        self.n_classes = n_classes

        self.dropout = dropout
        self.input_pos_embedding = torch.nn.Embedding(20, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(full_node_feature_dims[0], embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=2,
            dropout=self.dropout,
            dim_feedforward=8 * channels,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.input_projection = Linear(3, channels)
        # self.output_projection = Linear(n_decoder_inputs, channels)

        self.linear = Linear(channels, 1)

        # Multiplier by 2 since each gate and node type has same encoding out dimension
        # self.gnn = GNN(self.node_encoder,self.node_enc_outdim*2)
        # Node encoding has dimension 3 and number of incoming inverted edges has dimension 1
        self.gnn = GNN(self.node_encoder, self.node_enc_outdim + 1)

        self.fcs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # GNN + (synthesis flow encoding + synthesis convolution)
        self.in_dim_to_fcs = int(self.gnn_emb_dim + 50)
        self.fcs.append(torch.nn.Linear(self.in_dim_to_fcs, self.hidden_dim))

        for layer in range(1, self.num_layers - 1):
            self.fcs.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))

        self.fcs.append(torch.nn.Linear(self.hidden_dim, self.n_classes))
        self.syn_linear = Linear(80, 50)

    def encode_src(self, src):
        # print('src_size---->', src.size())
        # src_start = self.input_projection(src).permute(1, 0, 2)
        src_start = src.permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start

        return src

    def forward(self, batch_data):

        graphEmbed = self.gnn(batch_data)
        synthFlow = batch_data.synVec
        h_syn = self.synth_encoder(synthFlow.reshape(-1, 20))
        src = h_syn
        synconv1_out = self.encode_src(src)
        synconv1_out = synconv1_out.permute(1, 0, 2)
        synconv1_out = synconv1_out.reshape(synconv1_out.size(0), -1)
        # print('synconv1_out---->', synconv1_out.size())
        synconv1_out = self.syn_linear(synconv1_out)

        concatenatedInput = torch.cat([graphEmbed, synconv1_out], dim=1)
        # concatenatedInput = torch.cat([graphEmbed, synconv1_out, synconv2_out, synconv3_out], dim=1)
        x = F.relu(self.fcs[0](concatenatedInput))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in range(1, self.num_layers - 1):
            x = F.relu(self.fcs[layer](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x
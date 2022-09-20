import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv, SAGEConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.utils import add_self_loops, degree

allowable_synthesis_features = {
    'synth_type': [0, 1, 2, 3, 4, 5, 6, 7]
}


def get_synth_feature_dims():
    return list(map(len, [
        allowable_synthesis_features['synth_type']
    ]))


full_synthesis_feature_dims = get_synth_feature_dims()

allowable_features = {
    'node_type': [0, 1, 2],
    'num_inverted_predecessors': [0, 1, 2],
    'edge_type': [0, 1]
}


def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))


def get_edge_feature_dims():
    return list(map(len, [
        allowable_features['edge_type']
    ]))


full_node_feature_dims = get_node_feature_dims()
full_edge_feature_dims = get_edge_feature_dims()


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


# class SAGE(torch.nn.Module):
#     def __init__(self, in_emb_dim, out_emb_dim):
#         super(SAGE, self).__init__()
#         self.conv1 = SAGEConv(in_emb_dim, out_emb_dim)
#         self.conv2 = SAGEConv(out_emb_dim, out_emb_dim)
#
#         self.batch_norm1 = torch.nn.BatchNorm1d(out_emb_dim)
#         self.batch_norm2 = torch.nn.BatchNorm1d(out_emb_dim)
#
#     def forward(self, h, edge_index, batch):
#
#         h = F.relu(self.batch_norm1(self.conv1(h, edge_index)))
#         #h = F.dropout(h, p=0.6, training=self.training)
#         # h = F.relu(self.batch_norm2(self.conv2(h, edge_index)))
#         h = self.batch_norm2(self.conv2(h, edge_index))
#
#         xF = torch.cat([global_max_pool(h, batch), global_mean_pool(h, batch)], dim=1)
#         return xF


class GINEEncoder(nn.Module):
    def __init__(self, in_channels, dim,  dropout):
        super(GINEEncoder, self).__init__()

        self.conv1 = GINEConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINEConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINEConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINEConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINEConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, 64)))


        # self.conv1 = GINEConv(
        #     Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv2 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv3 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv4 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv5 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv6 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv7 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv8 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv9 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()))
        #
        # self.conv10 = GINEConv(
        #     Sequential(Linear(dim, 64), BatchNorm1d(64)))


        # self.conv1 = GINEConv(
        #     Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv2 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv3 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv4 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv5 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv6 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv7 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv8 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv9 = GINEConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU()), edge_dim=2)
        #
        # self.conv10 = GINEConv(
        #     Sequential(Linear(dim, 64), BatchNorm1d(64)), edge_dim=2)

        self.dropout = nn.Dropout(dropout)
        self.act = F.relu
        self._reset_parameters()

        self.attr_encode = nn.Embedding(full_edge_feature_dims[0], 32)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data, x):
        edge_index, edge_attr, batch = data.edge_index, data.edge_type, data.batch
        #print('edge_attr---->', edge_attr)

        #x = self.x_encode(x)
        edge_attr = self.attr_encode(edge_attr.long()).squeeze()
        # print('x--->', x[0].size(-1))
        # print('edge_attr---->',edge_attr.size(-1))


        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.conv4(x, edge_index, edge_attr)
        x = self.conv5(x, edge_index, edge_attr)
        # x = self.conv6(x, edge_index, edge_attr)
        # x = self.conv7(x, edge_index, edge_attr)
        # x = self.conv8(x, edge_index, edge_attr)
        # x = self.conv9(x, edge_index, edge_attr)
        # x = self.conv10(x, edge_index, edge_attr)

        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        return x


class GNN(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, node_encoder, input_dim, emb_dim=32, gnn_type='gcn'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(GNN, self).__init__()
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder
        node_encoder = node_encoder

        self.conv_net = GINEEncoder(input_dim, emb_dim, dropout=0.1)
        #self.conv2 = GCNConv(emb_dim, emb_dim)
        # self.conv3 = GCNConv(emb_dim, emb_dim)

        #self.batch_norm1 = torch.nn.BatchNorm1d(emb_dim)
        #self.batch_norm2 = torch.nn.BatchNorm1d(emb_dim)
        # self.batch_norm3 = torch.nn.BatchNorm1d(emb_dim)

    def forward(self, batched_data):
        # edge_index, batch, edge_attr = batched_data.edge_index, batched_data.batch, batched_data.edge_type
        x = torch.cat([batched_data.node_type.reshape(-1, 1), batched_data.num_inverted_predecessors.reshape(-1, 1)],
                      dim=1)
        x = self.node_encoder(x)
        #xF = self.conv_net(h, edge_index, edge_attr, batch)
        xF = self.conv_net(batched_data, x)
        return xF


class SynthFlowEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(SynthFlowEncoder, self).__init__()
        self.synth_emb = torch.nn.Embedding(full_synthesis_feature_dims[0], emb_dim, padding_idx=0)
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

    def encode_src(self, src,mask):
        #print('src_size---->', src.size())
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
        #print('src---->',src)
        #print('pos_encoder---->',pos_encoder)

        src = self.encoder(src,src_key_padding_mask=mask) + src_start

        return src

    def forward(self, batch_data):

        graphEmbed = self.gnn(batch_data)
        synthFlow = batch_data.synVec
        flow = synthFlow.reshape(-1, 20)
        #print(flow.size())
        mask = (flow == 0).reshape(flow.shape[0], flow.shape[1])
        #print('mask---->',mask)
        h_syn = self.synth_encoder(flow)
        src = h_syn
        #print('src---->', src.size())
        synconv1_out = self.encode_src(src,mask)
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
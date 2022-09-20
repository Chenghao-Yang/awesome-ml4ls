import argparse
import dgl
import subprocess
import torch
import os
import itertools
import sys
import re
from dgl.data.utils import save_graphs

def _int_to_binary_tensor(n):
    return torch.tensor([int(d) for d in ('{0:' + str(args.v_length)  + 'b}').format(n).replace(' ', '0')])

def _torch_apply(func, M):
    tList = [func(m) for m in torch.unbind(M, dim=0) ]
    res = torch.stack(tList, dim=0)

    return res

def _gcn_message(edges):
    signal = edges.src['signal']
    inverted_edges = edges.data['inv'].eq(1).nonzero().view(-1)
    signal[inverted_edges] = torch.bitwise_not(signal[inverted_edges])
    return {
        'x': edges.src['signal']
    }

def _gcn_reduce(nodes):
    signal = nodes.mailbox['x'].narrow(1, 0, 1) & nodes.mailbox['x'].narrow(1, 1, 1)
    signal = signal.view(-1, 1)
    return {
        'signal': signal
    }


def _run_simulation(g, levels, v_length=64):
    input_nodes = g.ndata['type'].eq(0).nonzero().view(-1)
    and_nodes = g.ndata['type'].eq(1).nonzero().view(-1)
    output_nodes = g.ndata['type'].eq(2).nonzero().view(-1)
    gate_nodes = torch.cat((and_nodes, output_nodes)).sort().values
    print('run_simulation了吗？')
    
    # initialize signal
    if v_length == 64:
        v_length = 62   # to avoid overflow
    g.ndata['signal'] = torch.full((g.number_of_nodes(), ), -1, dtype=torch.int64).view(-1, 1)
    simulation_value = torch.randint(0, 2**v_length, (len(input_nodes),)).view(-1, 1)
    g.nodes[input_nodes].data['signal'] = simulation_value
    
    # run simulation
    for i in range(levels):
        g.send(g.edges(), _gcn_message)
        g.recv(gate_nodes, _gcn_reduce, inplace=True)

    return g


def _verilog_to_aig(verilog_file):
    aig_file = verilog_file + '.aig'
    try:
        abc_command = 'read ' + verilog_file
        abc_command += '; strash'
        abc_command += '; write_aiger ' + aig_file
        proc = subprocess.check_output(['yosys-abc', '-c', abc_command])
        return aig_file
    except Exception as e:
        print(e)
        return None

def _aig_to_aag(aig_file):
    aag_file = aig_file + '.aag'
    try:
        yosys_command = 'read_aiger ' + aig_file 
        yosys_command += '; write_aiger -ascii ' + aag_file
        proc = subprocess.check_output(['yosys', '-QT', '-p', yosys_command])
        return aag_file
    except Exception as e:
        print(e)
        return None

def _aag_to_graph(aag_file, levels=0, v_length=64, run_simulation=False):
    inputs = []
    latches = []
    outputs = []
    ands = {}
    with open(aag_file, 'r') as f:
        header = f.readline().split()
        M, I, L, O, A = list(map(lambda literal: int(literal), header[1:]))
        
        # print('M\tI\tL\tO\tA')
        # print(str(M) + '\t' + str(I) + '\t' + str(L) + '\t' + str(O) + '\t' + str(A))

        for i in range(I):
            i_node = f.readline().strip()
            inputs.append(int(i_node))
        
        for i in range(L):
            q, next_q = list(map(lambda n: int(n), f.readline().strip().split()))
            latches.append((q, next_q))

        for i in range(O):
            o_node = f.readline().strip()
            outputs.append(int(o_node))
        
        for i in range(A):
            output, in1, in2 = list(map(lambda n: int(n), f.readline().strip().split()))
            ands[output] = (in1, in2)
        
    # Inputs and AND gates are even numbers.
    # Inverter gates are odd numbers.
    # We x2 so that every node can represent its inverter in the graph by a node
    # We +2 as DGL starts node indices from 0. So, we skip nodes 0 and 1
    # So, we end up having a graph where every node is either AND or INV
    # g_m represented a `mirrored` graph that we use to extract sub-graphs
    g = dgl.DGLGraph()
    number_of_nodes = 2 * (len(inputs) + len(ands)) + 2
    g.add_nodes(number_of_nodes)

    # Add directed edges 
    for node, (src1, src2) in ands.items():
        if src1 % 2 == 0:
            # source is an AND gate
            g.add_edges(src1, node, data={'inv': torch.tensor([0])})
        else:
            # source is an INV gate
            g.add_edges(src1 - 1, node, data={'inv': torch.tensor([1])})
        
        if src2 % 2 == 0:
            # source is an AND gate
            g.add_edges(src2, node, data={'inv': torch.tensor([0])})
        else:
            # source is an INV gate
            g.add_edges(src2 - 1, node, data={'inv': torch.tensor([1])})
    
    # label nodes: 0 -> input, 1 -> AND, 2 -> output
    g.nodes[inputs].data['type'] = torch.tensor([0]*len(inputs))
    g.nodes[list(ands.keys())].data['type'] = torch.tensor([1]*len(ands.keys()))
    g.nodes[outputs].data['type'] = torch.tensor([2]*len(outputs))

    # remove odd nodes and 0
    nodes_marked_for_delete = [0] + [n for n in g.nodes() if n.item() % 2 == 1]
    g.remove_nodes(nodes_marked_for_delete)

    # check if we are making inference only
    if not run_simulation:
        g.ndata.pop('type')
        g.edata.pop('inv')
        return g

    # to get node features that will serve as a similarity metric
    print('Running simulation for ' + aag_file + ' ..')
    g = _run_simulation(g, levels, v_length=v_length)
    print('有图了？')

    # convert signal to binary vector
    # PyTorch has no mapping function!!
    for node in g.nodes():
        g.nodes[node].data['v_sim'] = _torch_apply(_int_to_binary_tensor, \
            g.nodes[node].data['signal'][0]).view(1, -1)
    g.ndata.pop('signal')

    return g

def _get_number_of_levels(aig_file):
    abc_command = "read " + aig_file + "; print_stats"
    try:
        proc = subprocess.check_output(['yosys-abc', '-c', abc_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'i/o' in line:
                ob = re.search(r'lev *= *[0-9]+', line)
                levels = int(ob.group().split('=')[1].strip())
                return levels
    except Exception as e:
        print(e)
        return None

def aig_to_graph(aig_file, v_length=64, run_simulation=False):
    aag_file = _aig_to_aag(aig_file)
    levels = 0
    if run_simulation:
        # this is  used to generate a dataset with similarity vector
        levels = _get_number_of_levels(aig_file)
        if levels:
            g = _aag_to_graph(aag_file, levels, v_length=v_length, run_simulation=run_simulation)
            os.remove(aag_file)
            return g
    else:
        # this is used for inference
        g = _aag_to_graph(aag_file, levels, v_length=v_length, run_simulation=run_simulation)
        os.remove(aag_file)
        return g

def read_verilog(verilog_file, v_length=64, run_simulation=False):
    aig_file = _verilog_to_aig(verilog_file)
    if aig_file:
        g = aig_to_graph(aig_file, v_length=v_length, run_simulation=run_simulation)
        os.remove(aig_file)
        return g

def _save_g(file_path, g, labels=None):
    save_graphs(file_path, g, labels=labels)

def save_dataset(file_path, Gs, Ls):
    _save_g(file_path + '.bin', Gs)
    with open(file_path + '.labels', 'w') as f:
        f.write('\n'.join(Ls))

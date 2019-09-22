import os
import os.path as osp
import argparse

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

def ConvertGraph(modelpath, outdir, num_output, prefix, name):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    net_model = load_model(modelpath)
    
    K.set_learning_phase(0)
    pred = [None]*num_output
    pred_node_names = [None]*num_output
    for i in range(num_output):
        pred_node_names[i] = prefix+'_'+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)
    
    sess = K.get_session()
    
    f = 'graph_def_for_reference.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), outdir, f, as_text=True)
    print('Saved the graph definition in ascii format at: ', osp.join(outdir, f))
    
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)    
    graph_io.write_graph(constant_graph, outdir, name, as_text=False)
    print('Saved the constant graph (ready for inference) at: ', osp.join(outdir, name))

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m', dest='model', required=True, help='REQUIRED: The HDF5 Keras model you wish to convert to .pb')
    parser.add_argument('--numout','-n', type=int, dest='num_out', required=True, help='REQUIRED: The number of outputs in the model.')
    parser.add_argument('--outdir','-o', dest='outdir', required=False, default='./', help='The directory to place the output files - default("./")')
    parser.add_argument('--prefix','-p', dest='prefix', required=False, default='k2tfout', help='The prefix for the output aliasing - default("k2tfout")')
    parser.add_argument('--name', dest='name', required=False, default='output_graph.pb', help='The name of the resulting output graph - default("output_graph.pb")')
    args = parser.parse_args()

    ConvertGraph( args.model, args.outdir, args.num_out, args.prefix, args.name )
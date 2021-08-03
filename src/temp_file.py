# import tensorflow as tf
# print(tf.__version__)
# %tensorflow_version 1.15
import tensorflow as tf
print(tf.__version__)
# from tensorflow.contrib.lite.python import convert_saved_model
model_path = 'output_graph.pb'
with tf.io.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
  if node.op == 'RefSwitch':
    node.op = 'Switch'
    for index in xrange(len(node.input)):
      node.input[index] = node.input[index] + '/read'
  elif node.op == 'AssignSub':
    node.op = 'Sub'
    if 'use_locking' in node.attr: del node.attr['use_locking']

with tf.compat.v1.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='prefix') 

# for node in graph_def.node:
#     if node.op == 'RefSwitch':
#         node.op = 'Switch'
#         for index in xrange(len(node.input)):
#             if 'moving_' in node.input[index]:
#                 node.input[index] = node.input[index] + '/read'
#     elif node.op == 'AssignSub':
#         node.op = 'Sub'
#         if 'use_locking' in node.attr: del node.attr['use_locking']
# with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name="")
# tf.import_graph_def(graph_def, name='')
# tf.train.write_graph(graph_def, './', 'good_frozen.pb', as_text=False)
# tf.train.write_graph(graph_def, './', 'good_frozen.pbtxt', as_text=True)
    
# for op in graph.get_operations():
#     abc = graph.get_tensor_by_name(op.name + ":0")
#     print(abc)

# for n in graph_def.node[0].op:

# print(len(graph_def.node))
#     print(n.name)
#     print( tf.shape(n.op) )
    #  Placeholder (0,) is_train
    #  NoOp (0,)  init
    #  int32
    
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='output_graph.pb', 
    input_arrays= 'is_train',
    output_arrays='init',
    input_shapes={'is_train' : [0,]}
)
tflite_model = converter.convert()

tflite_model_size = open('model.tflite', 'wb').write(tflite_model)
print('TFLite Model is %d bytes' % tflite_model_size)
# tflite_model = tf.compat.v1.contrib.lite.toco_convert(graph_def, graph_def.node[0].op, graph_def.node[1618])
# convert_saved_model.convert(saved_model_dir='../src/',output_arrays="NoOp",output_tflite='../graph.tflite')
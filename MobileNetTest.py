import Models as M
import Global as G
import numpy as np
import tensorflow as tf
import sys
import time

# CREATE AND LOAD THE MODEL
model = M.make_mobilenet_model( G.IMG_SHAPE )
name = 'models/mobilenet_weights_%d' % (G.EPOCHS)
model.load_weights( name )

path = "Dataset/test/"
cmd = path + "data.txt"
f = open( cmd, 'r' )
for line in f :
    row = line.split()
    img = tf.io.read_file( path + row[ 0 ] )
    img = tf.image.decode_jpeg( img, channels = 3 )
    img = tf.image.convert_image_dtype( img, tf.float32 )
    img = ( img - 0.5 ) / 0.5
    img = tf.image.resize( img, [ G.IMG_SIZE, G.IMG_SIZE ] )
    img = tf.reshape( img, ( 1, G.IMG_SIZE, G.IMG_SIZE, 3 ) )
    start_time = time.time()
    a = model.predict( img )
    time_sec = ( time.time() - start_time )
    prediction = int( a[ 0 ][ 0 ] >= 0.5 )
    print( prediction, time_sec )

f.close()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 3 outputs: 0 (ground-level), 1 (upstairs), 2 (downstairs)
def make_mobilenet_model_3class( myshape ) :
    base_model = keras.applications.MobileNet(input_shape = myshape,include_top = False,weights = 'imagenet')
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense( 3, activation = 'softmax' )
    model = keras.Sequential([ base_model, global_average_layer,prediction_layer])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),                    
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

def make_mobilenet_fusion_3class( myshape ) :
    base_model = keras.applications.MobileNet(input_shape = myshape, include_top = False,weights = 'imagenet')
    img_input = keras.Input(shape= myshape, name="img_input")
    x = base_model( img_input )
    x = layers.GlobalAveragePooling2D()( x )
    imu_input = keras.Input( shape = (6), name="imu_input")
    x = tf.concat([x, imu_input], 1)
    prediction = layers.Dense(3, activation='softmax')( x )
    model = keras.Model(
        inputs=[img_input, imu_input],
        outputs=[ prediction ],
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def make_mobilenet_model( myshape ) :
    base_model = keras.applications.MobileNet(input_shape = myshape,include_top = False,weights = 'imagenet')
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense( 1, activation = 'sigmoid' )
    model = keras.Sequential([ base_model, global_average_layer,prediction_layer])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

def make_mobilenet_fusion( myshape ) :
    base_model = keras.applications.MobileNet(input_shape = myshape, include_top = False,weights = 'imagenet')
    img_input = keras.Input(shape= myshape, name="img_input")
    x = base_model( img_input )
    x = layers.GlobalAveragePooling2D()( x )
    imu_input = keras.Input( shape = (6), name="imu_input")
    x = tf.concat([x, imu_input], 1)
    prediction = layers.Dense(1, activation='sigmoid', name='prediction')( x )
    model = keras.Model(
        inputs=[img_input, imu_input],
        outputs=[ prediction ],
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

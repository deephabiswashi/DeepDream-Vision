import tensorflow as tf
import numpy as np
from PIL import Image
import math

#############################
# Pre-trained Model Loading #
#############################

def load_pretrained_model(layer_names):
    """
    Loads InceptionV3 (pre-trained on ImageNet) and returns a model that outputs only the selected layers.
    """
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    outputs = [base_model.get_layer(name).output for name in layer_names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return dream_model

#########################
# Image Pre-/De-processing #
#########################

def preprocess_image(image_path):
    """
    Loads an image from disk, converts it to a tensor and applies InceptionV3 preprocessing.
    Preserves the original resolution.
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img

def deprocess_image(img):
    """
    Converts a processed tensor back into a PIL image.
    """
    img = img[0]
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img) + 1e-8)
    img = tf.clip_by_value(img, 0, 1)
    img = tf.cast(img * 255, tf.uint8)
    return Image.fromarray(img.numpy())

#####################
# DeepDream Methods #
#####################

def calc_loss(img, model, layer_index):
    """
    Computes the mean activation of a specified layer as loss.
    """
    activations = model(img)
    if isinstance(activations, list):
        act = activations[layer_index]
    else:
        act = activations
    loss = tf.reduce_mean(act)
    return loss

def gradient_ascent_step(img, model, layer_index, step_size):
    """
    Performs a single gradient ascent step on the image.
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model, layer_index)
    gradients = tape.gradient(loss, img)
    gradients /= (tf.math.reduce_std(gradients) + 1e-8)
    img = img + gradients * step_size
    return img, loss

def tiled_gradient_ascent_step(img, model, layer_index, step_size, tile_size=512):
    """
    Performs gradient ascent on overlapping tiles to avoid border artifacts.
    """
    _, h, w, _ = img.shape
    shift_x = np.random.randint(0, tile_size)
    shift_y = np.random.randint(0, tile_size)
    img_shift = tf.roll(tf.roll(img, shift_x, axis=2), shift_y, axis=1)
    
    gradients = tf.zeros_like(img_shift)
    num_tiles_x = math.ceil(w / tile_size)
    num_tiles_y = math.ceil(h / tile_size)
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_size
            y_start = i * tile_size
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)
            
            tile = img_shift[:, y_start:y_end, x_start:x_end, :]
            with tf.GradientTape() as tape:
                tape.watch(tile)
                loss = calc_loss(tile, model, layer_index)
            grad = tape.gradient(loss, tile)
            grad /= (tf.math.reduce_std(grad) + 1e-8)
            pad_top = y_start
            pad_bottom = h - y_end
            pad_left = x_start
            pad_right = w - x_end
            grad_padded = tf.pad(grad, [[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]])
            gradients += grad_padded

    gradients = tf.roll(tf.roll(gradients, -shift_x, axis=2), -shift_y, axis=1)
    img = img + gradients * step_size
    return img, loss

def run_deep_dream_with_tiles(img, model, layer_index, iterations=20, step_size=0.01, tile_size=512, use_tiles=True):
    """
    Runs gradient ascent (optionally in tiled mode) for a number of iterations.
    """
    for i in range(iterations):
        if use_tiles:
            img, loss = tiled_gradient_ascent_step(img, model, layer_index, step_size, tile_size)
        else:
            img, loss = gradient_ascent_step(img, model, layer_index, step_size)
    return img

def deep_dream(image_path, layer_name, iterations=20, step_size=0.01, octave_scale=1.4, num_octaves=3, tile_size=512):
    """
    Generates a deep dream image for a given input image and layer.
    The process includes octave scaling and tiled gradient ascent.
    The final output is resized back to the original resolution.
    """
    img = preprocess_image(image_path)
    model = load_pretrained_model([layer_name])
    original_shape = tf.shape(img)[1:3]
    img = tf.identity(img)
    
    for octave in range(num_octaves):
        new_size = tf.cast(tf.cast(original_shape, tf.float32) * (octave_scale ** octave), tf.int32)
        img = tf.image.resize(img, new_size)
        img = run_deep_dream_with_tiles(img, model, 0, iterations=iterations, step_size=step_size, tile_size=tile_size, use_tiles=True)
    
    img = tf.image.resize(img, original_shape)
    result = deprocess_image(img)
    return result

###############################
# Visualization Chart Methods #
###############################

def generate_chart_data(deepdream_results):
    """
    For each deep dream output, compute the average pixel intensity as a proxy for activation.
    """
    chart_data = {}
    for layer, img in deepdream_results.items():
        img_array = np.array(img).astype(np.float32) / 255.0
        chart_data[layer] = np.mean(img_array)
    return chart_data

def generate_chart(chart_data, save_path='static/images/chart.png'):
    """
    Generates a bar chart of average activations using matplotlib.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    layers = list(chart_data.keys())
    values = [chart_data[layer] for layer in layers]
    plt.figure(figsize=(8,4))
    plt.bar(layers, values, color='skyblue')
    plt.xlabel('Layer')
    plt.ylabel('Average Activation')
    plt.title('Model Layer Activations')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

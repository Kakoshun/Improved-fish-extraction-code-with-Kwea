import matplotlib
import matplotlib.pyplot as plt
import io
import os
import pathlib
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Function to load an image into a numpy array
# Συνάρτηση για φόρτωση εικόνας σε numpy array
def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Function to apply a color scale, choosing a color, and works with elif
# Προσθήκη συνάρτησης για αλλαγή σε κλίμακα του μπλε, επιλέγω χρώμα και δουλεύει το elif
def apply_color_scale(image_np, scale='blue'):
    if scale == 'blue':
        image_np[:, :, 0] = 0  # Zero out green channel
        # Μηδενισμός πράσινου
        image_np[:, :, 1] = 0  # Zero out red channel
        # Μηδενισμός κόκκινου
    elif scale == 'green':
        image_np[:, :, 0] = 0  # Zero out blue channel
        # Μηδενισμός μπλε
        image_np[:, :, 2] = 0  # Zero out red channel
        # Μηδενισμός κόκκινου
    elif scale == 'red':
        image_np[:, :, 1] = 0  # Zero out green channel
        # Μηδενισμός πράσινου
        image_np[:, :, 2] = 0  # Zero out blue channel
        # Μηδενισμός μπλε
    return image_np

def get_keypoint_tuples(eval_config):
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                   for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict

def show_inference(model, image_path, color_scale='blue'):
    # Load the original image
    # Φόρτωση της αρχικής εικόνας
    pil_img = Image.open(image_path)
    original_image_np = np.array(pil_img)  # Save the original image for later use
    # Αποθήκευση της αρχικής εικόνας για μετέπειτα χρήση
    w, h = pil_img.size
    print("pil_img size:", w, h)

    # Apply the selected color scale
    # Εφαρμογή της επιλεγμένης κλίμακας χρωμάτων
    image_np = apply_color_scale(np.copy(original_image_np), color_scale)

    # Actual detection
    output_dict = run_inference_for_single_image(model, image_np)

    # NPET: Process detection results
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    score_thres = 0.1 # Changing detection accuracy threshold for fish detection
    # Αλλάζω το ποσοστό ακρίβειας για τον εντοπισμό ψαριών
    for i in range(boxes.shape[0]):
        if scores[i] > score_thres:
            ymin, xmin, ymax, xmax = boxes[i]
            iymin, ixmin, iymax, ixmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
            pwleft, phup, pwright, phdown = 0.05, 0.05, 0.2, 0.2
            bw, bh = ixmax - ixmin, iymax - iymin
            bw2left, bw2right, bh2up, bh2down = int(pwleft * bw), int(pwright * bw), int(phup * bh), int(phdown * bh)
            ixmin2, iymin2 = max(1, ixmin - bw2left), max(1, iymin - bh2up)
            ixmax2, iymax2 = min(w - 1, ixmax + bw2right), min(h - 1, iymax + bh2down)

            # Extract the new png image from the original color image
            # Εξαγωγή της νέας png εικόνας, από την αρχική έγχρωμη εικόνα
            subimg = original_image_np[iymin2:iymax2, ixmin2:ixmax2]
            head_tail = os.path.split(image_path)
            filename, ext = head_tail[1].split(".")
            subfilename = os.path.join(head_tail[0], f"{filename}_f{i}.png")
            Image.fromarray(subimg).save(subfilename)

    # Visualization of detection results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    plt.imshow(Image.fromarray(image_np))
    plt.savefig("mygraph.png")

# List of strings for labeling
# List of strings for labeling
PATH_TO_LABELS = 'fish_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load test images
# Load test images
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('_video')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

detection_model = tf.saved_model.load('/saved_model')

# Run inference on each image
# Εκτέλεση για κάθε εικόνα
for image_path in TEST_IMAGE_PATHS:
    show_inference(detection_model, image_path, color_scale='blue')  # Change the color scale here
    # Αλλάζουμε από εδώ την κλίμακα χρώματος

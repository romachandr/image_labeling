import numpy as np
import os
import sys
import tensorflow as tf
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class classTF1ObjectDetection:
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = r'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = r'labels/mscoco_label_map.pbtxt'
    NUM_CLASSES = 90
    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    IMAGE_SIZE = (12, 8)
    DETECTION_THRESHOLD = 0.50

    def model_load(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)


    def load_image_into_numpy_array(self, image):
        (img_width, img_height) = image.size
        return np.array(image.getdata()).reshape(
            (img_height, img_width, 3)).astype(np.uint8), img_height, img_width

    def run_detection_img_folder(self, img_save=False,
                      detected_img_postfix="detect",
                      print_log=False):
        res_arr = []
        TEST_IMAGE_PATHS = sorted([os.path.join(self.PATH_TO_TEST_IMAGES_DIR, img) for img in os.listdir(self.PATH_TO_TEST_IMAGES_DIR) if img.find(detected_img_postfix)<0])
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                for image_path in TEST_IMAGE_PATHS:
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np, img_height, img_width = self.load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    if print_log==True:
                        print(f'image_path: {image_path}')
                        print(f'boxes:\n{boxes}')
                        print(f'scores:\n{scores}')
                        print(f'classes:\n{classes}')
                        print(f'num:\n{num}')
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    plt.figure(figsize=self.IMAGE_SIZE)
                    if img_save==True:
                        image_path_dect = image_path.replace('.jpg', f'_{detected_img_postfix}.jpg')
                        plt.imsave(image_path_dect, image_np)
                    # plt.imshow(image_np)
                    res_arr.append({"image_path":image_path, "boxes":boxes, "scores":scores, "classes":classes, "num":num})
        res_arr_converted = self.filter_convert_boxes(res_arr=res_arr, img_height=img_height, img_width=img_width)
        return res_arr_converted

    def run_detection_img(self, use_saved_img=True, image_path=None,
                          img_np_array=None,
                          img_save=False,
                          visualize_boxes_and_labels=True,
                          detected_img_postfix="detect",
                          print_log=False):
        res_arr = []
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                if use_saved_img==True:
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np, img_height, img_width = self.load_image_into_numpy_array(image)
                else:
                    if img_np_array is not None:
                        image_np, img_height, img_width = img_np_array, img_np_array.shape[0], img_np_array.shape[1]
                    else:
                        return
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                if print_log==True:
                    print(f'image_path: {image_path}')
                    print(f'boxes:\n{boxes}')
                    print(f'scores:\n{scores}')
                    print(f'classes:\n{classes}')
                    print(f'num:\n{num}')
                # Visualization of the results of a detection.
                if visualize_boxes_and_labels==True:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                plt.figure(figsize=self.IMAGE_SIZE)
                if img_save==True:
                    if use_saved_img==True:
                        image_path_dect = image_path.replace('.jpg', f'_{detected_img_postfix}.jpg')
                    else:
                        image_path_dect = os.path.join(self.PATH_TO_TEST_IMAGES_DIR, f'ailbl_{str(time.time())}_{detected_img_postfix}.jpg')
                    plt.imsave(image_path_dect, image_np)
                # plt.imshow(image_np)
                res_arr.append({"image_path":image_path, "boxes":boxes, "scores":scores, "classes":classes, "num":num})
        res_arr_converted = self.filter_convert_boxes(res_arr=res_arr, img_height=img_height, img_width=img_width)
        return res_arr_converted

    def box_convert(self, box, img_height, img_width):
        ymin, xmin, ymax, xmax = box
        (top, left, bottom, right) = (ymin * img_height, xmin * img_width, ymax * img_height, xmax * img_width)
        return (int(top), int(left), int(bottom), int(right))

    def filter_convert_boxes(self, res_arr, img_height, img_width):
        res_conv_arr = []

        for res in res_arr:
            res_dict = {"image_path":res["image_path"]}
            scores_filtered = res["scores"][res["scores"]>=self.DETECTION_THRESHOLD]
            boxes_count = len(scores_filtered)
            res_dict["scores"] = scores_filtered
            res_dict["classes"] = [int(cls) for cls in res["classes"][0][:boxes_count]]
            res_dict["categories"] = [int(cls) for cls in res["classes"][0][:boxes_count]]
            res_dict["boxes"] = [self.box_convert(box, img_height, img_width) for box in res["boxes"][0][:boxes_count]]
            res_conv_arr.append(res_dict)

        return res_conv_arr

if __name__ == '__main__':
    OD = classTF1ObjectDetection()

    OD.model_load()
    res_arr = OD.run_detection_img_folder(img_save=True)
    print(f"res_arr:\n{res_arr}")
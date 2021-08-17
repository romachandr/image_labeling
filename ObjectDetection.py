import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import datetime
import time
import json
from Labeling import classLabeling as lbl

class classObjectDetection:

    min_contour_area = 1000
    motion_max_count = 3000
    intersection_threshold = 5
    rect_color_default = (255, 0, 0)
    contour_color = (0, 255, 0)
    rect_color_notif = (0, 0, 255)
    max_contour_area = 900
    diff_thrsh = 18
    #---
    folder_moving_objects_images = r'moving_objects_images'
    labeling_folder = r'images'
    img_param_dict = {'min_object_size': {'w':100, "h":60},
                      'max_inter_dimensions':(640, 360),
                      'resize_tuple':(640, 360),
                      'detection_range':{"x":0, "y":0, "w":640, "h":200},
                      'base_params': {"x": 0, "y": 0, "w": 630, "h": 200},
                      }
    # img_param_dict = {'min_object_size': {'w':600, "h":300},
    #                   'max_inter_dimensions':(2688, 1520),
    #                   'resize_tuple':(1280, 720),
    #                   'detection_range':{"x":0, "y":0, "w":2688, "h":700},
    #                   'base_params': {"x": 0, "y": 0, "w": 2588, "h": 1000},
    #                   }
    #---

    LBL = lbl()

    def rect_conv(self, rect_base_params, intersection_threshold=intersection_threshold, max_inter_dimensions=None):
        rect_check_params = ((rect_base_params['x'] - intersection_threshold if (rect_base_params['x'] - intersection_threshold)>0 else 0,
                              rect_base_params['y'] - intersection_threshold if (rect_base_params['y'] - intersection_threshold) > 0 else 0),
                             (rect_base_params['x'] + rect_base_params['w'] + intersection_threshold
                              if (rect_base_params['x'] + rect_base_params['w'] + intersection_threshold) < max_inter_dimensions[0]
                              else max_inter_dimensions[0],
                              rect_base_params['y'] + rect_base_params['h'] + intersection_threshold
                              if (rect_base_params['y'] + rect_base_params['h'] + intersection_threshold) < max_inter_dimensions[1]
                              else max_inter_dimensions[1]))
        return rect_check_params

    def check_rectangle_intersections(self, rect_base_params, rect_alt_params, max_inter_dimensions=None,
                                      intersection_threshold=intersection_threshold, print_log=False):
        """
        Checking of rectangle intersections.

        :param rect_base_params: dict {'contour_area': float, 'x': int, 'y': int, 'w': int, 'h': int}
        :param rect_alt_params: dict {'contour_area': float, 'x': int, 'y': int, 'w': int, 'h': int}
        :param max_inter_dimensions: tuple (w, h)
        :param thresholds: int - pixels arround rectangle
        :return: dict {"intersection":bool, "rect_params":dict {'x': int, 'y': int, 'w': int, 'h': int}}
        """

        intersection = False
        rect_params = {"x":rect_base_params['x'], "y":rect_base_params['y'],
                       "w":rect_base_params['w'], "h":rect_base_params['h']}
        rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
        if (rect_base_params['x'] != rect_alt_params['x']) | \
                (rect_base_params['y'] != rect_alt_params['y']) | \
                (rect_base_params['w'] != rect_alt_params['w']) | \
                (rect_base_params['h'] != rect_alt_params['h']):
            #--- up left
            if (rect_alt_params['x'] > rect_check_params[0][0]) and (rect_alt_params['x'] < rect_check_params[1][0]) and \
                    (rect_alt_params['y'] > rect_check_params[0][1]) and (rect_alt_params['y'] < rect_check_params[1][1]):
                intersection = True
                if ((rect_alt_params['x']+rect_alt_params['w']) > rect_check_params[1][0]):
                    rect_params['w'] = rect_alt_params['x']+rect_alt_params['w'] - rect_params['x']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
                if(rect_alt_params['y']+rect_alt_params['h']) > rect_check_params[1][1]:
                    rect_params['h'] = rect_alt_params['y']+rect_alt_params['h'] - rect_params['y']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
            #---
            #--- up right
            if ((rect_alt_params['x']+rect_alt_params['w'])  > rect_check_params[0][0]) and \
                    ((rect_alt_params['x']+rect_alt_params['w']) < rect_check_params[1][0]) and \
                    (rect_alt_params['y'] > rect_check_params[0][1]) and (rect_alt_params['y'] < rect_check_params[1][1]):
                intersection = True
                if (rect_alt_params['x'] < rect_check_params[0][0]):
                    rect_params['w'] = rect_params['w'] + (rect_params['x'] - rect_alt_params['x'])
                    rect_params['x'] = rect_alt_params['x']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
                if(rect_alt_params['y']+rect_alt_params['h']) > rect_check_params[1][1]:
                    rect_params['h'] = rect_alt_params['y']+rect_alt_params['h'] - rect_params['y']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
            #---
            #--- bottom left
            if (rect_alt_params['x'] > rect_check_params[0][0]) and (rect_alt_params['x'] < rect_check_params[1][0]) and \
                    ((rect_alt_params['y']+rect_alt_params['h']) > rect_check_params[0][1]) and \
                    ((rect_alt_params['y']+rect_alt_params['h']) < rect_check_params[1][1]):
                intersection = True
                if ((rect_alt_params['x'] + rect_alt_params['w']) > rect_check_params[1][0]):
                    rect_params['w'] = rect_alt_params['x'] + rect_alt_params['w'] - rect_params['x']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
                if rect_alt_params['y'] < rect_check_params[0][1]:
                    rect_params['h'] = rect_params['h'] + (rect_params['y'] - rect_alt_params['y'])
                    rect_params['y'] = rect_alt_params['y']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
            #---
            #--- bottom right
            if ((rect_alt_params['x']+rect_alt_params['w']) > rect_check_params[0][0]) and \
                    ((rect_alt_params['x']+rect_alt_params['w']) < rect_check_params[1][0]) and \
                    (rect_alt_params['y'] > rect_check_params[0][1]) and \
                    (rect_alt_params['y'] < rect_check_params[1][1]):
                intersection = True
                if rect_alt_params['x'] < rect_check_params[0][0]:
                    rect_params['w'] = rect_params['w'] + (rect_params['x'] - rect_alt_params['x'])
                    rect_params['x'] = rect_alt_params['x']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
                if rect_alt_params['y'] < rect_check_params[0][1]:
                    rect_params['h'] = rect_params['h'] + (rect_params['y'] - rect_alt_params['y'])
                    rect_params['y'] = rect_alt_params['y']
                    rect_check_params = self.rect_conv(rect_params, intersection_threshold, max_inter_dimensions)
            #---
        if print_log:
            print(f"rect_base_params: {rect_base_params}")
            print(f"rect_alt_params: {rect_alt_params}")
            print(f"rect_check_params: {rect_check_params}")

        res_dict = {'intersection':intersection, 'rect_params':rect_params}
        return res_dict

    def rect_arr_filter(self, rect_arr, max_inter_dimensions=None, intersection_threshold=intersection_threshold,
                        print_log=False):
        rect_arr_copy = [{'x':elem['x'], 'y':elem['y'], 'w':elem['w'], 'h':elem['h']} for elem in rect_arr]
        filtered_rect_arr = []

        while len(rect_arr_copy)>0:
            rect_to_check = copy.deepcopy(rect_arr_copy[0])
            if len(rect_arr_copy)>1:
                intersection = False
                for rect in rect_arr_copy[1:]:
                    inter_res = self.check_rectangle_intersections(rect_to_check, rect,
                                                                   max_inter_dimensions=max_inter_dimensions,
                                                                   intersection_threshold=intersection_threshold,
                                                                   print_log=print_log)
                    if inter_res['intersection']==True:
                        intersection = True
                        rect_arr_copy.remove(rect)
                        rect_arr_copy.remove(rect_to_check)
                        rect_arr_copy.append(inter_res['rect_params'])
                        break
                    else:
                        inter_res = self.check_rectangle_intersections(rect, rect_to_check,
                                                                       max_inter_dimensions=max_inter_dimensions,
                                                                       intersection_threshold=intersection_threshold,
                                                                       print_log=print_log)
                        if inter_res['intersection'] == True:
                            intersection = True
                            rect_arr_copy.remove(rect)
                            rect_arr_copy.remove(rect_to_check)
                            rect_arr_copy.append(inter_res['rect_params'])
                            break

                if intersection==False:
                    rect_arr_copy.remove(rect_to_check)
                    filtered_rect_arr.append(rect_to_check)
            elif len(rect_arr_copy)==1:
                filtered_rect_arr.append(rect_to_check)
                rect_arr_copy.remove(rect_to_check)

        if print_log:
            print(f"len(rect_arr)= {len(rect_arr)}")
            print(f"rect_arr:\n{rect_arr}")
            print(f"len(filtered_rect_arr)= {len(filtered_rect_arr)}")
            print(f"filtered_rect_arr:\n{filtered_rect_arr}")
        # --- debug end
        return filtered_rect_arr

    def crop_resize_save(self, image_array,
                  crop=False, crop_range=None,
                  resize=False, resize_tuple=None,
                  BGRtoRBG=True,
                  file_save=True, file_name='tmp_image.jpg'):
        res = {'file_name':file_name}
        cropping_success = False
        crop_res_img = image_array
        if (crop==True) and (crop_range is not None):
            try:
                crop_res_img = image_array[crop_range['y']:crop_range['y'] + crop_range['h'],
                           crop_range['x']:crop_range['x'] + crop_range['w']]
            except Exception as e:
                print(f"image '{file_name}' cropping error: {e}")

        if (resize==True) and (resize_tuple is not None):
            try:
                crop_res_img = cv2.resize(crop_res_img, resize_tuple)
            except Exception as e:
                print(f"image '{file_name}' resizing error: {e}")

        if BGRtoRBG==True: crop_res_img = cv2.cvtColor(crop_res_img, cv2.COLOR_BGR2RGB)
        res['crop_res_img'] = crop_res_img
        if file_save==True:
            plt.imsave(file_name, crop_res_img)
            cropping_success = True

        res['cropping_success'] = cropping_success
        return res

    def save_pic_with_label(self, image_array,
                  pic_width, pic_height, pic_depth, label_name,
                  bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax,
                  BGRtoRBG=True,
                  file_save=True,
                  folder='',
                  filename='tmp_image.jpg',
                  path='tmp_image.jpg', ):
        res = {'path':path}
        crop_res_img = image_array

        if BGRtoRBG==True: crop_res_img = cv2.cvtColor(crop_res_img, cv2.COLOR_BGR2RGB)
        res['crop_res_img'] = crop_res_img
        if file_save==True:
            plt.imsave(path, crop_res_img)
        self.LBL.label_file_save(folder, filename, path, pic_width, pic_height, pic_depth, label_name,
                        bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax)
        return res

    def calc_pic_diff(self, frame1, frame2, conv_to_gray=True):
        if conv_to_gray == True:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(frame1, frame2)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, self.diff_thrsh, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        rect_arr = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            x2 = x + w
            y2 = y + h
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            else:
                if (w >= self.img_param_dict['min_object_size']['w']) \
                    and (h >= self.img_param_dict['min_object_size']['h']):
                    dr_x1 = self.img_param_dict['detection_range']['x']
                    dr_y1 = self.img_param_dict['detection_range']['y']
                    dr_x2 = self.img_param_dict['detection_range']['x'] + self.img_param_dict['detection_range']['w']
                    dr_y2 = self.img_param_dict['detection_range']['y'] + self.img_param_dict['detection_range']['h']
                    if (x >= dr_x1) and (x <= dr_x2) and (y >= dr_y1) and (y <= dr_y2) \
                            and (x2 >= dr_x1) and (x2 <= dr_x2) and (y2 >= dr_y1) and (y2 <= dr_y2):
                        rect_arr.append({"contour_area": cv2.contourArea(contour), "x": x, "y": y, "w": w, "h": h,
                                         # "contour":contour
                                         })

        filtered_rect_arr = self.rect_arr_filter(rect_arr=rect_arr, max_inter_dimensions=self.img_param_dict['max_inter_dimensions'])
        res_dict = {"frame1": frame1, "frame2": frame2, "contours": contours, "filtered_rect_arr": filtered_rect_arr}
        return res_dict

    def motion_detection(self, camera_stream=False, vcap_url=None,
                         stream_from_file=False, video_file=None,
                         movie_show=False, resize_tuple=None, draw_contours=False,
                         save_moving_objects=False,
                         motion_max_count=motion_max_count):
        cap = None
        if stream_from_file==True:
            cap = cv2.VideoCapture(video_file)
            if (cap.isOpened() == False):
                print("Error opening video stream or file")

        if camera_stream==True:
            cap = cv2.VideoCapture(vcap_url)


        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        while_flag = True if stream_from_file == False else (cap.isOpened())

        while while_flag:
            pic_diff = self.calc_pic_diff(frame1, frame2)
            contours = pic_diff["contours"]

            len_contours = len(contours)
            if len_contours<=motion_max_count:
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) < self.max_contour_area:
                        continue
                    if draw_contours==True:
                        cv2.drawContours(frame1, contours, -1, self.contour_color, 2)

            if len(contours)<=motion_max_count:
                filtered_rect_arr = pic_diff["filtered_rect_arr"]
                # contours = [x['contour'] for x in filtered_rect_arr]
                for rect in filtered_rect_arr:
                    x = rect["x"]
                    y = rect["y"]
                    w = rect["w"]
                    h = rect["h"]
                    intersection = self.check_rectangle_intersections(rect_base_params=self.img_param_dict['base_params'],
                                                                      rect_alt_params=rect,
                                      max_inter_dimensions=self.img_param_dict['max_inter_dimensions'])['intersection']
                    if intersection==True:
                        if save_moving_objects == True:
                            now = datetime.datetime.now()
                            file_name = os.path.join(os.path.join(self.folder_moving_objects_images,
                                                                  "mo_" + now.strftime("%Y%m%d%H%M%S%f")+'_'+str(x)+"x"+str(y)+'.jpg'))
                            crop_res = self.crop_resize_save(image_array=frame1,
                                                             crop=True,
                                                             crop_range=rect,
                                                             file_name=file_name)

                        if movie_show==True:
                            cv2.rectangle(frame1, (x, y), (x + w, y + h), self.rect_color_notif, 2)
                    else:
                        if movie_show == True:
                            cv2.rectangle(frame1, (x, y), (x + w, y + h), self.rect_color_default, 2)
                            # cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            #            1, (255, 0, 0), 3)

            if movie_show == True:
                frame_to_show = cv2.resize(frame1, resize_tuple)  # Downscale to improve frame rate
                cv2.imshow("Video", frame_to_show)

                ch = cv2.waitKey(5)
                if ch == 27:
                    break

            if camera_stream == True:
                cap = cv2.VideoCapture(vcap_url)
                ret, frame1 = cap.read()
                ret, frame2 = cap.read()
                print(f"{datetime.datetime.now()}: got camera frames")
                cap.release()
            else:
                frame1 = frame2
                ret, frame2 = cap.read()

        if camera_stream==True:
            cap.release()
            cv2.destroyAllWindows()

    def motion_detection_and_labeling(self, camera_stream=False, vcap_url=None,
                         stream_from_file=False, video_file=None,
                         movie_show=False, resize_tuple=None, draw_contours=False,
                         save_labels=True,
                         label_name='unknown',
                         motion_max_count=motion_max_count):
        cap = None
        if stream_from_file==True:
            cap = cv2.VideoCapture(video_file)
            if (cap.isOpened() == False):
                print("Error opening video stream or file")

        if camera_stream==True:
            cap = cv2.VideoCapture(vcap_url)


        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        while_flag = True if stream_from_file == False else (cap.isOpened())

        while while_flag:
            pic_diff = self.calc_pic_diff(frame1, frame2)
            contours = pic_diff["contours"]

            len_contours = len(contours)
            if len_contours<=motion_max_count:
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) < self.max_contour_area:
                        continue
                    if draw_contours==True:
                        cv2.drawContours(frame1, contours, -1, self.contour_color, 2)

            if len(contours)<=motion_max_count:
                filtered_rect_arr = pic_diff["filtered_rect_arr"]
                # contours = [x['contour'] for x in filtered_rect_arr]
                for rect in filtered_rect_arr:
                    x = rect["x"]
                    y = rect["y"]
                    w = rect["w"]
                    h = rect["h"]
                    intersection = self.check_rectangle_intersections(rect_base_params=self.img_param_dict['base_params'],
                                                                      rect_alt_params=rect,
                                      max_inter_dimensions=self.img_param_dict['max_inter_dimensions'])['intersection']
                    if intersection==True:
                        if save_labels == True:
                            now = datetime.datetime.now()
                            file_name = "ailbl_" + now.strftime("%Y%m%d%H%M%S%f")+'_'+str(x)+"x"+str(y)+'.jpg'
                            folder = self.labeling_folder
                            path = os.path.join(os.path.join(folder, file_name))
                            sres = self.save_pic_with_label(image_array=frame1,
                                                pic_width=frame1.shape[0],
                                                pic_height=frame1.shape[1],
                                                pic_depth=3,
                                                label_name=label_name,
                                                bndbox_xmin=x, bndbox_ymin=y, bndbox_xmax=x+w, bndbox_ymax=y+h,
                                                BGRtoRBG=True,
                                                file_save=True,
                                                folder=folder,
                                                filename=file_name,
                                                path=path, )


                        # if movie_show==True:
                        #     cv2.rectangle(frame1, (x, y), (x + w, y + h), self.rect_color_notif, 2)
                    else:
                        if movie_show == True:
                            cv2.rectangle(frame1, (x, y), (x + w, y + h), self.rect_color_default, 2)
                            # cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            #            1, (255, 0, 0), 3)

            if movie_show == True:
                frame_to_show = cv2.resize(frame1, resize_tuple)  # Downscale to improve frame rate
                cv2.imshow("Video", frame_to_show)

                ch = cv2.waitKey(5)
                if ch == 27:
                    break

            if camera_stream == True:
                cap = cv2.VideoCapture(vcap_url)
                ret, frame1 = cap.read()
                ret, frame2 = cap.read()
                print(f"{datetime.datetime.now()}: got camera frames")
                cap.release()
            else:
                frame1 = frame2
                ret, frame2 = cap.read()

        if camera_stream==True:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    OD = classObjectDetection()

    # #---
    # video_file = r'videos/192.168.1.105_01_20210807092201_20210807092221.mp4'
    # video_file = r'videos/192.168.1.105_01_20210806184710_20210806184727_small.mp4'
    video_file = r'videos/192.168.1.105_01_20210801173120_20210801173139_small.mp4'
    # video_file = r'videos/192.168.1.105_01_20210807092201_20210807092221_small.mp4'

    # OD.motion_detection(camera_stream=False, vcap_url='',
    #                     stream_from_file=True, video_file=video_file, resize_tuple=(1280, 720),
    #                     movie_show=True, draw_contours=False, motion_max_count=OD.motion_max_count,
    #                     save_moving_objects=True)

    OD.motion_detection_and_labeling(camera_stream=False, vcap_url='',
                        stream_from_file=True, video_file=video_file, resize_tuple=(1280, 720),
                        movie_show=True, draw_contours=False, motion_max_count=OD.motion_max_count,
                        save_labels=True,
                        label_name='car'
                        )

    #---

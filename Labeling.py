import xmltodict
from os import listdir
from os.path import isfile, join
import os
import glob
import copy
import pandas as pd
import xml.etree.ElementTree as ET

class classLabeling:
    sample_dict_base = {
                        "annotation":{
                            "folder":"train",
                            "filename":"photo_2021-08-01_12-59-28.jpg",
                            "path":"/mnt/sdb4/romachandr/PycharmProjects/tensorflow1/models/research/object_detection/images/train/photo_2021-08-01_12-59-28.jpg",
                            "source": {
                                "database":"Unknown"
                                },
                            "size":{
                                "width":1280,
                                "height":720,
                                "depth":3
                            },
                            "segmented":0,
                            "object":{
                                "name":"unknown_class",
                                "pose":"Unspecified",
                                "truncated":0,
                                "difficult":0,
                                "bndbox":{
                                    "xmin":111,
                                    "ymin":222,
                                    "xmax":333,
                                    "ymax":444
                                }
                            }
                        }
                   }

    sample_dict = {
                    "annotation":{
                        "folder":"train",
                        "filename":"photo_2021-08-01_12-59-28.jpg",
                        "path":"/mnt/sdb4/romachandr/PycharmProjects/tensorflow1/models/research/object_detection/images/train/photo_2021-08-01_12-59-28.jpg",
                        "source": {
                            "database":"Unknown"
                            },
                        "size":{
                            "width":1280,
                            "height":720,
                            "depth":3
                        },
                        "segmented":0,
                    }
               }
    object_dict = {
                    "name":"unknown_class",
                    "pose":"Unspecified",
                    "truncated":0,
                    "difficult":0,
                    "bndbox":{
                        "xmin":111,
                        "ymin":222,
                        "xmax":333,
                        "ymax":444
                    }
                }

    def label_base_file_save(self, folder, filename, path, pic_width, pic_height, pic_depth, label_name,
                        bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax,
                        delete_encoding_str='<?xml version="1.0" encoding="utf-8" ?>'):
        xml_dict = self.sample_dict_base.copy()
        xml_dict['annotation']['folder'] = folder
        xml_dict['annotation']['filename'] = filename
        xml_dict['annotation']['path'] = path
        xml_dict['annotation']['size']['width'] = pic_width
        xml_dict['annotation']['size']['height'] = pic_height
        xml_dict['annotation']['size']['depth'] = pic_depth
        xml_dict['annotation']['object']['name'] = label_name
        xml_dict['annotation']['object']['bndbox']['xmin'] = bndbox_xmin
        xml_dict['annotation']['object']['bndbox']['ymin'] = bndbox_ymin
        xml_dict['annotation']['object']['bndbox']['xmax'] = bndbox_xmax
        xml_dict['annotation']['object']['bndbox']['ymax'] = bndbox_ymax

        # print(xml_dict)
        if delete_encoding_str!= '':
            xml_dict_str = xmltodict.unparse(xml_dict, pretty=True).replace(delete_encoding_str, '')
        else:
            xml_dict_str = xmltodict.unparse(xml_dict, pretty=True)
        # print(xml_dict_str)
        xml_filename = filename.split('.')[0] + '.xml'
        xml_filepath = path.replace(filename, xml_filename)
        with open(xml_filepath, 'wt') as fl:
            fl.write(xml_dict_str)
            # fl.save()
        return

    def label_file_save(self, folder, filename, path, pic_width, pic_height, pic_depth, objects_array,
                        delete_encoding_str='<?xml version="1.0" encoding="utf-8"?>'):
        # label_name, bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax
        xml_dict = self.sample_dict.copy()
        xml_dict['annotation']['folder'] = folder
        xml_dict['annotation']['filename'] = filename
        xml_dict['annotation']['path'] = path
        xml_dict['annotation']['size']['width'] = pic_width
        xml_dict['annotation']['size']['height'] = pic_height
        xml_dict['annotation']['size']['depth'] = pic_depth

        for i, object in enumerate(objects_array):
            xml_dict['annotation'][f'object{i}'] = copy.deepcopy(self.object_dict)
            xml_dict['annotation'][f'object{i}']['name'] = object['label_name']
            xml_dict['annotation'][f'object{i}']['bndbox']['xmin'] = object['bndbox_xmin']
            xml_dict['annotation'][f'object{i}']['bndbox']['ymin'] = object['bndbox_ymin']
            xml_dict['annotation'][f'object{i}']['bndbox']['xmax'] = object['bndbox_xmax']
            xml_dict['annotation'][f'object{i}']['bndbox']['ymax'] = object['bndbox_ymax']

        # print(xml_dict)
        if delete_encoding_str!= '':
            xml_dict_str = xmltodict.unparse(xml_dict, pretty=True).replace(delete_encoding_str, '')
        else:
            xml_dict_str = xmltodict.unparse(xml_dict, pretty=True)

        for j in range(0, len(objects_array)):
            xml_dict_str = xml_dict_str.replace(f'object{j}', 'object')


        print(xml_dict_str)
        xml_filename = filename.split('.')[0] + '.xml'
        xml_filepath = path.replace(filename, xml_filename)
        with open(xml_filepath, 'wt') as fl:
            fl.write(xml_dict_str)
            # fl.save()
        return

    def xml_to_csv(self, folder, filename_arr):
        xml_list = []
        for file in filename_arr:
            xml_file = join(folder, file)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df

    def drop_encoding_str(self, folder_path, delete_encoding_str='<?xml version="1.0" encoding="utf-8" ?>'):

        files_arr = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        # print(files_arr)
        for file in files_arr:
            file_path = join(folder_path, file)
            txt_arr = None
            with open(file_path, 'rt') as fl:
                txt_arr = fl.readlines()
                # print(txt_arr)
                if txt_arr[0].find('encoding')>=0:
                    txt_arr = txt_arr[1:].copy()
                # print(txt_arr)
            with open(file_path, 'wt') as fl:
                for txt in txt_arr:
                    fl.writelines(txt)

        return


if __name__ == '__main__':
    LB = classLabeling()

    #---
    folder = r'c:\Users\Xiaomi\Downloads\test'
    filename = 'test_100.jpg'
    path = r'c:\Users\Xiaomi\Downloads\test\test_100.jpg'
    pic_width = 394
    pic_height = 500
    pic_depth = 3
    objects_array = [{'label_name':'car',
                    'bndbox_xmin':151,
                    'bndbox_ymin':71,
                    'bndbox_xmax':335,
                    'bndbox_ymax':267},
                    {'label_name': 'person',
                    'bndbox_xmin': 52,
                    'bndbox_ymin': 172,
                    'bndbox_xmax': 246,
                    'bndbox_ymax': 302}]

    LB.label_file_save(folder=folder, filename=filename, path=path, pic_width=pic_width, pic_height=pic_height, \
                       pic_depth=pic_depth, objects_array=objects_array)

    # #---
    # folder_path = r'c:\Users\Xiaomi\Downloads\dataset\annotations\xmls'
    # LB.drop_encoding_str(folder_path, delete_encoding_str='<?xml version="1.0" encoding="utf-8" ?>')
    # #---

    # #---
    #
    # folder = r'c:\Users\Xiaomi\Downloads\dataset\annotations\xmls'
    #
    # annotations_folder = r'c:\Users\Xiaomi\Downloads\dataset\annotations'
    # for file in ['trainval.txt', 'test.txt']:
    #     file_path = join(annotations_folder, file)
    #     with open(file_path, 'rt') as fl:
    #         filename_arr = fl.readlines()
    #     filename_arr = filename_arr[1:]
    #     filename_arr = [i.split(" ")[0]+'.xml' for i in filename_arr]
    #     xml_df = LB.xml_to_csv(folder, filename_arr)
    #     xml_df.to_csv(join(annotations_folder, file.split('.')[0]+'_labels.csv'), index=None)
    #     print('Successfully converted xml to csv.')

import xmltodict

class classLabeling:
    sample_dict = {"annotation":{
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
                            "name":"car",
                            "pose":"Unspecified",
                            "truncated":0,
                            "difficult":0,
                            "bndbox":{
                                "xmin":859,
                                "ymin":23,
                                "xmax":911,
                                "ymax":202
                            }
                        }
                    }
               }

    def label_file_save(self, folder, filename, path, pic_width, pic_height, pic_depth, label_name,
                        bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax):
        xml_dict = self.sample_dict.copy()
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
        xml_dict_str = xmltodict.unparse(xml_dict, pretty=True)
        # print(xml_dict_str)
        xml_filename = filename.split('.')[0] + '.xml'
        xml_filepath = path.replace(filename, xml_filename)
        with open(xml_filepath, 'wt') as fl:
            fl.write(xml_dict_str)
            # fl.save()
        return


if __name__ == '__main__':
    LB = classLabeling()
    folder = 'images'
    filename = 'Abyssinian_100.jpg'
    path = 'images/Abyssinian_100.jpg'
    pic_width = 394
    pic_height = 500
    pic_depth = 3
    label_name = 'test_label'
    bndbox_xmin = 151
    bndbox_ymin = 71
    bndbox_xmax = 335
    bndbox_ymax = 267
    LB.label_file_save(folder, filename, path, pic_width, pic_height, pic_depth, label_name,
                        bndbox_xmin, bndbox_ymin, bndbox_xmax, bndbox_ymax)
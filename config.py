from easydict import EasyDict as edict

cfg = edict()

cfg.CLASSES_voc = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)

cfg.CLASSES_ours_15cls = (
    'pencil',
    'highlighter',
    'penholder',
    'tape',
    'soap',
    'scissors',
    'dumbbells',
    'toothbrush',
    'bottle',
    'indexcards',
    'boarderaser',
    'glue',
    'baseball',
    'screwdriver',
    'crayon',
)

cfg.VOC07images = '/opt/VOCdevkit/VOC2007/JPEGImages/'
cfg.list_file_train_voc07 = './label_list/voc07_train.txt'
cfg.list_file_test_voc07  = './label_list/voc07_test.txt'

cfg.VOC12images = '/opt/VOCdevkit/VOC2012/JPEGImages/'
cfg.list_file_train_voc12 = './label_list/voc12_train.txt'
cfg.list_file_test_voc12  = './label_list/voc12_test.txt'

cfg.ours_15cls_images = '/home/share/dataset/VOC2007For15Cls/JPEGImages/'
cfg.list_file_train_ours_15cls = './label_list/ours_15cls_train.txt'
cfg.list_file_test_ours_15cls = './label_list/ours_15cls_test.txt'

cfg.ours_15cls_4300_images = '/home/share/dataset/VOC2007For4300/JPEGImages/'
cfg.list_file_train_ours_15cls_4300 = './label_list/ours_15cls_4300_train.txt'
cfg.list_file_train_ours_15cls_4300 = './label_list/ours_15cls_4300_train3.txt'
cfg.list_file_test_ours_15cls_4300 = './label_list/ours_15cls_4300_test.txt'

cfg.ours_15cls_4600_images = '/home/share/dataset/VOC2007For4600/JPEGImages/'
cfg.list_file_train_ours_15cls_4600 = './label_list/ours_15cls_4600_train.txt'
#cfg.list_file_train_ours_15cls_4600 = './label_list/ours_15cls_4600_train2.txt'
cfg.list_file_test_ours_15cls_4600 = './label_list/ours_15cls_4600_test.txt'

cfg.ours_15cls_4300_Tmp_50_images = '/home/share/dataset/VOC2007For4300_Tmp_50/JPEGImages/'
cfg.list_file_train_ours_15cls_4300_Tmp_50 = './label_list/ours_15cls_4300_Tmp_50_train3.txt'
#cfg.list_file_train_ours_15cls_4600 = './label_list/ours_15cls_4600_train2.txt'
cfg.list_file_test_ours_15cls_4300_Tmp_50 = './label_list/ours_15cls_4300_test.txt'

cfg.supplement100_images = '/home/share/dataset/supplement100/JPEGImages/'
cfg.list_file_train_supplement100 = './label_list/supplement100_train.txt'
cfg.list_file_test_supplement100 = './label_list/supplement100_test.txt'

cfg.supplement100_images = '/home/ellin/Downloads/supplement/JPEGImages/'
cfg.list_file_train_supplement100 = './label_list/supplement100_train.txt'
cfg.list_file_test_supplement100 = './label_list/supplement100_test.txt'

cfg.CLASSES = cfg.CLASSES_ours_15cls
cfg.img_root = cfg.supplement100_images
cfg.label_train = cfg.list_file_train_supplement100
cfg.label_test = cfg.list_file_test_supplement100

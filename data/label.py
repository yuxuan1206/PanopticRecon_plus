#!/usr/bin/python
#
# KITTI-360 labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.
    'raw_id'      , 

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!
labels = []
def get_labels(scene):
    if scene == '0087_02':
        # ## 87
        labels = [
            #       name                     id   raw_id,    color
            Label(  'unlabeled'            ,  0 ,     0 ,   (100,100,100) ),
            Label(  'floor'                ,  1 ,     1 ,   (152,223,138) ),
            Label(  'chair'                ,  2 ,     2 ,   (188,189, 34) ),
            Label(  'couch'                ,  3 ,     2 ,   (140, 86, 75) ),
            Label(  'table'                ,  4 ,     3,    (255,152,150) ),
            Label(  'wall'                 ,  5 ,     4,    (174,199,232) ),
            Label(  'telephone'            ,  6 ,     5,    (255,127, 14) ),
            Label(  'curtain'              ,  7 ,     5,    (219,219,141) ),
            Label(  'door'                 ,  8 ,     5,    (214, 39, 40) ),
            Label(  'clothes'              ,  9 ,     5,    (94, 106,211) ),
        ]

    elif scene == '0088_00':
        # ## 88
        labels = [
            #       name                     id   raw_id,    color
            Label(  'unlabeled'            ,  0 ,     0 ,   (100,100,100) ),
            Label(  'floor'                ,  1 ,     1 ,   (152,223,138) ),
            Label(  'chair'                ,  2 ,     2 ,   (188,189, 34) ),
            Label(  'table'                ,  3 ,     3 ,   (255,152,150) ),
            Label(  'wall'                 ,  4 ,     4 ,   (174,199,232) ),
            Label(  'whiteboard'           ,  5 ,     5 ,   (96, 207,209) ),
            Label(  'trash can'            ,  6 ,     6 ,   ( 82, 84,163) ),
            Label(  'door'                 ,  7 ,     7 ,   (214, 39, 40) ),
        ]

    elif scene == '0420_01':
        labels = [
            #       name                     id   raw_id,    color
            Label(  'unlabeled'            ,  0 ,     0 ,   (100,100,100) ),
            Label(  'floor'                ,  1 ,     1 ,   (152,223,138) ),
            Label(  'chair'                ,  2 ,     2 ,   (188,189, 34) ),
            Label(  'table'                ,  3 ,     3,    (255,152,150) ),
            Label(  'wall'                 ,  4 ,     4,    (174,199,232) ),
            Label(  'trash can'            ,  5 ,     5,    ( 82, 84,163) ),
            Label(  'window'               ,  6 ,     5,    (197,176,213) ),
            Label(  'door'                 ,  7 ,     5,    (214, 39, 40) ),
            Label(  'blackboard'           ,  8 ,     5,    (96, 207,209) ),
            Label(  'cabinet'              ,  9 ,     5,    ( 31,119,180) ),
            Label(  'other'                , 10 ,     5,    (100, 85,144) ),
        ]

    elif scene == '0628_02':
        labels = [
            #       name                     id   raw_id,    color
            Label(  'unlabeled'            ,  0 ,     0 ,   (100,100,100) ),
            Label(  'floor'                ,  1 ,     1 ,   (152,223,138) ),
            Label(  'chair'                ,  2 ,     2 ,   (188,189, 34) ),
            Label(  'table'                ,  3 ,     3,    (255,152,150) ),
            Label(  'wall'                 ,  4 ,     4,    (174,199,232) ),
            Label(  'blackboard'           ,  5 ,     5,    (96, 207,209) ),
            Label(  'window'               ,  6 ,     5,    (197,176,213) ),
            Label(  'trash can'            ,  7 ,     5,    ( 82, 84,163) ),
            Label(  'backpack'             ,  8 ,     5,    ( 44,160, 44) ),
            Label(  'clothes'              ,  9 ,     5,    ( 94,106,211) ),
            Label(  'box'                  , 10 ,     5,    (148,103,189) ),
            Label(  'laptop'               , 11 ,     5,    (112,128,144) ),
            Label(  'door'                 , 12 ,     5,    (214, 39, 40) ),
            Label(  'other'                , 13 ,     5,    (100, 85,144) ),
        ]

    elif scene == '5748ce6f01':
        labels = [
            #       name                     id   raw_id,    color
            Label(  'unlabeled'            ,  0 ,     0 ,   (100,100,100) ),
            Label(  'floor'                ,  1 ,     1 ,   (152,223,138) ),
            Label(  'chair'                ,  2 ,     2 ,   (188,189, 34) ),
            Label(  'table'                ,  3 ,     3,    (255,152,150) ),
            Label(  'wall'                 ,  4 ,     4,    (174,199,232) ),
            Label(  'whiteboard'           ,  5 ,     5,    ( 96,207,209) ),
            Label(  'tv'                   ,  6 ,     5,    ( 78, 71,183) ),
            Label(  'door'                 ,  7 ,     5,    (214, 39, 40) ),
            Label(  'ceiling'              ,  8 ,     5,    (140, 57,197) ),
            Label(  'other'                ,  9 ,     5,    (100, 85,144) ),
        ]

    elif scene == '1ada7a0617':
        labels = [
            #       name                     id   raw_id,   color
            Label(  'unlabeled'            ,  0 ,     0 ,   (100,100,100) ),
            Label(  'floor'                ,  1 ,     1 ,   (152,223,138) ),
            Label(  'chair'                ,  2 ,     2 ,   (188,189, 34) ),
            Label(  'table'                ,  3 ,     3,    (255,152,150) ),
            Label(  'wall'                 ,  4 ,     4,    (174,199,232) ),
            Label(  'whiteboard'           ,  5 ,     5,    (96, 207,209) ),
            Label(  'cabinet'              ,  6 ,     5,    ( 31,119,180) ),
            Label(  'door'                 ,  7 ,     5,    (214, 39, 40) ),
            Label(  'ceiling'              ,  8 ,     5,    (140, 57,197) ),
            Label(  'curtain'              ,  9 ,     5,    (219,219,141) ),
            Label(  'trash can'            , 10 ,     5,    ( 82, 84,163) ),
            Label(  'display'              , 11 ,     5,    ( 78, 71,183) ),
            Label(  'keyboard'             , 12 ,     5,    ( 23,190,207) ),
            Label(  'other'                , 13 ,     5,    (100, 85,144) ),
        ]

    elif scene == 'f6659a3107':
        labels = [
            #       name                     id   raw_id,    color
            Label(  'unlabeled'            ,  0 ,     0 ,   (100,100,100) ),
            Label(  'floor'                ,  1 ,     1 ,   (152,223,138) ),
            Label(  'chair'                ,  2 ,     2 ,   (188,189, 34) ),
            Label(  'table'                ,  3 ,     3,    (255,152,150) ),
            Label(  'wall'                 ,  4 ,     4,    (174,199,232) ),
            Label(  'whiteboard'           ,  5 ,     5,    ( 96,207,209) ),
            Label(  'door'                 ,  6 ,     5,    (214, 39, 40) ),
            Label(  'ceiling'              ,  7 ,     5,    (140, 57,197) ),
            Label(  'window'               ,  8 ,     5,    (197,176,213) ),
            Label(  'tv'                   ,  9 ,     5,    ( 78, 71,183) ),
            Label(  'other'                , 10 ,     5,    (100, 85,144) ),
        ]

    id2label= { label.id      : label for label in labels           }
    return id2label


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# KITTI-360 ID to cityscapes ID
kittiId2label   = { label.kittiId : label for label in labels           }

color2label     = { label.color   : label for label in labels           }

# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


import numpy as np
import colorsys
def id2rgb(id):
    # Convert ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3,), dtype=np.uint8)
    if id==0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)
    return rgb

import os

def get_scene_config(dataset, scene, dataset_root_path="/mnt/nas_new/yx/dataset/"):
    """
    Get scene configuration
    
    Args:
        dataset: Dataset name, such as 'scannet', 'scannet++', etc.
        scene: Scene ID, such as '0087_02'
        dataset_root_path: Dataset root directory
        
    Returns:
        Dictionary containing scene configuration
    """
    config = {
        "dataset_root_path": dataset_root_path
    }
    
    if dataset == 'scannet':
        config["image_path"] = os.path.join(dataset_root_path, f"scannet/scans/scene{scene}/color")
        config["save_path"] = os.path.join(dataset_root_path, f"scannet/GroundedSAM/scene{scene}")
        
        # scannet 0087_02
        if scene == '0087_02':
            config["text_prompt"] = "floor.chair.couch.sofa.table.wall.telephone.curtain.door.clothes."
            config["class_name"] = config["text_prompt"].split('.')
            config["class_label"] = [1,2,3,3,4,5,6,7,8,9]
            config["thing"] = [0,1,1,1,0,0,0,0,0]
        
        # scannet 0088_00
        elif scene == '0088_00':
            config["text_prompt"] = "floor.chair.table.wall.whiteboard.trash can.door."
            config["class_name"] = config["text_prompt"].split('.')
            config["class_label"] = [1,2,3,4,5,6,7]
            config["thing"] = [0,1,1,0,0,0,0]

        # scannet 0420_01
        elif scene == '0420_01':
            config["text_prompt"] = "floor.chair.table.wall.wall skirting.dustbin.trash can.window.door.blackboard.cabinet.projector"
            config["class_name"] = config["text_prompt"].split('.')
            config["class_label"] = [1,2,3,4,4,5,5,6,7,8,9,10]
            config["thing"] = [0,1,1,0,1,0,0,0,1,0]

        # scannet 0628_02
        elif scene == '0628_02':
            config["text_prompt"] = "floor.chair.table.desk.wall.column.walltile.limewall.skirting.blackboard.window.trashcan.recyclebin.backpack.clothes.cardboard box.laptop.door.paper.book.headset.cup.clock.backpack eraser"
            config["class_name"] = config["text_prompt"].split('.')
            config["class_label"] = [1,2,3,3,4,4,4,4,4,5,6,7,7,8,9,10,11,12,13,13,13,13,13,13]
            config["thing"] = [0,1,1,0,0,0,1,1,0,1,1,0,0]
            
    elif dataset == 'scannet++':
        config["image_path"] = os.path.join(dataset_root_path, f"scannet++/scans/scene{scene}/color")
        config["save_path"] = os.path.join(dataset_root_path, f"scannet++/GroundedSAM/scene{scene}")

        # 5748ce6f01
        if scene == '5748ce6f01':
            config["text_prompt"] = "floor.chair.armchair.table.coffee table.wall.wainscoting.column.whiteboard.tv.tv stand.door.ceiling.cup.ceiling lamp.remote control"
            config["class_name"] = config["text_prompt"].split('.')
            config["class_label"] = [1,2,2,3,3,4,4,4,5,6,6,7,8,9,9,9]
            config["thing"] = [0,1,1,0,1,1,1,0,0]

        # 1ada7a0617
        elif scene == '1ada7a0617':
            config["text_prompt"] = "floor.chair.office chair.table.desk.wall.wainscoting.whiteboard.cabinet.storage cabinet.cupboard.door.doorframe.ceiling.ceiling beam.blinds.trashcan.bucket.monitor.screen.keyboard.ceiling lamp.fan.telephone.mouse.heater.windowsill.computer tower.bottle.light switch.clothes hanger.socket.electric wire"
            config["class_name"] = config["text_prompt"].split('.')
            config["class_label"] = [1,2,2,3,3,4,4,5,6,6,6,7,7,8,8,9,10,10,11,11,12,13,13,13,13,13,13,13,13,13,13,13]
            config["thing"] = [0,1,1,0,1,1,1,0,0,1,1,1,0]

        # f6659a3107
        elif scene == 'f6659a3107':
            config["text_prompt"] = "floor.chair.office chair.table.desk.wall.wainscoting.whiteboard.door.doorframe.ceiling.ceiling beam.window.tv.ceiling lamp.heater.light switch.clothes hanger.socket.electric wire"
            config["class_name"] = config["text_prompt"].split('.')
            config["class_label"] = [1,2,2,3,3,4,4,5,6,6,7,7,8,9,10,10,10,10,10,10,10,10]
            config["thing"] = [0,1,1,0,1,1,0,0,1,0]
            
    return config 
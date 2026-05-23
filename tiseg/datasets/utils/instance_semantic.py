import numpy as np
from skimage import morphology


def re_instance(instance_map):
    """convert sparse instance ids to continual instance ids for instance
    map."""
    instance_ids = list(np.unique(instance_map))
    instance_ids.remove(0) if 0 in instance_ids else None
    new_instance_map = np.zeros_like(instance_map, dtype=np.int32)

    for id, instance_id in enumerate(instance_ids):
        new_instance_map[instance_map == instance_id] = id + 1

    return new_instance_map


def convert_instance_to_semantic(instance_map, with_edge=True):
    """Convert instance mask to semantic mask.

    Args:
        instances (numpy.ndarray): The mask contains each instances with
            different label value.
        with_edge (bool): Convertion with edge class label.

    Returns:
        mask (numpy.ndarray): mask contains two or three classes label
            (background, nuclei)
    """
    mask = np.zeros_like(instance_map, dtype=np.uint8)
    instance_ids = list(np.unique(instance_map))
    for instance_id in instance_ids:
        single_instance_map = (instance_map == instance_id).astype(np.uint8)
        if with_edge:
            boundary = morphology.dilation(single_instance_map) & (~morphology.erosion(single_instance_map))
            mask += single_instance_map
            mask[boundary > 0] = 2
        else:
            mask += single_instance_map

    return mask


def get_tc_from_inst(inst_seg):
    """Calculate three class segmentation mask from instance map."""
    tc_sem_seg = np.zeros_like(inst_seg)
    inst_id_list = list(np.unique(inst_seg))
    # TODO: move it to dataset conversion
    for inst_id in inst_id_list:
        if inst_id == 0:
            continue
        inst_id_mask = inst_seg == inst_id
        bound = inst_id_mask & (~morphology.erosion(inst_id_mask, selem=morphology.selem.disk(2)))
        tc_sem_seg[inst_id_mask > 0] = 1
        tc_sem_seg[bound > 0] = 2

    return tc_sem_seg


def to_one_hot(mask, num_classes):
    ret = np.zeros((num_classes, *mask.shape))
    for i in range(num_classes):
        ret[i, mask == i] = 1

    return ret


def assign_sem_class_to_insts(inst_seg, sem_seg, num_classes):
    inst_id_list = list(np.unique(inst_seg))

    if 0 not in inst_id_list:
        inst_id_list.insert(0, 0)

    sem_seg_one_hot = to_one_hot(sem_seg, num_classes)

    # Remove background class
    inst_id_list_per_class = {}
    for inst_id in inst_id_list:
        inst_mask = (inst_seg == inst_id).astype(np.uint8)

        tp = np.sum(inst_mask * sem_seg_one_hot, axis=(-2, -1))

        if np.sum(tp[1:]) > 0 and inst_id != 0:
            belong_sem_id = np.argmax(tp[1:]) + 1
        else:
            belong_sem_id = 0

        if belong_sem_id not in inst_id_list_per_class:
            inst_id_list_per_class[belong_sem_id] = [inst_id]
        else:
            inst_id_list_per_class[belong_sem_id].append(inst_id)

    return inst_id_list_per_class

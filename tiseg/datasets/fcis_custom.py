import os
import os.path as osp
import warnings
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from tiseg.utils import (pre_eval_all_semantic_metric, pre_eval_to_sem_metrics, pre_eval_bin_aji, pre_eval_bin_pq,
                         pre_eval_to_aji, pre_eval_to_pq, pre_eval_to_inst_dice, pre_eval_to_imw_pq,
                         pre_eval_to_imw_aji, pre_eval_to_imw_inst_dice, pre_eval_to_imw_sem_metrics,
                         pre_eval_to_bin_aji, pre_eval_to_bin_pq)

from .builder import DATASETS
from .dataset_mapper import DatasetMapper
from .utils import colorize_seg_map, re_instance, get_tc_from_inst


def draw_all(save_folder,
             img_name,
             img_file_name,
             sem_pred,
             sem_gt,
             inst_pred,
             inst_gt,
             tc_sem_pred,
             tc_sem_gt,
             edge_id=2,
             sem_palette=None):
    
    # print("sem gt", np.unique(sem_gt))

    print('inst pred', img_name)
    print('=' * 100)
    np.save(f"/data1/zhangye/FCIS/zy_pannuke_instmap/{img_name}.npy", inst_pred)

    plt.figure(figsize=(5 * 4, 5 * 2 + 3))

    # image drawing
    img = cv2.imread(img_file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(241)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image', fontsize=15, color='black')

    canvas = np.zeros((*sem_pred.shape, 3), dtype=np.uint8)
    canvas[(sem_pred > 0) * (sem_gt > 0), :] = (0, 0, 255)
    canvas[canvas == edge_id] = 0
    canvas[(sem_pred == 0) * (sem_gt > 0), :] = (0, 255, 0)
    canvas[(sem_pred > 0) * (sem_gt == 0), :] = (255, 0, 0)
    plt.subplot(242)
    plt.imshow(canvas)
    plt.axis('off')
    plt.title('Error Analysis: FN-FP-TP', fontsize=15, color='black')

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    label_list = [
        'TP',
        'FN',
        'FP',
    ]
    for color, label in zip(colors, label_list):
        color = list(color)
        color = [x / 255 for x in color]
        plt.plot(0, 0, '-', color=tuple(color), label=label)
    plt.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

    plt.subplot(243)
    plt.imshow(colorize_seg_map(inst_pred))
    plt.axis('off')
    plt.title('Instance Level Prediction')

    plt.subplot(244)
    plt.imshow(colorize_seg_map(inst_gt))
    plt.axis('off')
    plt.title('Instance Level Ground Truth')

    plt.subplot(245)
    plt.imshow(colorize_seg_map(sem_pred, sem_palette))
    plt.axis('off')
    plt.title('Semantic Level Prediction')

    plt.subplot(246)
    plt.imshow(colorize_seg_map(sem_gt, sem_palette))
    plt.axis('off')
    plt.title('Semantic Level Ground Truth')

    tc_palette = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]

    # print("tc_sem_pred", tc_sem_pred.shape, np.unique(tc_sem_pred))
    # print("tc_sem_gt", tc_sem_gt.shape, np.unique(tc_sem_gt))

    plt.subplot(247)
    plt.imshow(colorize_seg_map(tc_sem_pred, sem_palette)) #tc_palette))
    plt.axis('off')
    plt.title('Three-class Semantic Level Prediction')

    plt.subplot(248)
    plt.imshow(colorize_seg_map(tc_sem_gt, tc_palette))
    plt.axis('off')
    plt.title('Three-class Semantic Level Ground Truth')

    # results visulization
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{img_name}_compare.png', dpi=300)


@DATASETS.register_module()
class FCISCustomDataset(Dataset):
    """Nuclei Custom Foundation Segmentation Dataset.
    Although, this dataset is a instance segmentation task, this dataset also
    support a multiple class semantic segmentation task (Background, Nuclei1, Nuclei2, ...).

    related suffix:
        "_sem.png": semantic level label map (multiple class semantic map without boundary).
        "_inst.npy": instance level label map.
    """

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self,
                 processes,
                 img_dir,
                 ann_dir,
                 adj_dir,
                 dis_dir,
                 data_root=None,
                 img_suffix='.tif',
                 sem_suffix='_sem.png',
                 inst_suffix='_inst.npy',
                 adj_suffix='_adj.yaml',
                 dis_suffix='_inst.npy',
                 test_mode=False,
                 split=None):

        self.mapper = DatasetMapper(test_mode, processes=processes)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.adj_dir = adj_dir
        self.dis_dir = dis_dir
        self.data_root = data_root

        self.img_suffix = img_suffix
        self.sem_suffix = sem_suffix
        self.inst_suffix = inst_suffix
        self.adj_suffix = adj_suffix
        self.dis_suffix = dis_suffix

        self.test_mode = test_mode
        self.split = split

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.adj_dir is None or osp.isabs(self.adj_dir)):
                self.adj_dir = osp.join(self.data_root, self.adj_dir)
            if not (self.dis_dir is None or osp.isabs(self.dis_dir)):
                self.dis_dir = osp.join(self.data_root, self.dis_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        self.data_infos = self.load_annotations(self.img_dir, self.ann_dir, self.adj_dir, self.dis_dir, self.img_suffix, self.sem_suffix,
                                                self.inst_suffix, self.adj_suffix, self.dis_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def __getitem__(self, index):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        data_info = self.data_infos[index]
        return self.mapper(data_info)

    def load_annotations(self, img_dir, ann_dir, adj_dir, dis_dir, img_suffix, sem_suffix, inst_suffix, adj_suffix, dis_suffix, split=None):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory.
            ann_dir (str): Path to annotation directory.
            img_suffix (str): Suffix of images.
            ann_suffix (str): Suffix of segmentation maps.
            split (str | None): Split txt file. If split is specified, only
                file with suffix in the splits will be loaded.
        Returns:
            list[dict]: All data info of dataset, data info contains image,
                segmentation map.
        """
        data_infos = []
        if split is not None:
            with open(split, 'r') as fp:
                for line in fp.readlines():
                    img_id = line.strip()
                    img_name = img_id + img_suffix
                    sem_name = img_id + sem_suffix
                    inst_name = img_id + inst_suffix
                    adj_name = img_id + adj_suffix
                    dis_name = img_id + dis_suffix
                    img_file_name = osp.join(img_dir, img_name)
                    sem_file_name = osp.join(ann_dir, sem_name)
                    inst_file_name = osp.join(ann_dir, inst_name)
                    adj_file_name = osp.join(adj_dir, adj_name)
                    dis_file_name = osp.join(dis_dir, dis_name)

                    data_info = dict(
                        data_id=osp.splitext(img_name)[0],
                        file_name=img_file_name,
                        sem_file_name=sem_file_name,
                        inst_file_name=inst_file_name,
                        adj_file_name=adj_file_name,
                        dis_file_name=dis_file_name
                        )
                    data_infos.append(data_info)
        else:
            for img_name in mmcv.scandir(img_dir, img_suffix, recursive=True):
                sem_name = img_name.replace(img_suffix, sem_suffix)
                inst_name = img_name.replace(img_suffix, inst_suffix)
                img_file_name = osp.join(img_dir, img_name)
                sem_file_name = osp.join(ann_dir, sem_name)
                inst_file_name = osp.join(ann_dir, inst_name)
                adj_file_name = osp.join(adj_dir, adj_name)
                dis_file_name = osp.join(dis_dir, dis_name)
                data_info = dict(
                    data_id=osp.splitext(img_name)[0],
                    file_name=img_file_name,
                    sem_file_name=sem_file_name,
                    inst_file_name=inst_file_name,
                    adj_file_name=adj_file_name,
                    dis_file_name=dis_file_name)
                data_infos.append(data_info)
                

        return data_infos

    def pre_eval(self, preds, indices, show=False, show_folder=None):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
            show (bool): Illustrate semantic level & instance level prediction &
                ground truth. Default: False
            show_folder (str | None, optional): The folder path of
                illustration. Default: None
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        if show_folder is None and show:
            warnings.warn('show_semantic or show_instance is set to True, but the '
                          'show_folder is None. We will use default show_folder: '
                          '.nuclei_show')
            show_folder = '.nuclei_show'
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            # img related infos
            img_file_name = self.data_infos[index]['file_name']
            img_name = osp.splitext(osp.basename(img_file_name))[0]
            sem_file_name = self.data_infos[index]['sem_file_name']
            # semantic level label make
            sem_gt = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name']
            inst_gt = re_instance(np.load(inst_file_name))

            data_id = osp.basename(self.data_infos[index]['sem_file_name']).replace(self.sem_suffix, '')

            # metric calculation post process codes:
            # 'sem_pred' is a three-class map: 0-1-2 sem map w/ edge;
            sem_pred = pred['sem_pred']
            inst_pred = pred['inst_pred']

            # semantic metric calculation (remove background class)
            sem_pre_eval_res = pre_eval_all_semantic_metric(sem_pred, sem_gt, len(self.CLASSES))

            # make contiguous ids
            inst_pred = re_instance(inst_pred)
            inst_gt = re_instance(inst_gt)

            # instance metric calculation
            bin_aji_pre_eval_res = pre_eval_bin_aji(inst_pred, inst_gt)
            bin_pq_pre_eval_res = pre_eval_bin_pq(inst_pred, inst_gt)

            single_loop_results = dict(
                name=data_id,
                bin_aji_pre_eval_res=bin_aji_pre_eval_res,
                bin_pq_pre_eval_res=bin_pq_pre_eval_res,
                sem_pre_eval_res=sem_pre_eval_res)
            pre_eval_results.append(single_loop_results)

            # illustrating semantic level & instance level results
            if show:
                if 'tc_sem_pred' in pred:
                    tc_sem_pred = pred['tc_sem_pred']
                else:
                    tc_sem_pred = pred['sem_pred']
                tc_sem_gt = get_tc_from_inst(inst_gt)
                draw_all(
                    show_folder,
                    img_name,
                    img_file_name,
                    sem_pred,
                    sem_gt,
                    inst_pred,
                    inst_gt,
                    tc_sem_pred,
                    tc_sem_gt,
                )

        return pre_eval_results

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            processor (object): The result processor.
            metric (str | list[str]): Metrics to be evaluated. 'Aji',
                'Dice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """

        img_ret_metrics = {}
        ret_metrics = {}

        # list to dict
        for result in results:
            for key, value in result.items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        # All dataset
        # name id
        img_ret_metrics['name'] = ret_metrics.pop('name')

        # semantic metrics
        sem_pre_eval_results = ret_metrics.pop('sem_pre_eval_res')
        ret_metrics.update(pre_eval_to_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))
        img_ret_metrics.update(
            pre_eval_to_imw_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))

        # instance metrics (aji style)
        bin_aji_pre_eval_results = ret_metrics.pop('bin_aji_pre_eval_res')
        ret_metrics.update(pre_eval_to_aji(bin_aji_pre_eval_results))
        for k, v in pre_eval_to_bin_aji(bin_aji_pre_eval_results).items():
            ret_metrics['b' + k] = v
        img_ret_metrics.update(pre_eval_to_imw_aji(bin_aji_pre_eval_results))

        # instance metrics (pq style)
        bin_pq_pre_eval_results = ret_metrics.pop('bin_pq_pre_eval_res')
        ret_metrics.update(pre_eval_to_pq(bin_pq_pre_eval_results))
        for k, v in pre_eval_to_bin_pq(bin_pq_pre_eval_results).items():
            ret_metrics['b' + k] = v
        ret_metrics.update(pre_eval_to_inst_dice(bin_pq_pre_eval_results))
        img_ret_metrics.update(pre_eval_to_imw_pq(bin_pq_pre_eval_results))
        img_ret_metrics.update(pre_eval_to_imw_inst_dice(bin_pq_pre_eval_results))

        assert 'name' in img_ret_metrics
        name_list = img_ret_metrics.pop('name')
        name_list.append('Average')

        # insert average value into image wise list
        for key in img_ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            if len(img_ret_metrics[key].shape) == 2:
                img_ret_metrics[key] = img_ret_metrics[key][:, 0]
            average_value = np.nanmean(img_ret_metrics[key])
            img_ret_metrics[key] = img_ret_metrics[key].tolist()
            img_ret_metrics[key].append(average_value)
            img_ret_metrics[key] = np.array(img_ret_metrics[key])

        vital_keys = ['Dice', 'Precision', 'Recall', 'Aji', 'DQ', 'SQ', 'PQ', 'InstDice']
        mean_metrics = {}
        overall_metrics = {}
        # calculate average metric
        for key in vital_keys:
            # XXX: Using average value may have lower metric value than using
            mean_metrics['imw' + key] = img_ret_metrics[key][-1]
            overall_metrics['m' + key] = ret_metrics[key]

        for key in ['bAji', 'bDQ', 'bSQ', 'bPQ']:
            overall_metrics[key] = ret_metrics[key]

        # per sample
        sample_metrics = OrderedDict(
            {sample_key: np.round(metric_value * 100, 2)
             for sample_key, metric_value in img_ret_metrics.items()})
        sample_metrics.update({'name': name_list})
        sample_metrics.move_to_end('name', last=False)

        items_table_data = PrettyTable()
        for key, val in sample_metrics.items():
            items_table_data.add_column(key, val)

        print_log('Per samples:', logger)
        print_log('\n' + items_table_data.get_string(), logger=logger)

        # mean table
        mean_metrics = OrderedDict(
            {mean_key: np.round(np.mean(value) * 100, 2)
             for mean_key, value in mean_metrics.items()})

        mean_table_data = PrettyTable()
        for key, val in mean_metrics.items():
            mean_table_data.add_column(key, [val])

        # overall table
        overall_metrics = OrderedDict(
            {sem_key: np.round(np.mean(value) * 100, 2)
             for sem_key, value in overall_metrics.items()})

        overall_table_data = PrettyTable()
        for key, val in overall_metrics.items():
            overall_table_data.add_column(key, [val])

        print_log('Mean Total:', logger)
        print_log('\n' + mean_table_data.get_string(), logger=logger)
        print_log('Overall Total:', logger)
        print_log('\n' + overall_table_data.get_string(), logger=logger)

        storage_results = {
            'mean_metrics': mean_metrics,
            'overall_metrics': overall_metrics,
        }

        eval_results = {}

        for k, v in mean_metrics.items():
            eval_results[k] = v
        for k, v in overall_metrics.items():
            eval_results[k] = v

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results, storage_results

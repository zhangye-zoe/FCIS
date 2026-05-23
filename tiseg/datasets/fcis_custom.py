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

from tiseg.utils import (
    pre_eval_all_semantic_metric,
    pre_eval_bin_aji,
    pre_eval_bin_pq,
    pre_eval_to_aji,
    pre_eval_to_bin_aji,
    pre_eval_to_bin_pq,
    pre_eval_to_imw_aji,
    pre_eval_to_imw_inst_dice,
    pre_eval_to_imw_pq,
    pre_eval_to_imw_sem_metrics,
    pre_eval_to_inst_dice,
    pre_eval_to_pq,
    pre_eval_to_sem_metrics,
)

from .builder import DATASETS
from .dataset_mapper import DatasetMapper
from .utils import colorize_seg_map, get_tc_from_inst, re_instance


def draw_all(
    save_folder,
    img_name,
    img_file_name,
    sem_pred,
    sem_gt,
    inst_pred,
    inst_gt,
    tc_sem_pred,
    tc_sem_gt,
    edge_id=2,
    sem_palette=None,
):
    """Visualize image, semantic prediction, instance prediction, and errors."""

    plt.figure(figsize=(5 * 4, 5 * 2 + 3))

    # Original image.
    img = cv2.imread(img_file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(241)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Image", fontsize=15, color="black")

    # Error map: TP / FN / FP.
    canvas = np.zeros((*sem_pred.shape, 3), dtype=np.uint8)
    canvas[(sem_pred > 0) * (sem_gt > 0), :] = (0, 0, 255)
    canvas[canvas == edge_id] = 0
    canvas[(sem_pred == 0) * (sem_gt > 0), :] = (0, 255, 0)
    canvas[(sem_pred > 0) * (sem_gt == 0), :] = (255, 0, 0)

    plt.subplot(242)
    plt.imshow(canvas)
    plt.axis("off")
    plt.title("Error Analysis: FN-FP-TP", fontsize=15, color="black")

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labels = ["TP", "FN", "FP"]

    for color, label in zip(colors, labels):
        color = tuple(x / 255 for x in color)
        plt.plot(0, 0, "-", color=color, label=label)

    plt.legend(loc="upper center", fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

    # Instance-level visualization.
    plt.subplot(243)
    plt.imshow(colorize_seg_map(inst_pred))
    plt.axis("off")
    plt.title("Instance Level Prediction")

    plt.subplot(244)
    plt.imshow(colorize_seg_map(inst_gt))
    plt.axis("off")
    plt.title("Instance Level Ground Truth")

    # Semantic-level visualization.
    plt.subplot(245)
    plt.imshow(colorize_seg_map(sem_pred, sem_palette))
    plt.axis("off")
    plt.title("Semantic Level Prediction")

    plt.subplot(246)
    plt.imshow(colorize_seg_map(sem_gt, sem_palette))
    plt.axis("off")
    plt.title("Semantic Level Ground Truth")

    # Three-class semantic visualization.
    tc_palette = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]

    plt.subplot(247)
    plt.imshow(colorize_seg_map(tc_sem_pred, sem_palette))
    plt.axis("off")
    plt.title("Three-class Semantic Level Prediction")

    plt.subplot(248)
    plt.imshow(colorize_seg_map(tc_sem_gt, tc_palette))
    plt.axis("off")
    plt.title("Three-class Semantic Level Ground Truth")

    plt.tight_layout()
    plt.savefig(f"{save_folder}/{img_name}_compare.png", dpi=300)


@DATASETS.register_module()
class FCISCustomDataset(Dataset):
    """Custom nuclei segmentation dataset for FCIS.

    This dataset supports instance-level labels and semantic-level labels.

    Related suffixes:
        - "_sem.png": semantic-level label map.
        - "_inst.npy": instance-level label map.
    """

    CLASSES = ("background", "nuclei")
    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(
        self,
        processes,
        img_dir,
        ann_dir,
        adj_dir,
        dis_dir,
        data_root=None,
        img_suffix=".tif",
        sem_suffix="_sem.png",
        inst_suffix="_inst.npy",
        adj_suffix="_adj.yaml",
        dis_suffix="_inst.npy",
        test_mode=False,
        split=None,
    ):
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

        if self.data_root is not None:
            self._join_data_root()

        self.data_infos = self.load_annotations(
            self.img_dir,
            self.ann_dir,
            self.adj_dir,
            self.dis_dir,
            self.img_suffix,
            self.sem_suffix,
            self.inst_suffix,
            self.adj_suffix,
            self.dis_suffix,
            self.split,
        )

    def _join_data_root(self):
        """Join relative paths with ``data_root``."""
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

    def __len__(self):
        """Return the number of samples."""
        return len(self.data_infos)

    def __getitem__(self, index):
        """Return one sample after the data pipeline."""
        data_info = self.data_infos[index]
        return self.mapper(data_info)

    def load_annotations(
        self,
        img_dir,
        ann_dir,
        adj_dir,
        dis_dir,
        img_suffix,
        sem_suffix,
        inst_suffix,
        adj_suffix,
        dis_suffix,
        split=None,
    ):
        """Load image and annotation paths.

        Args:
            img_dir (str): Image directory.
            ann_dir (str): Annotation directory.
            adj_dir (str): Adjacency annotation directory.
            dis_dir (str): Distance/instance annotation directory.
            img_suffix (str): Image suffix.
            sem_suffix (str): Semantic label suffix.
            inst_suffix (str): Instance label suffix.
            adj_suffix (str): Adjacency file suffix.
            dis_suffix (str): Distance/instance file suffix.
            split (str | None): Optional split file. If provided, only samples
                listed in this file are loaded.

        Returns:
            list[dict]: Dataset metadata.
        """
        data_infos = []

        if split is not None:
            with open(split, "r") as fp:
                for line in fp.readlines():
                    img_id = line.strip()

                    img_name = img_id + img_suffix
                    sem_name = img_id + sem_suffix
                    inst_name = img_id + inst_suffix
                    adj_name = img_id + adj_suffix
                    dis_name = img_id + dis_suffix

                    data_info = dict(
                        data_id=osp.splitext(img_name)[0],
                        file_name=osp.join(img_dir, img_name),
                        sem_file_name=osp.join(ann_dir, sem_name),
                        inst_file_name=osp.join(ann_dir, inst_name),
                        adj_file_name=osp.join(adj_dir, adj_name),
                        dis_file_name=osp.join(dis_dir, dis_name),
                    )
                    data_infos.append(data_info)
        else:
            for img_name in mmcv.scandir(img_dir, img_suffix, recursive=True):
                sem_name = img_name.replace(img_suffix, sem_suffix)
                inst_name = img_name.replace(img_suffix, inst_suffix)

                # Keep the original path construction logic unchanged.
                data_info = dict(
                    data_id=osp.splitext(img_name)[0],
                    file_name=osp.join(img_dir, img_name),
                    sem_file_name=osp.join(ann_dir, sem_name),
                    inst_file_name=osp.join(ann_dir, inst_name),
                    adj_file_name=osp.join(adj_dir, adj_name),
                    dis_file_name=osp.join(dis_dir, dis_name),
                )
                data_infos.append(data_info)

        return data_infos

    def pre_eval(self, preds, indices, show=False, show_folder=None):
        """Collect pre-evaluation results for each prediction.

        Args:
            preds (list[dict] | dict): Model predictions.
            indices (list[int] | int): Dataset indices corresponding to preds.
            show (bool): Whether to save visualization results.
            show_folder (str | None): Visualization output folder.

        Returns:
            list[dict]: Per-sample pre-evaluation results.
        """
        if not isinstance(indices, list):
            indices = [indices]

        if not isinstance(preds, list):
            preds = [preds]

        if show_folder is None and show:
            warnings.warn(
                "show_semantic or show_instance is set to True, but "
                "show_folder is None. We will use default show_folder: "
                ".nuclei_show"
            )
            show_folder = ".nuclei_show"
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            img_file_name = self.data_infos[index]["file_name"]
            img_name = osp.splitext(osp.basename(img_file_name))[0]

            sem_file_name = self.data_infos[index]["sem_file_name"]
            sem_gt = mmcv.imread(sem_file_name, flag="unchanged", backend="pillow")
            sem_gt_draw = sem_gt.copy()
            sem_pred_draw = pred["fc_pred"]

            inst_file_name = self.data_infos[index]["inst_file_name"]
            inst_gt = re_instance(np.load(inst_file_name))

            data_id = osp.basename(self.data_infos[index]["sem_file_name"]).replace(
                self.sem_suffix, ""
            )

            # Dice / Precision / Recall use binary foreground-background masks.
            # Four-color prediction is used for instance decoding and
            # visualization, not for semantic Dice calculation.
            inst_pred = pred["inst_pred"]
            sem_pred = (inst_pred > 0).astype(np.uint8)
            sem_gt = (inst_gt > 0).astype(np.uint8)

            gt_empty = bool(inst_gt.max() == 0)
            pred_empty = bool(inst_pred.max() == 0)
            empty_empty = gt_empty and pred_empty
            empty_correct = int(empty_empty)

            sem_pre_eval_res = pre_eval_all_semantic_metric(
                sem_pred, sem_gt, len(self.CLASSES)
            )

            # Make instance ids contiguous before computing instance metrics.
            inst_pred = re_instance(inst_pred)
            inst_gt = re_instance(inst_gt)

            bin_aji_pre_eval_res = pre_eval_bin_aji(inst_pred, inst_gt)
            bin_pq_pre_eval_res = pre_eval_bin_pq(inst_pred, inst_gt)

            single_loop_results = dict(
                name=data_id,
                empty_gt=gt_empty,
                empty_pred=pred_empty,
                empty_empty=empty_empty,
                empty_correct=empty_correct,
                bin_aji_pre_eval_res=bin_aji_pre_eval_res,
                bin_pq_pre_eval_res=bin_pq_pre_eval_res,
                sem_pre_eval_res=sem_pre_eval_res,
            )
            pre_eval_results.append(single_loop_results)

            if show:
                tc_sem_pred = pred["tc_sem_pred"] if "tc_sem_pred" in pred else pred["sem_pred"]
                tc_sem_gt = get_tc_from_inst(inst_gt)

                draw_all(
                    show_folder,
                    img_name,
                    img_file_name,
                    sem_pred_draw,
                    sem_gt_draw,
                    inst_pred,
                    inst_gt,
                    tc_sem_pred,
                    tc_sem_gt,
                )

        return pre_eval_results

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate all pre-evaluation results.

        Args:
            results (list[dict]): Pre-evaluation results from ``pre_eval``.
            logger (logging.Logger | None | str): Logger for output.

        Returns:
            tuple[dict, dict]: Evaluation results and storage results.
        """
        img_ret_metrics = {}
        ret_metrics = {}

        # Convert list of dicts to dict of lists.
        for result in results:
            for key, value in result.items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        img_ret_metrics["name"] = ret_metrics.pop("name")

        # Empty-patch statistics are reported separately and are not folded into
        # DQ/SQ/PQ, because empty-empty samples have no detection target.
        empty_gt_list = np.array(ret_metrics.pop("empty_gt"), dtype=bool)
        empty_pred_list = np.array(ret_metrics.pop("empty_pred"), dtype=bool)
        empty_correct_list = np.array(ret_metrics.pop("empty_correct"), dtype=np.float32)

        img_ret_metrics["EmptyCorrect"] = empty_correct_list
        ret_metrics["EmptyAcc"] = (
            np.mean(empty_correct_list[empty_gt_list])
            if np.any(empty_gt_list)
            else np.nan
        )
        ret_metrics["NumEmptyGT"] = np.sum(empty_gt_list)
        ret_metrics["NumEmptyPred"] = np.sum(empty_pred_list)

        # Semantic metrics.
        sem_pre_eval_results = ret_metrics.pop("sem_pre_eval_res")
        ret_metrics.update(
            pre_eval_to_sem_metrics(
                sem_pre_eval_results, metrics=["Dice", "Precision", "Recall"]
            )
        )
        img_ret_metrics.update(
            pre_eval_to_imw_sem_metrics(
                sem_pre_eval_results, metrics=["Dice", "Precision", "Recall"]
            )
        )

        # AJI-style instance metrics.
        bin_aji_pre_eval_results = ret_metrics.pop("bin_aji_pre_eval_res")
        ret_metrics.update(pre_eval_to_aji(bin_aji_pre_eval_results))

        for key, value in pre_eval_to_bin_aji(bin_aji_pre_eval_results).items():
            ret_metrics["b" + key] = value

        img_ret_metrics.update(pre_eval_to_imw_aji(bin_aji_pre_eval_results))

        # PQ-style instance metrics.
        bin_pq_pre_eval_results = ret_metrics.pop("bin_pq_pre_eval_res")
        ret_metrics.update(pre_eval_to_pq(bin_pq_pre_eval_results))

        for key, value in pre_eval_to_bin_pq(bin_pq_pre_eval_results).items():
            ret_metrics["b" + key] = value

        ret_metrics.update(pre_eval_to_inst_dice(bin_pq_pre_eval_results))
        img_ret_metrics.update(pre_eval_to_imw_pq(bin_pq_pre_eval_results))
        img_ret_metrics.update(pre_eval_to_imw_inst_dice(bin_pq_pre_eval_results))

        empty_empty_list = empty_gt_list & empty_pred_list

        # Ignore empty-empty samples for image-wise instance metrics.
        for key in ["Aji", "DQ", "SQ", "PQ", "InstDice"]:
            if key in img_ret_metrics:
                metric_values = np.asarray(img_ret_metrics[key], dtype=np.float32)
                metric_values[empty_empty_list] = np.nan
                img_ret_metrics[key] = metric_values

        assert "name" in img_ret_metrics
        name_list = img_ret_metrics.pop("name")
        name_list.append("Average")

        # Add image-wise average as the final row.
        for key in img_ret_metrics.keys():
            if len(img_ret_metrics[key].shape) == 2:
                img_ret_metrics[key] = img_ret_metrics[key][:, 0]

            average_value = np.nanmean(img_ret_metrics[key])
            img_ret_metrics[key] = img_ret_metrics[key].tolist()
            img_ret_metrics[key].append(average_value)
            img_ret_metrics[key] = np.array(img_ret_metrics[key])

        vital_keys = [
            "Dice",
            "Precision",
            "Recall",
            "Aji",
            "DQ",
            "SQ",
            "PQ",
            "InstDice",
        ]

        mean_metrics = {}
        overall_metrics = {}

        for key in vital_keys:
            mean_metrics["imw" + key] = img_ret_metrics[key][-1]
            overall_metrics["m" + key] = ret_metrics[key]

        mean_metrics["imwEmptyCorrect"] = img_ret_metrics["EmptyCorrect"][-1]

        for key in [
            "bAji",
            "bDQ",
            "bSQ",
            "bPQ",
            "EmptyAcc",
            "NumEmptyGT",
            "NumEmptyPred",
        ]:
            overall_metrics[key] = ret_metrics[key]

        # Per-sample table.
        sample_metrics = OrderedDict(
            {
                sample_key: np.round(metric_value * 100, 2)
                for sample_key, metric_value in img_ret_metrics.items()
            }
        )
        sample_metrics.update({"name": name_list})
        sample_metrics.move_to_end("name", last=False)

        items_table_data = PrettyTable()
        for key, value in sample_metrics.items():
            items_table_data.add_column(key, value)

        print_log("Per samples:", logger)
        print_log("\n" + items_table_data.get_string(), logger=logger)

        # Mean table: display metric values as percentages.
        mean_metrics = OrderedDict(
            {
                mean_key: np.round(np.nanmean(value) * 100, 2)
                for mean_key, value in mean_metrics.items()
            }
        )

        mean_table_data = PrettyTable()
        for key, value in mean_metrics.items():
            mean_table_data.add_column(key, [value])

        # Overall table: keep count fields as counts, display metrics as percentages.
        overall_metrics = OrderedDict(
            {
                sem_key: (
                    np.round(np.nanmean(value), 2)
                    if sem_key.startswith("Num")
                    else np.round(np.nanmean(value) * 100, 2)
                )
                for sem_key, value in overall_metrics.items()
            }
        )

        overall_table_data = PrettyTable()
        for key, value in overall_metrics.items():
            overall_table_data.add_column(key, [value])

        print_log("Mean Total:", logger)
        print_log("\n" + mean_table_data.get_string(), logger=logger)
        print_log("Overall Total:", logger)
        print_log("\n" + overall_table_data.get_string(), logger=logger)

        storage_results = {
            "mean_metrics": mean_metrics,
            "overall_metrics": overall_metrics,
        }

        eval_results = {}

        for key, value in mean_metrics.items():
            eval_results[key] = value

        for key, value in overall_metrics.items():
            eval_results[key] = value

        # EvalHook writes these metrics into runner.log_buffer.output. Then
        # TextLoggerHook prints them to the logger.
        return eval_results, storage_results

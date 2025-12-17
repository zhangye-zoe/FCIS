import mmcv
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.runner import get_dist_info


def single_gpu_test(model, data_loader, pre_eval=False, pre_eval_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        pre_eval (bool): Whether to evaluation by pre_eval mode. Default: False
        pre_eval_args (dict): The arguments of `def pre_eval` of dataset. Default: {}

    Returns:
        object: The processor containing results.
    """
    # when none of them is set true, return segmentation results as
    # a list of np.array.

    model.eval()
    results = []
    dataset = data_loader.dataset

    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    total_time = 0
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            
            
            # import time
            # start_time = time.time()
            
            result = model(**data)

            # end_time = time.time()
            # inference_time = end_time - start_time
            # total_time += inference_time
            
            # print(f"  Inference time of current image: {inference_time:.4f} seconds")
            
    
            
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices, **pre_eval_args)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    # print(f"  Average inference time: {total_time/10:.4f} seconds")

    return results


def multi_gpu_test(model, data_loader, pre_eval=False, pre_eval_args={}):
    """Test model with multiple gpus by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        pre_eval_args (dict): The arguments of `def pre_eval` of dataset. Default: {}

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """

    # when none of them is set true, return segmentation results as
    # a list of np.array.

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices, **pre_eval_args)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    # NOTE: GPU memory is really expensive
    gpu_collect = False
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), None)
    return results

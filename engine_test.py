# Copyright (c) 2020-2023 WongKinYiu and Ultralytics LLC. All rights reserved.
#
# This software is licensed under the GPL-3.0 License.
# For details, see the LICENSE file in the root directory of this distribution.

import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Attempt to import TensorRT and PyCUDA, but don't make them hard dependencies
# if the user is only using PyTorch models.
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Handles CUDA context initialization and cleanup
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    # print("TensorRT or PyCUDA not found. TensorRT engine support will be disabled.")

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel


# TensorRT Inference Wrapper Class
class TRTModule:
    def __init__(self, engine_path, device_str, input_name, output_names_from_engine, stride, class_names_list):
        """
        Initializes the TensorRT inference module.

        Args:
            engine_path (str): Path to the TensorRT .engine file.
            device_str (str): Device string, e.g., 'cuda:0'. (Note: PyCUDA manages device context)
            input_name (str): Name of the input binding in the TensorRT engine.
            output_names_from_engine (list): List of output binding names from the TensorRT engine.
            stride (int): Max stride of the model.
            class_names_list (list): List of class names.
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT or PyCUDA is not installed. Please install them to use TensorRT engines.")

        self.logger = trt.Logger(trt.Logger.WARNING) # Or INFO for more verbosity
        self.engine_path = engine_path
        self.device_str = device_str # For PyTorch tensor placement if needed, PyCUDA handles context
        self.input_name = input_name
        self.class_names = {i: name for i, name in enumerate(class_names_list)}
        self.stride = torch.tensor([float(stride)]) # Make it a tensor like PyTorch model

        # Load the TensorRT engine
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create an execution context
        self.context = self.engine.create_execution_context()

        # Allocate buffers for input and output
        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}
        self.bindings = [None] * self.engine.num_bindings # Use list for easier binding by index
        self.output_shapes_from_engine = {} # Store output shapes from engine, keyed by name
        self.engine_output_names = output_names_from_engine # Store the order of output names as per engine

        self.input_binding_idx = -1
        self.input_shape_from_engine = None
        self.input_dtype = None

        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            is_input = self.engine.binding_is_input(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            shape = tuple(self.engine.get_binding_shape(binding_idx)) # Shape from engine (may contain -1 for dynamic)

            if is_input:
                if binding_name == self.input_name:
                    self.input_binding_idx = binding_idx
                    self.input_shape_from_engine = shape
                    self.input_dtype = dtype
                    # Device input buffer will be allocated in __call__ if dynamic,
                    # or here if static and max batch size is known.
                    # For simplicity, we'll handle dynamic allocation in __call__.
                    if -1 not in shape: # Static shape
                        vol = trt.volume(shape)
                        if vol < 0: vol *= -1 # For older TRT versions if batch dim is not first
                        h_input = cuda.pagelocked_empty(vol, dtype=dtype)
                        d_input = cuda.mem_alloc(h_input.nbytes)
                        self.host_inputs[binding_name] = h_input
                        self.device_inputs[binding_name] = d_input
                        self.bindings[binding_idx] = int(d_input)
                else:
                    print(f"Warning: Engine has multiple input bindings. Only '{self.input_name}' will be used actively.")
            else: # Output
                self.output_shapes_from_engine[binding_name] = shape
                vol = trt.volume(shape)
                if vol < 0: vol *= -1
                h_output = cuda.pagelocked_empty(vol, dtype=dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)
                self.host_outputs[binding_name] = h_output
                self.device_outputs[binding_name] = d_output
                self.bindings[binding_idx] = int(d_output)
        
        if self.input_binding_idx == -1:
            raise ValueError(f"Input binding name '{self.input_name}' not found in the engine.")

        # For PyTorch compatibility
        self.names = self.class_names
        self._is_eval = True # TRT engines are always in "eval" mode

    def eval(self):
        """Sets the module in evaluation mode (no-op for TRT)."""
        self._is_eval = True

    def half(self):
        """
        Sets model to half precision (no-op for TRT as precision is fixed at build time).
        This is for API compatibility with PyTorch models.
        """
        engine_precision_is_half = any(ptype in self.engine_path.upper() for ptype in ["FP16", "HALF"])
        if not engine_precision_is_half:
            print(f"{colorstr('TensorRT Warning: ')}model.half() called, but TensorRT engine might not be FP16. "
                  "Ensure engine precision matches intended input data type.")
        pass # Precision is fixed at engine build time

    def __call__(self, img_batch_torch, augment=False):
        """
        Performs inference on a batch of images.

        Args:
            img_batch_torch (torch.Tensor): Input batch of images, preprocessed (e.g., NCHW, normalized).
            augment (bool): Augmentation flag (not used by TensorRT inference).

        Returns:
            tuple: (output_tensor, None)
                   output_tensor (torch.Tensor): The primary detection output from the engine.
                   None: Placeholder for train_out, as TRT engines don't produce it.
        """
        current_batch_size = img_batch_torch.shape[0]
        
        # Ensure input tensor is on CPU and is a C-contiguous NumPy array
        if img_batch_torch.is_cuda:
            img_batch_torch = img_batch_torch.cpu()
        img_batch_np = img_batch_torch.numpy()

        if not img_batch_np.flags['C_CONTIGUOUS']:
            img_batch_np = np.ascontiguousarray(img_batch_np)

        # --- Input Binding Handling ---
        current_input_shape_trt = (current_batch_size,) + self.input_shape_from_engine[1:] # (N, C, H, W)

        # If engine input shape is dynamic or current device input buffer is not suitable
        if -1 in self.input_shape_from_engine or \
           self.input_name not in self.device_inputs or \
           self.host_inputs[self.input_name].size != img_batch_np.size: # Check total size for safety

            # This is a simplified dynamic allocation. Robust handling uses optimization profiles.
            # Free existing device input if it exists
            if self.input_name in self.device_inputs:
                self.device_inputs[self.input_name].free()

            self.host_inputs[self.input_name] = np.ascontiguousarray(img_batch_np) # Use current batch directly
            self.device_inputs[self.input_name] = cuda.mem_alloc(self.host_inputs[self.input_name].nbytes)
            self.bindings[self.input_binding_idx] = int(self.device_inputs[self.input_name])
            
            # For dynamic shapes, must set binding shape in context
            if -1 in self.input_shape_from_engine:
                 self.context.set_binding_shape(self.input_binding_idx, current_input_shape_trt)
        else: # Static shape, copy data into existing pagelocked buffer
            np.copyto(self.host_inputs[self.input_name].reshape(current_input_shape_trt), img_batch_np)


        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.device_inputs[self.input_name], self.host_inputs[self.input_name], stream=None) # Use default stream

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=0)

        # Transfer predictions back from GPU
        for out_name in self.engine_output_names:
            if out_name in self.host_outputs and out_name in self.device_outputs:
                 cuda.memcpy_dtoh_async(self.host_outputs[out_name], self.device_outputs[out_name], stream=None)
            else:
                print(f"Warning: Output binding '{out_name}' not found in allocated buffers during DTOH copy.")


        # Synchronize the stream
        cuda.synchronize_stream(0)

        # --- Output Processing ---
        # The raw output from host_outputs needs to be reshaped.
        # This assumes the FIRST output name in `self.engine_output_names` is the main detection tensor.
        # This is a common convention but might need adjustment based on the specific engine.
        primary_output_name = self.engine_output_names[0]
        raw_output_np = self.host_outputs[primary_output_name]
        
        # Get the actual output shape after inference, especially for dynamic shapes
        actual_output_shape_from_context = self.context.get_binding_shape(self.engine.get_binding_index(primary_output_name))
        
        # If any dimension is -1 (dynamic), it means it's determined at runtime.
        # The volume of raw_output_np should correspond to this actual shape.
        # Example: (batch_size, num_predictions, num_params) like (1, 25200, 85)
        
        # Ensure the shape from context is fully defined (no -1s)
        if any(dim == -1 for dim in actual_output_shape_from_context):
            # Fallback: try to infer from raw_output_np size and other known dimensions
            # This part is tricky for fully dynamic outputs without more info.
            # For YOLO, typically batch_size is dynamic, num_predictions and num_params are fixed.
            num_predictions = self.output_shapes_from_engine[primary_output_name][1]
            num_params = self.output_shapes_from_engine[primary_output_name][2]
            expected_elements_per_item = num_predictions * num_params
            # Calculate effective batch size from total elements / elements_per_item
            # This assumes raw_output_np is trimmed to actual output data.
            if expected_elements_per_item > 0:
                effective_batch_size = raw_output_np.size // expected_elements_per_item
            else:
                effective_batch_size = 0
                
            if effective_batch_size != current_batch_size:
                 print(f"Warning: Mismatch in effective batch size ({effective_batch_size}) and input batch size ({current_batch_size}). Using input batch size for reshaping.")
                 effective_batch_size = current_batch_size # Trust input batch size

            reshaped_output_np = raw_output_np[:effective_batch_size * num_predictions * num_params].reshape(
                effective_batch_size, num_predictions, num_params
            )
        else:
            # Use the shape obtained from the context after execution
            reshaped_output_np = raw_output_np.reshape(actual_output_shape_from_context)
            # Slice if current_batch_size is less than the batch dim of actual_output_shape_from_context
            if reshaped_output_np.shape[0] > current_batch_size:
                reshaped_output_np = reshaped_output_np[:current_batch_size]


        out_tensor = torch.from_numpy(reshaped_output_np).to(self.device_str if torch.cuda.is_available() else 'cpu')
        
        # TRT engines typically don't have the auxiliary 'train_out'
        return out_tensor, None


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True, # For PyTorch model; for TRT, input prep based on engine
         trace=False,      # For PyTorch model tracing
         is_coco=False,    # Automatically set if data file ends with coco.yaml
         v5_metric=False,
         engine_path=None, # Path to TensorRT engine file
         model_stride=32,  # Max stride of the model (needed for TRT engine)
         trt_input_name='images', # Default TRT input node name
         trt_output_names=None # Comma-separated TRT output node names (e.g., 'output0,output1')
         ):
    """
    Main testing function. Can load PyTorch models or TensorRT engines.
    """
    training = model is not None # True if called by train.py

    # --- Device Setup ---
    if training:
        device = next(model.parameters()).device # Get device from PyTorch model
    else:
        set_logging()
        device = select_device(opt.device, batch_size=batch_size) # opt.device from argparse

    # --- Directories ---
    if not training:
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # --- Load Data Configuration ---
    if isinstance(data, str): # data is a path to YAML file
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    else: # data is already a dictionary
        data_cfg = data
        # Try to infer is_coco from dataset path if available in dict
        val_path_str = str(data_cfg.get('val', ''))
        is_coco = 'coco' in val_path_str.lower() and 'annotations' not in val_path_str.lower()


    nc = 1 if single_cls else int(data_cfg['nc'])
    class_names_from_data = data_cfg['names']

    # --- Model Loading (PyTorch or TensorRT) ---
    if not training: # If called directly for testing/validation
        if engine_path:
            print(f"{colorstr('TensorRT: ')}Loading engine {engine_path}...")
            if not TRT_AVAILABLE:
                raise RuntimeError("TensorRT or PyCUDA not found, but --engine was specified.")
            
            if trt_output_names:
                trt_engine_output_names = [name.strip() for name in trt_output_names.split(',')]
            else:
                print(f"{colorstr('TensorRT Warning: ')}--trt-output-names not specified. Attempting to infer from engine.")
                temp_logger = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f_eng, trt.Runtime(temp_logger) as temp_runtime:
                    temp_engine = temp_runtime.deserialize_cuda_engine(f_eng.read())
                trt_engine_output_names = [temp_engine.get_binding_name(i) for i in range(temp_engine.num_bindings) if not temp_engine.binding_is_input(i)]
                if not trt_engine_output_names:
                    raise ValueError("Could not automatically determine output names from TensorRT engine.")
                print(f"{colorstr('TensorRT: ')}Inferred output names: {trt_engine_output_names}")


            model = TRTModule(engine_path, str(device), trt_input_name, trt_engine_output_names, model_stride, class_names_from_data)
            gs = max(int(model.stride.max()), 32)
            imgsz = check_img_size(imgsz, s=gs)
            # Determine if input should be half precision for TRT engine
            half_input_for_trt = "FP16" in engine_path.upper() or "HALF" in engine_path.upper()
            half = half_input_for_trt
            if half:
                print(f"{colorstr('TensorRT: ')}Input data will be prepared as FP16 for the engine.")
            else:
                print(f"{colorstr('TensorRT: ')}Input data will be prepared as FP32 for the engine.")

        elif weights:
            model = attempt_load(weights, map_location=device)
            gs = max(int(model.stride.max()), 32)
            imgsz = check_img_size(imgsz, s=gs)
            if trace:
                model = TracedModel(model, device, imgsz)
            # `half` for PyTorch model
            half = device.type != 'cpu' and half_precision
            if half:
                model.half()
        else:
            raise ValueError("Must specify --weights (for PyTorch model) or --engine (for TensorRT model).")
        
        model.eval()

    # --- Dataloader Setup ---
    if not training:
        if device.type != 'cpu' and weights and not engine_path:
            dummy_input_type = torch.half if half else torch.float
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type(dummy_input_type)) # run once

        task = opt.task if hasattr(opt, 'task') and opt.task in ('train', 'val', 'test') else 'val'
        rect_val = opt.rect if hasattr(opt, 'rect') else True 
        dataloader = create_dataloader(data_cfg[task], imgsz, batch_size, gs, opt, pad=0.5, rect=rect_val,
                                       prefix=colorstr(f'{task}: '))[0]

    # --- Metrics Setup ---
    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else class_names_from_data
    coco91class = coco80_to_coco91_class()
    s_format = '%20s' + '%12s' * 6
    s_header = s_format % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p_metric, r_metric, f1_metric, mp_metric, mr_metric, map50_metric, map_metric, t0_inf, t1_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap_all, ap_class_indices, wandb_images = [], [], [], [], []
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    # --- Main Evaluation Loop ---
    for batch_i, (img_batch, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s_header)):
        img_batch = img_batch.to(device, non_blocking=True)
        img_processed = img_batch.half() if half else img_batch.float()
        img_processed /= 255.0
        
        targets = targets.to(device)
        nb, _, height, width = img_processed.shape

        with torch.no_grad():
            t_inf_start = time_synchronized()
            out_raw, train_out = model(img_processed, augment=augment)
            t0_inf += time_synchronized() - t_inf_start

            if compute_loss and train_out is not None:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]
            elif compute_loss and engine_path and batch_i == 0:
                print(f"{colorstr('TensorRT: ')}Skipping loss computation as 'train_out' is not available from engine.")

            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            
            t_nms_start = time_synchronized()
            if out_raw is None:
                 print(f"Warning: Raw output from model is None at batch {batch_i}. Skipping NMS.")
                 out_nms = [torch.empty(0,6).to(device)] * nb
            else:
                 out_nms = non_max_suppression(out_raw, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1_nms += time_synchronized() - t_nms_start

        for si, pred in enumerate(out_nms):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            scale_coords(img_processed[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                for *xyxy, conf, cls_id in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls_id, *xywh, conf) if save_conf else (cls_id, *xywh)
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_json:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])
                box[:, :2] -= box[:, 2:] / 2
                for p_item, b_item in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p_item[5])] if is_coco else int(p_item[5]),
                                  'bbox': [round(x, 3) for x in b_item],
                                  'score': round(p_item[4], 5)})

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img_processed[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                for cls_idx in torch.unique(tcls_tensor):
                    ti = (cls_idx == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls_idx == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if plots and batch_i < 3:
            f_labels = save_dir / f'test_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img_batch, targets, paths, f_labels, names), daemon=True).start()
            f_pred = save_dir / f'test_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img_batch, output_to_target(out_nms), paths, f_pred, names), daemon=True).start()

    # --- Compute Final Statistics ---
    stats = [np.concatenate(x, 0) for x in zip(*stats)] if len(stats) else []
    if len(stats) and stats[0].any():
        p_metric, r_metric, ap_all, f1_metric, ap_class_indices = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50_metric, ap_metric = ap_all[:, 0], ap_all.mean(1)
        mp_metric, mr_metric, map50_metric, map_metric = p_metric.mean(), r_metric.mean(), ap50_metric.mean(), ap_metric.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)
        mp_metric, mr_metric, map50_metric, map_metric = 0., 0., 0., 0.
        ap50_metric, ap_metric = np.zeros(nc), np.zeros(nc)
        ap_class_indices = list(range(nc))

    print(s_format % ('all', seen, nt.sum(), mp_metric, mr_metric, map50_metric, map_metric))

    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats) and stats[0].any():
        for i, c_idx in enumerate(ap_class_indices):
            print(s_format % (names[c_idx], seen, nt[c_idx], p_metric[i], r_metric[i], ap50_metric[i], ap_metric[i]))

    t_total_inf_ms = t0_inf / seen * 1E3 if seen > 0 else 0
    t_total_nms_ms = t1_nms / seen * 1E3 if seen > 0 else 0
    t_total_combined_ms = (t0_inf + t1_nms) / seen * 1E3 if seen > 0 else 0
    speed_stats = (t_total_inf_ms, t_total_nms_ms, t_total_combined_ms) + (imgsz, imgsz, batch_size)
    if not training:
        print('Speed: %.1fms inference, %.1fms NMS, %.1fms total per %gx%g image at batch-size %g' % speed_stats)

    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    if save_json and len(jdict):
        if weights and isinstance(weights, list) and weights[0]:
            w_stem = Path(weights[0]).stem
        elif engine_path:
            w_stem = Path(engine_path).stem
        else:
            w_stem = 'model'

        anno_path_str = str(data_cfg.get('val', '')).replace('images', 'annotations').replace('val2017.txt', 'instances_val2017.json')
        if not Path(anno_path_str).exists():
            base_path = Path(data_cfg.get('path', '.'))
            anno_path_str = str(base_path / 'annotations' / 'instances_val2017.json')
        
        pred_json_path = str(save_dir / f"{w_stem}_predictions.json")
        print(f'\nEvaluating pycocotools mAP... saving {pred_json_path}...')
        with open(pred_json_path, 'w') as f:
            json.dump(jdict, f)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_path_str)
            pred = anno.loadRes(pred_json_path)
            eval = COCOeval(anno, pred, 'bbox')
            
            if is_coco:
                if hasattr(dataloader.dataset, 'img_files'):
                     img_ids_to_eval = [int(Path(x).stem) for x in dataloader.dataset.img_files]
                     eval.params.imgIds = img_ids_to_eval
                elif 'val_img_ids' in data_cfg and Path(data_cfg['val_img_ids']).exists():
                    with open(data_cfg['val_img_ids'], 'r') as f_img_ids:
                        eval.params.imgIds = [int(line.strip()) for line in f_img_ids]

            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map_metric, map50_metric = eval.stats[:2]
        except Exception as e:
            print(f'{colorstr("pycocotools error: ")}pycocotools unable to run: {e}')
            print(f"Ensure annotation file path is correct: {anno_path_str}")

    if not isinstance(model, TRTModule) and hasattr(model, 'float'):
        model.float()

    if not training:
        s_save = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s_save}")
    
    maps_per_class = np.zeros(nc) + map_metric
    if len(stats) and stats[0].any() and len(ap_all) > 0 :
        for i, c_idx in enumerate(ap_class_indices):
            if i < len(ap_all):
                 maps_per_class[c_idx] = ap_all[i].mean()

    final_loss = (loss.cpu().numpy() / len(dataloader)).tolist() if len(dataloader) > 0 else [0,0,0]
    return (mp_metric, mr_metric, map50_metric, map_metric, *final_loss), maps_per_class, speed_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default=None, help='model.pt path(s) or model.engine path(s)')
    parser.add_argument('--engine', type=str, default=None, help='TensorRT .engine file path (takes priority over --weights if both provided)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.yaml path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model (PyTorch only)')
    parser.add_argument('--half-precision', action='store_true', help='use FP16 half-precision inference (PyTorch only, for TRT determined by engine)')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    # TensorRT specific arguments
    parser.add_argument('--model-stride', type=int, default=32, help='Max stride of the model (needed for TRT engine)')
    parser.add_argument('--trt-input-name', type=str, default='images', help='Input node name of the TRT engine')
    parser.add_argument('--trt-output-names', type=str, default=None, help="Comma-separated output node names of the TRT engine (e.g., 'output0,output1')")
    parser.add_argument('--rect', action='store_true', help='rectangular inference')

    opt = parser.parse_args()

    # --- Robust Argument Handling ---
    # This logic handles the case where user passes a .engine file via --weights
    if opt.weights:
        engine_files_in_weights = [w for w in opt.weights if Path(w).suffix.lower() == '.engine']
        if engine_files_in_weights:
            if len(engine_files_in_weights) > 1:
                print(f"{colorstr('Warning: ')}Multiple .engine files found in --weights. Using the first one: {engine_files_in_weights[0]}")

            # If an engine is already specified via --engine, give a warning and prioritize --engine argument.
            if opt.engine and opt.engine != engine_files_in_weights[0]:
                 print(f"{colorstr('Warning: ')}TensorRT engine specified via both --engine ({opt.engine}) and --weights ({engine_files_in_weights[0]}). "
                       f"Prioritizing the one from --engine argument: {opt.engine}")
            else:
                 # Move the engine file from weights to the engine argument
                 opt.engine = engine_files_in_weights[0]

            # Remove engine files from weights list
            opt.weights = [w for w in opt.weights if Path(w).suffix.lower() != '.engine']
            if not opt.weights: # If the list becomes empty
                opt.weights = None
    
    # --- Final Input Validation ---
    if not opt.weights and not opt.engine:
        raise ValueError("After parsing, no valid --weights (for PyTorch model) or --engine (for TensorRT model) was found.")
    
    if opt.weights and opt.engine:
        print(f"{colorstr('Warning: ')}Both a PyTorch model ({opt.weights}) and a TensorRT engine ({opt.engine}) are specified. "
              "The script will prioritize the TensorRT engine. The PyTorch model will be ignored.")
        opt.weights = None

    opt.save_json |= opt.data.endswith('coco.yaml') if opt.data else False
    opt.data = check_file(opt.data)
    print(opt)

    # --- Main Execution Logic ---
    if opt.task in ('train', 'val', 'test'):
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace and not opt.engine,
             half_precision=opt.half_precision,
             v5_metric=opt.v5_metric,
             engine_path=opt.engine,
             model_stride=opt.model_stride,
             trt_input_name=opt.trt_input_name,
             trt_output_names=opt.trt_output_names
             )

    elif opt.task == 'speed':
        model_paths = opt.weights if opt.weights else ([opt.engine] if opt.engine else [])
        for model_p in model_paths:
            current_weights = [model_p] if Path(model_p).suffix.lower() == '.pt' else None
            current_engine = model_p if Path(model_p).suffix.lower() == '.engine' else None
            if not current_weights and not current_engine: continue

            test(opt.data, current_weights, opt.batch_size, opt.img_size, 0.25, 0.45,
                 save_json=False, plots=False, v5_metric=opt.v5_metric,
                 engine_path=current_engine, model_stride=opt.model_stride,
                 trt_input_name=opt.trt_input_name, trt_output_names=opt.trt_output_names,
                 half_precision=opt.half_precision)

    elif opt.task == 'study':
        print(f"{colorstr('Warning: ')}'study' task requires an engine with dynamic image size support if a .engine file is used.")
        
        x_img_sizes = list(range(256, 1536 + 128, 128))
        model_paths_for_study = opt.weights if opt.weights else ([opt.engine] if opt.engine else [])

        for model_p_study in model_paths_for_study:
            current_weights_study = [model_p_study] if Path(model_p_study).suffix.lower() == '.pt' else None
            current_engine_study = model_p_study if Path(model_p_study).suffix.lower() == '.engine' else None
            if not current_weights_study and not current_engine_study: continue

            study_filename_stem = Path(opt.data).stem + '_' + Path(model_p_study).stem
            study_results_file = f'study_{study_filename_stem}.txt'
            
            y_results = []
            for current_imgsz in x_img_sizes:
                print(f'\nRunning {study_results_file} point {current_imgsz}...')
                results, _, speed = test(opt.data, current_weights_study, opt.batch_size, current_imgsz,
                                         opt.conf_thres, opt.iou_thres, opt.save_json,
                                         plots=False, v5_metric=opt.v5_metric,
                                         engine_path=current_engine_study, model_stride=opt.model_stride,
                                         trt_input_name=opt.trt_input_name, trt_output_names=opt.trt_output_names,
                                         half_precision=opt.half_precision)
                y_results.append(results + speed)
            np.savetxt(study_results_file, y_results, fmt='%10.4g')
        
        if model_paths_for_study:
            os.system('zip -r study.zip study_*.txt')
            plot_study_txt(x=x_img_sizes)

from typing import Tuple, Union
from mindspore import nn, ops
import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore import Tensor
import mindspore.numpy as mnp

__all__ = ['L1BalancedCELoss', 'PSEDiceLoss', 'EASTLoss', 'DRRGLoss']


class L1BalancedCELoss(nn.LossBase):
    """
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    """

    def __init__(self, eps=1e-6, bce_scale=5, l1_scale=10, bce_replace="bceloss"):
        super().__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()

        if bce_replace == "bceloss":
            self.bce_loss = BalancedBCELoss()
        elif bce_replace == "diceloss":
            self.bce_loss = DiceLoss()
        else:
            raise ValueError(f"bce_replace should be in ['bceloss', 'diceloss'], but get {bce_replace}")

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(self, pred: Union[Tensor, Tuple[Tensor]], gt: Tensor, gt_mask: Tensor, thresh_map: Tensor,
                  thresh_mask: Tensor):
        """
        Compute dbnet loss
        Args:
            pred (Tuple[Tensor]): network prediction consists of
                binary: The text segmentation prediction.
                thresh: The threshold prediction (optional)
                thresh_binary: Value produced by `step_function(binary - thresh)`. (optional)
            gt (Tensor): Text regions bitmap gt.
            mask (Tensor): Ignore mask, pexels where value is 1 indicates no contribution to loss.
            thresh_mask (Tensor): Mask indicates regions cared by thresh supervision.
            thresh_map (Tensor): Threshold gt.
        Return:
            loss value (Tensor)
        """
        if isinstance(pred, ms.Tensor):
            loss = self.bce_loss(pred, gt, gt_mask)
        else:
            binary, thresh, thresh_binary = pred
            bce_loss_output = self.bce_loss(binary, gt, gt_mask)
            l1_loss = self.l1_loss(thresh, thresh_map, thresh_mask)
            dice_loss = self.dice_loss(thresh_binary, gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + self.bce_scale * bce_loss_output

        '''
        if isinstance(pred, tuple):
            binary, thresh, thresh_binary = pred
        else:
            binary = pred

        bce_loss_output = self.bce_loss(binary, gt, gt_mask)

        if isinstance(pred, tuple):
            l1_loss = self.l1_loss(thresh, thresh_map, thresh_mask)
            dice_loss = self.dice_loss(thresh_binary, gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + self.bce_scale * bce_loss_output
        else:
            loss = bce_loss_output
        '''
        return loss


class DiceLoss(nn.LossBase):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred, gt, mask):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
              the losses of two heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        pred = pred.squeeze(axis=1) * mask
        gt = gt.squeeze(axis=1) * mask

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() + self._eps
        return 1 - 2.0 * intersection / union


class MaskL1Loss(nn.LossBase):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred, gt, mask):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        pred = pred.squeeze(axis=1)
        return ((pred - gt).abs() * mask).sum() / (mask.sum() + self._eps)


class BalancedBCELoss(nn.LossBase):
    """Balanced cross entropy loss."""

    def __init__(self, negative_ratio=3, eps=1e-6):
        super().__init__()
        self._negative_ratio = negative_ratio
        self._eps = eps
        self._bce_loss = ops.BinaryCrossEntropy(reduction='none')

    def construct(self, pred, gt, mask):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        pred = pred.squeeze(axis=1)
        gt = gt.squeeze(axis=1)

        positive = gt * mask
        negative = (1 - gt) * mask

        pos_count = positive.sum(axis=(1, 2), keepdims=True).astype(ms.int32)
        neg_count = negative.sum(axis=(1, 2), keepdims=True).astype(ms.int32)
        neg_count = ops.minimum(neg_count, pos_count * self._negative_ratio).squeeze(axis=(1, 2))

        loss = self._bce_loss(pred, gt, None)

        pos_loss = loss * positive
        neg_loss = (loss * negative).view(loss.shape[0], -1)

        neg_vals, _ = ops.sort(neg_loss)
        neg_index = ops.stack((mnp.arange(loss.shape[0]), neg_vals.shape[1] - neg_count), axis=1)
        min_neg_score = ops.expand_dims(ops.gather_nd(neg_vals, neg_index), axis=1)

        neg_loss_mask = (neg_loss >= min_neg_score).astype(ms.float32)  # filter values less than top k
        neg_loss_mask = ops.stop_gradient(neg_loss_mask)

        neg_loss = neg_loss_mask * neg_loss

        return (pos_loss.sum() + neg_loss.sum()) / \
            (pos_count.astype(ms.float32).sum() + neg_count.astype(ms.float32).sum() + self._eps)


class PSEDiceLoss(nn.Cell):
    def __init__(self, alpha=0.7, ohem_ratio=3):
        super().__init__()
        self.threshold0 = Tensor(0.5, mstype.float32)
        self.zero_float32 = Tensor(0.0, mstype.float32)
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.negative_one_int32 = Tensor(-1, mstype.int32)
        self.concat = ops.Concat()
        self.less_equal = ops.LessEqual()
        self.greater = ops.Greater()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dims = ops.ReduceSum(keep_dims=True)
        self.reduce_mean = ops.ReduceMean()
        self.reduce_min = ops.ReduceMin()
        self.cast = ops.Cast()
        self.minimum = ops.Minimum()
        self.expand_dims = ops.ExpandDims()
        self.select = ops.Select()
        self.fill = ops.Fill()
        self.topk = ops.TopK(sorted=True)
        self.shape = ops.Shape()
        self.sigmoid = ops.Sigmoid()
        self.reshape = ops.Reshape()
        self.slice = ops.Slice()
        self.logical_and = ops.LogicalAnd()
        self.logical_or = ops.LogicalOr()
        self.equal = ops.Equal()
        self.zeros_like = ops.ZerosLike()
        self.add = ops.Add()
        self.gather = ops.Gather()
        self.upsample = nn.ResizeBilinear()

    def ohem_batch(self, scores, gt_texts, training_masks):
        '''

        :param scores: [N * H * W]
        :param gt_texts:  [N * H * W]
        :param training_masks: [N * H * W]
        :return: [N * H * W]
        '''
        batch_size = scores.shape[0]
        h, w = scores.shape[1:]
        selected_masks = ()
        for i in range(batch_size):
            score = self.slice(scores, (i, 0, 0), (1, h, w))
            score = self.reshape(score, (h, w))

            gt_text = self.slice(gt_texts, (i, 0, 0), (1, h, w))
            gt_text = self.reshape(gt_text, (h, w))

            training_mask = self.slice(training_masks, (i, 0, 0), (1, h, w))
            training_mask = self.reshape(training_mask, (h, w))

            selected_mask = self.ohem_single(score, gt_text, training_mask)
            selected_masks = selected_masks + (selected_mask,)

        selected_masks = self.concat(selected_masks)
        return selected_masks

    def ohem_single(self, score, gt_text, training_mask):
        h, w = score.shape[0:2]
        k = int(h * w)
        pos_num = self.logical_and(self.greater(gt_text, self.threshold0),
                                   self.greater(training_mask, self.threshold0))
        pos_num = self.reduce_sum(self.cast(pos_num, mstype.float32))

        neg_num = self.less_equal(gt_text, self.threshold0)
        neg_num = self.reduce_sum(self.cast(neg_num, mstype.float32))
        neg_num = self.minimum(self.ohem_ratio * pos_num, neg_num)
        neg_num = self.cast(neg_num, mstype.int32)

        neg_num = neg_num + k - 1
        neg_mask = self.less_equal(gt_text, self.threshold0)
        ignore_score = self.fill(mstype.float32, (h, w), -1e3)
        neg_score = self.select(neg_mask, score, ignore_score)
        neg_score = self.reshape(neg_score, (h * w,))

        topk_values, _ = self.topk(neg_score, k)
        threshold = self.gather(topk_values, neg_num, 0)

        selected_mask = self.logical_and(
            self.logical_or(self.greater(score, threshold),
                            self.greater(gt_text, self.threshold0)),
            self.greater(training_mask, self.threshold0))

        selected_mask = self.cast(selected_mask, mstype.float32)
        selected_mask = self.expand_dims(selected_mask, 0)

        return selected_mask

    def dice_loss(self, input_params, target, mask):
        '''

        :param input: [N, H, W]
        :param target: [N, H, W]
        :param mask: [N, H, W]
        :return:
        '''
        batch_size = input_params.shape[0]
        input_sigmoid = self.sigmoid(input_params)

        input_reshape = self.reshape(input_sigmoid, (batch_size, -1))
        target = self.reshape(target, (batch_size, -1))
        mask = self.reshape(mask, (batch_size, -1))

        input_mask = input_reshape * mask
        target = target * mask

        a = self.reduce_sum(input_mask * target, 1)
        b = self.reduce_sum(input_mask * input_mask, 1) + 0.001
        c = self.reduce_sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        dice_loss = self.reduce_mean(d)
        return 1 - dice_loss

    def avg_losses(self, loss_list):
        loss_kernel = loss_list[0]
        for i in range(1, len(loss_list)):
            loss_kernel += loss_list[i]
        loss_kernel = loss_kernel / len(loss_list)
        return loss_kernel

    def construct(self, model_predict, gt_texts, gt_kernels, training_masks):
        '''

        :param model_predict: [N * 7 * H * W]
        :param gt_texts: [N * H * W]
        :param gt_kernels:[N * 6 * H * W]
        :param training_masks:[N * H * W]
        :return:
        '''
        batch_size = model_predict.shape[0]
        model_predict = self.upsample(model_predict, scale_factor=4)
        h, w = model_predict.shape[2:]
        texts = self.slice(model_predict, (0, 0, 0, 0), (batch_size, 1, h, w))
        texts = self.reshape(texts, (batch_size, h, w))
        selected_masks_text = self.ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.dice_loss(texts, gt_texts, selected_masks_text)
        kernels = []
        loss_kernels = []
        for i in range(1, 7):
            kernel = self.slice(model_predict, (0, i, 0, 0), (batch_size, 1, h, w))
            kernel = self.reshape(kernel, (batch_size, h, w))
            kernels.append(kernel)

        mask0 = self.sigmoid(texts)
        selected_masks_kernels = self.logical_and(self.greater(mask0, self.threshold0),
                                                  self.greater(training_masks, self.threshold0))
        selected_masks_kernels = self.cast(selected_masks_kernels, mstype.float32)

        for i in range(6):
            gt_kernel = self.slice(gt_kernels, (0, i, 0, 0), (batch_size, 1, h, w))
            gt_kernel = self.reshape(gt_kernel, (batch_size, h, w))
            loss_kernel_i = self.dice_loss(kernels[i], gt_kernel, selected_masks_kernels)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = self.avg_losses(loss_kernels)

        loss = self.alpha * loss_text + (1 - self.alpha) * loss_kernel
        return loss


class DiceCoefficient(nn.Cell):
    def __init__(self):
        super(DiceCoefficient, self).__init__()
        self.sum = ops.ReduceSum()
        self.eps = 1e-5

    def construct(self, true_cls, pred_cls):
        intersection = self.sum(true_cls * pred_cls, ())
        union = self.sum(true_cls, ()) + self.sum(pred_cls, ()) + self.eps
        loss = 1. - (2 * intersection / union)

        return loss


class EASTLoss(nn.LossBase):
    def __init__(self):
        super(EASTLoss, self).__init__()
        self.split = ops.Split(1, 5)
        self.log = ops.Log()
        self.cos = ops.Cos()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sum = ops.ReduceSum()
        self.eps = 1e-5
        self.dice = DiceCoefficient()
        self.abs = ops.Abs()

    def construct(
            self,
            pred,
            score_map,
            geo_map,
            training_mask):
        ans = self.sum(score_map)
        classification_loss = self.dice(
            score_map, pred['score'] * (1 - training_mask))

        # n * 5 * h * w
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = self.split(geo_map)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = self.split(pred['geo'])
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = self._min(d2_gt, d2_pred) + self._min(d4_gt, d4_pred)
        h_union = self._min(d1_gt, d1_pred) + self._min(d3_gt, d3_pred)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_loss_map = -self.log((area_intersect + 1.0) /
                                 (area_union + 1.0))  # iou_loss_map
        angle_loss_map = 1 - self.cos(theta_pred - theta_gt)  # angle_loss_map

        angle_loss = self.sum(angle_loss_map * score_map) / ans
        iou_loss = self.sum(iou_loss_map * score_map) / ans
        geo_loss = 10 * angle_loss + iou_loss

        return geo_loss + classification_loss

    def _min(self, a, b):
        return (a + b - self.abs(a - b)) / 2


class DRRGLoss(nn.LossBase):
    def __init__(self, ohem_ratio=3.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.downsample_ratio = 1.0

    def balance_bce_loss(self, pred, gt, mask):
        """Balanced Binary-CrossEntropy Loss.

        Args:
            pred (Tensor): Shape of :math:`(1, H, W)`.
            gt (Tensor): Shape of :math:`(1, H, W)`.
            mask (Tensor): Shape of :math:`(1, H, W)`.

        Returns:
            Tensor: Balanced bce loss.
        """
        assert pred.shape == gt.shape == mask.shape
        assert ops.all(pred >= 0) and ops.all(pred <= 1)
        assert ops.all(gt >= 0) and ops.all(gt <= 1)
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.sum())

        if positive_count > 0:
            loss = ops.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = ops.sum(loss * positive)
            negative_loss = loss * negative
            negative_count = min(
                int(negative.sum()), int(positive_count * self.ohem_ratio))
        else:
            positive_loss = ms.Tensor(0.0)
            loss = ops.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative
            negative_count = 100
        negative_loss, _ = ops.topk(
            negative_loss.reshape([-1]), negative_count)

        balance_loss = (positive_loss + ops.sum(negative_loss)) / (
                float(positive_count + negative_count) + 1e-5)

        return balance_loss

    def gcn_loss(self, gcn_data):
        """CrossEntropy Loss from gcn module.

        Args:
            gcn_data (tuple(Tensor, Tensor)): The first is the
                prediction with shape :math:`(N, 2)` and the
                second is the gt label with shape :math:`(m, n)`
                where :math:`m * n = N`.

        Returns:
            Tensor: CrossEntropy loss.
        """
        gcn_pred, gt_labels = gcn_data
        gt_labels = gt_labels.reshape([-1])
        loss = ops.cross_entropy(gcn_pred, gt_labels)

        return loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        batch_size = len(bitmasks)
        results = []

        kernel = []
        for batch_inx in range(batch_size):
            mask = bitmasks[batch_inx]
            # hxw
            mask_sz = mask.shape
            # left, right, top, bottom
            pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
            mask = ops.pad(mask, pad, mode='constant', value=0)
            kernel.append(mask)
        kernel = ops.stack(kernel)
        results.append(kernel)

        return results

    def construct(self, preds, gt_text_mask, gt_center_region_mask, gt_mask, gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map, gt_comp_attribs):
        """Compute Drrg loss.
        """

        # assert isinstance(preds, tuple)
        # gt_text_mask, gt_center_region_mask, gt_mask, gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map = labels[
        #                                                                                                              1:8]

        downsample_ratio = self.downsample_ratio

        pred_maps, gcn_data = preds
        # pred_maps = preds
        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_top_height_map = pred_maps[:, 4, :, :]
        pred_bot_height_map = pred_maps[:, 5, :, :]
        feature_sz = pred_maps.shape

        # bitmask 2 tensor
        mapping = {
            'gt_text_mask': ops.cast(gt_text_mask, ms.float32),
            'gt_center_region_mask':
                ops.cast(gt_center_region_mask, ms.float32),
            'gt_mask': ops.cast(gt_mask, ms.float32),
            'gt_top_height_map': ops.cast(gt_top_height_map, ms.float32),
            'gt_bot_height_map': ops.cast(gt_bot_height_map, ms.float32),
            'gt_sin_map': ops.cast(gt_sin_map, ms.float32),
            'gt_cos_map': ops.cast(gt_cos_map, ms.float32)
        }
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(downsample_ratio - 1.0) < 1e-2:
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key in ['gt_top_height_map', 'gt_bot_height_map']:
                    gt[key] = [item * downsample_ratio for item in gt[key]]
            gt[key] = [item for item in gt[key]]

        scale = ops.sqrt(1.0 / (pred_sin_map ** 2 + pred_cos_map ** 2 + 1e-8))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        loss_text = self.balance_bce_loss(
            ops.sigmoid(pred_text_region), gt['gt_text_mask'][0],
            gt['gt_mask'][0])

        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0])
        negative_text_mask = ((1 - gt['gt_text_mask'][0]) * gt['gt_mask'][0])
        loss_center_map = ops.binary_cross_entropy(
            ops.sigmoid(pred_center_region),
            gt['gt_center_region_mask'][0],
            reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center_positive = ops.sum(loss_center_map *
                                              text_mask) / ops.sum(text_mask)
        else:
            loss_center_positive = ms.Tensor(0.0)
        loss_center_negative = ops.sum(
            loss_center_map *
            negative_text_mask) / ops.sum(negative_text_mask)
        loss_center = loss_center_positive + 0.5 * loss_center_negative

        center_mask = (gt['gt_center_region_mask'][0] * gt['gt_mask'][0])
        if int(center_mask.sum()) > 0:
            map_sz = pred_top_height_map.shape
            ones = ops.ones(map_sz, dtype=ms.float32)
            loss_top = ops.smooth_l1_loss(
                pred_top_height_map / (gt['gt_top_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            loss_bot = ops.smooth_l1_loss(
                pred_bot_height_map / (gt['gt_bot_height_map'][0] + 1e-2),
                ones,
                reduction='none')
            gt_height = (
                    gt['gt_top_height_map'][0] + gt['gt_bot_height_map'][0])
            loss_height = ops.sum(
                (ops.log(gt_height + 1) *
                 (loss_top + loss_bot)) * center_mask) / ops.sum(center_mask)

            loss_sin = ops.sum(
                ops.smooth_l1_loss(
                    pred_sin_map, gt['gt_sin_map'][0],
                    reduction='none') * center_mask) / ops.sum(center_mask)
            loss_cos = ops.sum(
                ops.smooth_l1_loss(
                    pred_cos_map, gt['gt_cos_map'][0],
                    reduction='none') * center_mask) / ops.sum(center_mask)
        else:
            loss_height = ms.Tensor(0.0)
            loss_sin = ms.Tensor(0.0)
            loss_cos = ms.Tensor(0.0)

        loss_gcn = self.gcn_loss(gcn_data)

        loss = loss_text + loss_center + loss_height + loss_sin + loss_cos + loss_gcn
        # loss = loss_text + loss_center + loss_height + loss_sin + loss_cos
        print('total loss', loss)
        print('loss_text', loss_text)
        print('loss_center', loss_center)
        print('loss_height', loss_height)
        print('loss_sin', loss_sin)
        print('loss_cos', loss_cos)
        print('loss_gcn', loss_gcn)
        # results = dict(
        #     loss=loss,
        #     loss_text=loss_text,
        #     loss_center=loss_center,
        #     loss_height=loss_height,
        #     loss_sin=loss_sin,
        #     loss_cos=loss_cos,
        #     loss_gcn=loss_gcn)
        # results = (loss, loss_text, loss_center, loss_height, loss_sin, loss_cos, loss_gcn)

        return loss

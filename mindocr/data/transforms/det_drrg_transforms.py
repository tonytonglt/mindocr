import random
import math
import cv2
import numpy as np
from mindspore.dataset.vision import RandomColorAdjust
from shapely.geometry import Polygon
from lanms import merge_quadrangle_n9 as la_nms
from numpy.linalg import norm

__all__ = ['ColorJitter', 'RandomScaling', 'RandomCropFlip', 'RandomCropPolyInstances', 'RandomRotatePolyInstances',
           'SquareResizePad', 'DRRGTargets']


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5, **kwargs):
        self.p = p
        self.transforms = RandomColorAdjust(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, data):
        img = data['image']
        if random.random() < self.p:
            img = self.transforms(img)
            data['image'] = img
            return data
        else:
            return data


class RandomScaling:
    def __init__(self, size=800, scale=(3. / 4, 5. / 2), **kwargs):
        """Random scale the image while keeping aspect.

        Args:
            size (int) : Base size before scaling.
            scale (tuple(float)) : The range of scaling.
        """
        assert isinstance(size, int)
        assert isinstance(scale, float) or isinstance(scale, tuple)
        self.size = size
        self.scale = scale if isinstance(scale, tuple) \
            else (1 - scale, 1 + scale)

    def __call__(self, data):
        image = data['image']
        text_polys = data['polys']
        h, w, _ = image.shape

        aspect_ratio = np.random.uniform(min(self.scale), max(self.scale))
        scales = self.size * 1.0 / max(h, w) * aspect_ratio
        scales = np.array([scales, scales])
        out_size = (int(h * scales[1]), int(w * scales[0]))
        image = cv2.resize(image, out_size[::-1])

        data['image'] = image
        text_polys[:, :, 0::2] = text_polys[:, :, 0::2] * scales[1]
        text_polys[:, :, 1::2] = text_polys[:, :, 1::2] * scales[0]
        data['polys'] = text_polys

        return data


class RandomCropFlip:
    def __init__(self,
                 pad_ratio=0.1,
                 crop_ratio=0.5,
                 iter_num=1,
                 min_area_ratio=0.2,
                 **kwargs):
        """Random crop and flip a patch of the image.

        Args:
            crop_ratio (float): The ratio of cropping.
            iter_num (int): Number of operations.
            min_area_ratio (float): Minimal area ratio between cropped patch
                and original image.
        """
        assert isinstance(crop_ratio, float)
        assert isinstance(iter_num, int)
        assert isinstance(min_area_ratio, float)

        self.pad_ratio = pad_ratio
        self.epsilon = 1e-2
        self.crop_ratio = crop_ratio
        self.iter_num = iter_num
        self.min_area_ratio = min_area_ratio

    def __call__(self, results):
        for i in range(self.iter_num):
            results = self.random_crop_flip(results)

        return results

    def poly_intersection(self, poly_det, poly_gt, buffer=0.0001):
        """Calculate the intersection area between two polygon.

        Args:
            poly_det (Polygon): A polygon predicted by detector.
            poly_gt (Polygon): A gt polygon.

        Returns:
            intersection_area (float): The intersection area between two polygons.
        """
        assert isinstance(poly_det, Polygon)
        assert isinstance(poly_gt, Polygon)

        if buffer == 0:
            poly_inter = poly_det & poly_gt
        else:
            poly_inter = poly_det.buffer(buffer) & poly_gt.buffer(buffer)
        return poly_inter.area, poly_inter

    def random_crop_flip(self, results):
        image = results['image']
        polygons = results['polys']
        ignore_tags = results['ignore_tags']
        if len(polygons) == 0:
            return results

        if np.random.random() >= self.crop_ratio:
            return results

        h, w, _ = image.shape
        area = h * w
        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        h_axis, w_axis = self.generate_crop_target(image, polygons, pad_h,
                                                   pad_w)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return results

        attempt = 0
        while attempt < 50:
            attempt += 1
            polys_keep = []
            polys_new = []
            ignore_tags_keep = []
            ignore_tags_new = []
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area * self.min_area_ratio:
                # area too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin],
                            [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = Polygon(pts)
            fail_flag = False
            for polygon, ignore_tag in zip(polygons, ignore_tags):
                ppi = Polygon(polygon.reshape(-1, 2))
                ppiou, _ = self.poly_intersection(ppi, pp, buffer=0)
                if np.abs(ppiou - float(ppi.area)) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area)) < self.epsilon:
                    polys_new.append(polygon)
                    ignore_tags_new.append(ignore_tag)
                else:
                    polys_keep.append(polygon)
                    ignore_tags_keep.append(ignore_tag)

            if fail_flag:
                continue
            else:
                break

        cropped = image[ymin:ymax, xmin:xmax, :]
        select_type = np.random.randint(3)
        if select_type == 0:
            img = np.ascontiguousarray(cropped[:, ::-1])
        elif select_type == 1:
            img = np.ascontiguousarray(cropped[::-1, :])
        else:
            img = np.ascontiguousarray(cropped[::-1, ::-1])
        image[ymin:ymax, xmin:xmax, :] = img
        results['img'] = image

        if len(polys_new) != 0:
            height, width, _ = cropped.shape
            if select_type == 0:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    polys_new[idx] = poly
            elif select_type == 1:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly
            else:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly
            polygons = polys_keep + polys_new
            ignore_tags = ignore_tags_keep + ignore_tags_new
            results['polys'] = np.array(polygons)
            results['ignore_tags'] = ignore_tags

        return results

    def generate_crop_target(self, image, all_polys, pad_h, pad_w):
        """Generate crop target and make sure not to crop the polygon
        instances.

        Args:
            image (ndarray): The image waited to be crop.
            all_polys (list[list[ndarray]]): All polygons including ground
                truth polygons and ground truth ignored polygons.
            pad_h (int): Padding length of height.
            pad_w (int): Padding length of width.
        Returns:
            h_axis (ndarray): Vertical cropping range.
            w_axis (ndarray): Horizontal cropping range.
        """
        h, w, _ = image.shape
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

        text_polys = []
        for polygon in all_polys:
            rect = cv2.minAreaRect(polygon.astype(np.int32).reshape(-1, 2))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            text_polys.append([box[0], box[1], box[2], box[3]])

        polys = np.array(text_polys, dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        return h_axis, w_axis


class RandomCropPolyInstances:
    """Randomly crop images and make sure to contain at least one intact
    instance."""

    def __init__(self, crop_ratio=5.0 / 8.0, min_side_ratio=0.4, **kwargs):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.min_side_ratio = min_side_ratio

    def sample_valid_start_end(self, valid_array, min_len, max_start, min_end):

        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        start_array = valid_array.copy()
        max_start = min(len(start_array) - min_len, max_start)
        start_array[max_start:] = 0
        start_array[0] = 1
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        end_array = valid_array.copy()
        min_end = max(start + min_len, min_end)
        end_array[:min_end] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        return start, end

    def sample_crop_box(self, img_size, results):
        """Generate crop box and make sure not to crop the polygon instances.

        Args:
            img_size (tuple(int)): The image size (h, w).
            results (dict): The results dict.
        """

        assert isinstance(img_size, tuple)
        h, w = img_size[:2]

        key_masks = results['polys']

        x_valid_array = np.ones(w, dtype=np.int32)
        y_valid_array = np.ones(h, dtype=np.int32)

        selected_mask = key_masks[np.random.randint(0, len(key_masks))]
        selected_mask = selected_mask.reshape((-1, 2)).astype(np.int32)
        max_x_start = max(np.min(selected_mask[:, 0]) - 2, 0)
        min_x_end = min(np.max(selected_mask[:, 0]) + 3, w - 1)
        max_y_start = max(np.min(selected_mask[:, 1]) - 2, 0)
        min_y_end = min(np.max(selected_mask[:, 1]) + 3, h - 1)

        for mask in key_masks:
            mask = mask.reshape((-1, 2)).astype(np.int32)
            clip_x = np.clip(mask[:, 0], 0, w - 1)
            clip_y = np.clip(mask[:, 1], 0, h - 1)
            min_x, max_x = np.min(clip_x), np.max(clip_x)
            min_y, max_y = np.min(clip_y), np.max(clip_y)

            x_valid_array[min_x - 2:max_x + 3] = 0
            y_valid_array[min_y - 2:max_y + 3] = 0

        min_w = int(w * self.min_side_ratio)
        min_h = int(h * self.min_side_ratio)

        x1, x2 = self.sample_valid_start_end(x_valid_array, min_w, max_x_start,
                                             min_x_end)
        y1, y2 = self.sample_valid_start_end(y_valid_array, min_h, max_y_start,
                                             min_y_end)

        return np.array([x1, y1, x2, y2])

    def crop_img(self, img, bbox):
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __call__(self, results):
        image = results['image']
        polygons = results['polys']
        ignore_tags = results['ignore_tags']
        if len(polygons) < 1:
            return results

        if np.random.random_sample() < self.crop_ratio:

            crop_box = self.sample_crop_box(image.shape, results)
            img = self.crop_img(image, crop_box)
            results['image'] = img
            # crop and filter masks
            x1, y1, x2, y2 = crop_box
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            polygons[:, :, 0::2] = polygons[:, :, 0::2] - x1
            polygons[:, :, 1::2] = polygons[:, :, 1::2] - y1

            valid_masks_list = []
            valid_tags_list = []
            for ind, polygon in enumerate(polygons):
                if (polygon[:, ::2] > -4).all() and (
                        polygon[:, ::2] < w + 4).all() and (
                            polygon[:, 1::2] > -4).all() and (
                                polygon[:, 1::2] < h + 4).all():
                    polygon[:, ::2] = np.clip(polygon[:, ::2], 0, w)
                    polygon[:, 1::2] = np.clip(polygon[:, 1::2], 0, h)
                    valid_masks_list.append(polygon)
                    valid_tags_list.append(ignore_tags[ind])

            results['polys'] = np.array(valid_masks_list)
            results['ignore_tags'] = valid_tags_list

        return results


class RandomRotatePolyInstances:
    def __init__(self,
                 rotate_ratio=0.5,
                 max_angle=10,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Randomly rotate images and polygon masks.

        Args:
            rotate_ratio (float): The ratio of samples to operate rotation.
            max_angle (int): The maximum rotation angle.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rotated image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def rotate(self, center, points, theta, center_shift=(0, 0)):
        # rotate points.
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[:, ::2], points[:, 1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[:, ::2], points[:, 1::2] = _x, _y
        return points

    def cal_canvas_size(self, ori_size, degree):
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    def sample_angle(self, max_angle):
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    def rotate_img(self, img, angle, canvas_size):
        h, w = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
        rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)

        if self.pad_with_fixed_color:
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=self.pad_value)
        else:
            mask = np.zeros_like(img)
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            img_cut = cv2.resize(img_cut, (canvas_size[1], canvas_size[0]))

            mask = cv2.warpAffine(
                mask,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[1, 1, 1])
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[0, 0, 0])
            target_img = target_img + img_cut * mask

        return target_img

    def __call__(self, results):
        if np.random.random_sample() < self.rotate_ratio:
            image = results['image']
            polygons = results['polys']
            h, w = image.shape[:2]

            angle = self.sample_angle(self.max_angle)
            canvas_size = self.cal_canvas_size((h, w), angle)
            center_shift = (int((canvas_size[1] - w) / 2), int(
                (canvas_size[0] - h) / 2))
            image = self.rotate_img(image, angle, canvas_size)
            results['image'] = image
            # rotate polygons
            rotated_masks = []
            for mask in polygons:
                rotated_mask = self.rotate((w / 2, h / 2), mask, angle,
                                           center_shift)
                rotated_masks.append(rotated_mask)
            results['polys'] = np.array(rotated_masks)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class SquareResizePad:
    def __init__(self,
                 target_size,
                 pad_ratio=0.6,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Resize or pad images to be square shape.

        Args:
            target_size (int): The target size of square shaped image.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rescales image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        assert isinstance(target_size, int)
        assert isinstance(pad_ratio, float)
        assert isinstance(pad_with_fixed_color, bool)
        assert isinstance(pad_value, tuple)

        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def resize_img(self, img, keep_ratio=True):
        h, w, _ = img.shape
        if keep_ratio:
            t_h = self.target_size if h >= w else int(h * self.target_size / w)
            t_w = self.target_size if h <= w else int(w * self.target_size / h)
        else:
            t_h = t_w = self.target_size
        img = cv2.resize(img, (t_w, t_h))
        return img, (t_h, t_w)

    def square_pad(self, img):
        h, w = img.shape[:2]
        if h == w:
            return img, (0, 0)
        pad_size = max(h, w)
        if self.pad_with_fixed_color:
            expand_img = np.ones((pad_size, pad_size, 3), dtype=np.uint8)
            expand_img[:] = self.pad_value
        else:
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            expand_img = cv2.resize(img_cut, (pad_size, pad_size))
        if h > w:
            y0, x0 = 0, (h - w) // 2
        else:
            y0, x0 = (w - h) // 2, 0
        expand_img[y0:y0 + h, x0:x0 + w] = img
        offset = (x0, y0)

        return expand_img, offset

    def square_pad_mask(self, points, offset):
        x0, y0 = offset
        pad_points = points.copy()
        pad_points[::2] = pad_points[::2] + x0
        pad_points[1::2] = pad_points[1::2] + y0
        return pad_points

    def __call__(self, results):
        image = results['image']
        polygons = results['polys']
        h, w = image.shape[:2]

        if np.random.random_sample() < self.pad_ratio:
            image, out_size = self.resize_img(image, keep_ratio=True)
            image, offset = self.square_pad(image)
        else:
            image, out_size = self.resize_img(image, keep_ratio=False)
            offset = (0, 0)
        results['image'] = image
        try:
            polygons[:, :, 0::2] = polygons[:, :, 0::2] * out_size[
                1] / w + offset[0]
            polygons[:, :, 1::2] = polygons[:, :, 1::2] * out_size[
                0] / h + offset[1]
        except:
            pass
        results['polys'] = polygons

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class DRRGTargets(object):
    def __init__(self,
                 orientation_thr=2.0,
                 resample_step=8.0,
                 num_min_comps=9,
                 num_max_comps=600,
                 min_width=8.0,
                 max_width=24.0,
                 center_region_shrink_ratio=0.3,
                 comp_shrink_ratio=1.0,
                 comp_w_h_ratio=0.3,
                 text_comp_nms_thr=0.25,
                 min_rand_half_height=8.0,
                 max_rand_half_height=24.0,
                 jitter_level=0.2,
                 **kwargs):

        super().__init__()
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.num_max_comps = num_max_comps
        self.num_min_comps = num_min_comps
        self.min_width = min_width
        self.max_width = max_width
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.comp_shrink_ratio = comp_shrink_ratio
        self.comp_w_h_ratio = comp_w_h_ratio
        self.text_comp_nms_thr = text_comp_nms_thr
        self.min_rand_half_height = min_rand_half_height
        self.max_rand_half_height = max_rand_half_height
        self.jitter_level = jitter_level
        self.eps = 1e-8

    def vector_angle(self, vec1, vec2):
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps)
        return np.arccos(
            np.clip(
                np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def vector_slope(self, vec):
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + self.eps))

    def vector_sin(self, vec):
        assert len(vec) == 2
        return vec[1] / (norm(vec) + self.eps)

    def vector_cos(self, vec):
        assert len(vec) == 2
        return vec[0] / (norm(vec) + self.eps)

    def find_head_tail(self, points, orientation_thr):

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(adjacent_edge_vec[0],
                                                        adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(
                    pad_points[1:] - poly_center, axis=-1),
                norm(
                    pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / (np.max(edge_dist) + self.eps)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[(i + 2):(i + len(
                    score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(score_matrix.argmax(),
                                                          score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(points[
                        2] - points[1]) + self.vector_slope(points[0] - points[
                            3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] - points[
                vertical_edge_inds[0][1]]) + norm(points[vertical_edge_inds[1][
                    0]] - points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(points[horizontal_edge_inds[0][
                0]] - points[horizontal_edge_inds[0][1]]) + norm(points[
                    horizontal_edge_inds[1][0]] - points[horizontal_edge_inds[1]
                                                         [1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def reorder_poly_edge(self, points):

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_inds, tail_inds = self.find_head_tail(points, self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline

    def cal_curve_length(self, line):

        assert line.ndim == 2
        assert len(line) >= 2

        edges_length = np.sqrt((line[1:, 0] - line[:-1, 0])**2 + (line[
            1:, 1] - line[:-1, 1])**2)
        total_length = np.sum(edges_length)
        return edges_length, total_length

    def resample_line(self, line, n):

        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 2

        edges_length, total_length = self.cal_curve_length(line)
        t_org = np.insert(np.cumsum(edges_length), 0, 0)
        unit_t = total_length / (n - 1)
        t_equidistant = np.arange(1, n - 1, dtype=np.float32) * unit_t
        edge_ind = 0
        points = [line[0]]
        for t in t_equidistant:
            while edge_ind < len(edges_length) - 1 and t > t_org[edge_ind + 1]:
                edge_ind += 1
            t_l, t_r = t_org[edge_ind], t_org[edge_ind + 1]
            weight = np.array(
                [t_r - t, t - t_l], dtype=np.float32) / (t_r - t_l + self.eps)
            p_coords = np.dot(weight, line[[edge_ind, edge_ind + 1]])
            points.append(p_coords)
        points.append(line[-1])
        resampled_line = np.vstack(points)

        return resampled_line

    def resample_sidelines(self, sideline1, sideline2, resample_step):

        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        _, length1 = self.cal_curve_length(sideline1)
        _, length2 = self.cal_curve_length(sideline2)

        avg_length = (length1 + length2) / 2
        resample_point_num = max(int(float(avg_length) / resample_step) + 1, 3)

        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    def dist_point2line(self, point, line):

        assert isinstance(line, tuple)
        point1, point2 = line
        d = abs(np.cross(point2 - point1, point - point1)) / (
            norm(point2 - point1) + 1e-8)
        return d

    def draw_center_region_maps(self, top_line, bot_line, center_line,
                                center_region_mask, top_height_map,
                                bot_height_map, sin_map, cos_map,
                                region_shrink_ratio):

        assert top_line.shape == bot_line.shape == center_line.shape
        assert (center_region_mask.shape == top_height_map.shape ==
                bot_height_map.shape == sin_map.shape == cos_map.shape)
        assert isinstance(region_shrink_ratio, float)

        h, w = center_region_mask.shape
        for i in range(0, len(center_line) - 1):

            top_mid_point = (top_line[i] + top_line[i + 1]) / 2
            bot_mid_point = (bot_line[i] + bot_line[i + 1]) / 2

            sin_theta = self.vector_sin(top_mid_point - bot_mid_point)
            cos_theta = self.vector_cos(top_mid_point - bot_mid_point)

            tl = center_line[i] + (top_line[i] - center_line[i]
                                   ) * region_shrink_ratio
            tr = center_line[i + 1] + (top_line[i + 1] - center_line[i + 1]
                                       ) * region_shrink_ratio
            br = center_line[i + 1] + (bot_line[i + 1] - center_line[i + 1]
                                       ) * region_shrink_ratio
            bl = center_line[i] + (bot_line[i] - center_line[i]
                                   ) * region_shrink_ratio
            current_center_box = np.vstack([tl, tr, br, bl]).astype(np.int32)

            cv2.fillPoly(center_region_mask, [current_center_box], color=1)
            cv2.fillPoly(sin_map, [current_center_box], color=sin_theta)
            cv2.fillPoly(cos_map, [current_center_box], color=cos_theta)

            current_center_box[:, 0] = np.clip(current_center_box[:, 0], 0,
                                               w - 1)
            current_center_box[:, 1] = np.clip(current_center_box[:, 1], 0,
                                               h - 1)
            min_coord = np.min(current_center_box, axis=0).astype(np.int32)
            max_coord = np.max(current_center_box, axis=0).astype(np.int32)
            current_center_box = current_center_box - min_coord
            box_sz = (max_coord - min_coord + 1)

            center_box_mask = np.zeros((box_sz[1], box_sz[0]), dtype=np.uint8)
            cv2.fillPoly(center_box_mask, [current_center_box], color=1)

            inds = np.argwhere(center_box_mask > 0)
            inds = inds + (min_coord[1], min_coord[0])
            inds_xy = np.fliplr(inds)
            top_height_map[(inds[:, 0], inds[:, 1])] = self.dist_point2line(
                inds_xy, (top_line[i], top_line[i + 1]))
            bot_height_map[(inds[:, 0], inds[:, 1])] = self.dist_point2line(
                inds_xy, (bot_line[i], bot_line[i + 1]))

    def generate_center_mask_attrib_maps(self, img_size, text_polys):

        assert isinstance(img_size, tuple)

        h, w = img_size

        center_lines = []
        center_region_mask = np.zeros((h, w), np.uint8)
        top_height_map = np.zeros((h, w), dtype=np.float32)
        bot_height_map = np.zeros((h, w), dtype=np.float32)
        sin_map = np.zeros((h, w), dtype=np.float32)
        cos_map = np.zeros((h, w), dtype=np.float32)

        for poly in text_polys:
            polygon_points = poly
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            if self.vector_slope(center_line[-1] - center_line[0]) > 2:
                if (center_line[-1] - center_line[0])[1] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]
            else:
                if (center_line[-1] - center_line[0])[0] < 0:
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]

            line_head_shrink_len = np.clip(
                (norm(top_line[0] - bot_line[0]) * self.comp_w_h_ratio),
                self.min_width, self.max_width) / 2
            line_tail_shrink_len = np.clip(
                (norm(top_line[-1] - bot_line[-1]) * self.comp_w_h_ratio),
                self.min_width, self.max_width) / 2
            num_head_shrink = int(line_head_shrink_len // self.resample_step)
            num_tail_shrink = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > num_head_shrink + num_tail_shrink + 2:
                center_line = center_line[num_head_shrink:len(center_line) -
                                          num_tail_shrink]
                resampled_top_line = resampled_top_line[num_head_shrink:len(
                    resampled_top_line) - num_tail_shrink]
                resampled_bot_line = resampled_bot_line[num_head_shrink:len(
                    resampled_bot_line) - num_tail_shrink]
            center_lines.append(center_line.astype(np.int32))

            self.draw_center_region_maps(
                resampled_top_line, resampled_bot_line, center_line,
                center_region_mask, top_height_map, bot_height_map, sin_map,
                cos_map, self.center_region_shrink_ratio)

        return (center_lines, center_region_mask, top_height_map,
                bot_height_map, sin_map, cos_map)

    def generate_rand_comp_attribs(self, num_rand_comps, center_sample_mask):

        assert isinstance(num_rand_comps, int)
        assert num_rand_comps > 0
        assert center_sample_mask.ndim == 2

        h, w = center_sample_mask.shape

        max_rand_half_height = self.max_rand_half_height
        min_rand_half_height = self.min_rand_half_height
        max_rand_height = max_rand_half_height * 2
        max_rand_width = np.clip(max_rand_height * self.comp_w_h_ratio,
                                 self.min_width, self.max_width)
        margin = int(
            np.sqrt((max_rand_height / 2)**2 + (max_rand_width / 2)**2)) + 1

        if 2 * margin + 1 > min(h, w):

            assert min(h, w) > (np.sqrt(2) * (self.min_width + 1))
            max_rand_half_height = max(min(h, w) / 4, self.min_width / 2 + 1)
            min_rand_half_height = max(max_rand_half_height / 4,
                                       self.min_width / 2)

            max_rand_height = max_rand_half_height * 2
            max_rand_width = np.clip(max_rand_height * self.comp_w_h_ratio,
                                     self.min_width, self.max_width)
            margin = int(
                np.sqrt((max_rand_height / 2)**2 + (max_rand_width / 2)**2)) + 1

        inner_center_sample_mask = np.zeros_like(center_sample_mask)
        inner_center_sample_mask[margin:h - margin, margin:w - margin] = \
            center_sample_mask[margin:h - margin, margin:w - margin]
        kernel_size = int(np.clip(max_rand_half_height, 7, 21))
        inner_center_sample_mask = cv2.erode(
            inner_center_sample_mask,
            np.ones((kernel_size, kernel_size), np.uint8))

        center_candidates = np.argwhere(inner_center_sample_mask > 0)
        num_center_candidates = len(center_candidates)
        sample_inds = np.random.choice(num_center_candidates, num_rand_comps)
        rand_centers = center_candidates[sample_inds]

        rand_top_height = np.random.randint(
            min_rand_half_height,
            max_rand_half_height,
            size=(len(rand_centers), 1))
        rand_bot_height = np.random.randint(
            min_rand_half_height,
            max_rand_half_height,
            size=(len(rand_centers), 1))

        rand_cos = 2 * np.random.random(size=(len(rand_centers), 1)) - 1
        rand_sin = 2 * np.random.random(size=(len(rand_centers), 1)) - 1
        scale = np.sqrt(1.0 / (rand_cos**2 + rand_sin**2 + 1e-8))
        rand_cos = rand_cos * scale
        rand_sin = rand_sin * scale

        height = (rand_top_height + rand_bot_height)
        width = np.clip(height * self.comp_w_h_ratio, self.min_width,
                        self.max_width)

        rand_comp_attribs = np.hstack([
            rand_centers[:, ::-1], height, width, rand_cos, rand_sin,
            np.zeros_like(rand_sin)
        ]).astype(np.float32)

        return rand_comp_attribs

    def jitter_comp_attribs(self, comp_attribs, jitter_level):
        """Jitter text components attributes.

        Args:
            comp_attribs (ndarray): The text component attributes.
            jitter_level (float): The jitter level of text components
                attributes.

        Returns:
            jittered_comp_attribs (ndarray): The jittered text component
                attributes (x, y, h, w, cos, sin, comp_label).
        """

        assert comp_attribs.shape[1] == 7
        assert comp_attribs.shape[0] > 0
        assert isinstance(jitter_level, float)

        x = comp_attribs[:, 0].reshape((-1, 1))
        y = comp_attribs[:, 1].reshape((-1, 1))
        h = comp_attribs[:, 2].reshape((-1, 1))
        w = comp_attribs[:, 3].reshape((-1, 1))
        cos = comp_attribs[:, 4].reshape((-1, 1))
        sin = comp_attribs[:, 5].reshape((-1, 1))
        comp_labels = comp_attribs[:, 6].reshape((-1, 1))

        x += (np.random.random(size=(len(comp_attribs), 1)) - 0.5) * (
            h * np.abs(cos) + w * np.abs(sin)) * jitter_level
        y += (np.random.random(size=(len(comp_attribs), 1)) - 0.5) * (
            h * np.abs(sin) + w * np.abs(cos)) * jitter_level

        h += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
              ) * h * jitter_level
        w += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
              ) * w * jitter_level

        cos += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
                ) * 2 * jitter_level
        sin += (np.random.random(size=(len(comp_attribs), 1)) - 0.5
                ) * 2 * jitter_level

        scale = np.sqrt(1.0 / (cos**2 + sin**2 + 1e-8))
        cos = cos * scale
        sin = sin * scale

        jittered_comp_attribs = np.hstack([x, y, h, w, cos, sin, comp_labels])

        return jittered_comp_attribs

    def generate_comp_attribs(self, center_lines, text_mask, center_region_mask,
                              top_height_map, bot_height_map, sin_map, cos_map):
        """Generate text component attributes.

        Args:
            center_lines (list[ndarray]): The list of text center lines .
            text_mask (ndarray): The text region mask.
            center_region_mask (ndarray): The text center region mask.
            top_height_map (ndarray): The map on which the distance from points
                to top side lines will be drawn for each pixel in text center
                regions.
            bot_height_map (ndarray): The map on which the distance from points
                to bottom side lines will be drawn for each pixel in text
                center regions.
            sin_map (ndarray): The sin(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
            cos_map (ndarray): The cos(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).

        Returns:
            pad_comp_attribs (ndarray): The padded text component attributes
                of a fixed size.
        """

        assert isinstance(center_lines, list)
        assert (
            text_mask.shape == center_region_mask.shape == top_height_map.shape
            == bot_height_map.shape == sin_map.shape == cos_map.shape)

        center_lines_mask = np.zeros_like(center_region_mask)
        cv2.polylines(center_lines_mask, center_lines, 0, 1, 1)
        center_lines_mask = center_lines_mask * center_region_mask
        comp_centers = np.argwhere(center_lines_mask > 0)

        y = comp_centers[:, 0]
        x = comp_centers[:, 1]

        top_height = top_height_map[y, x].reshape(
            (-1, 1)) * self.comp_shrink_ratio
        bot_height = bot_height_map[y, x].reshape(
            (-1, 1)) * self.comp_shrink_ratio
        sin = sin_map[y, x].reshape((-1, 1))
        cos = cos_map[y, x].reshape((-1, 1))

        top_mid_points = comp_centers + np.hstack(
            [top_height * sin, top_height * cos])
        bot_mid_points = comp_centers - np.hstack(
            [bot_height * sin, bot_height * cos])

        width = (top_height + bot_height) * self.comp_w_h_ratio
        width = np.clip(width, self.min_width, self.max_width)
        r = width / 2

        tl = top_mid_points[:, ::-1] - np.hstack([-r * sin, r * cos])
        tr = top_mid_points[:, ::-1] + np.hstack([-r * sin, r * cos])
        br = bot_mid_points[:, ::-1] + np.hstack([-r * sin, r * cos])
        bl = bot_mid_points[:, ::-1] - np.hstack([-r * sin, r * cos])
        text_comps = np.hstack([tl, tr, br, bl]).astype(np.float32)

        score = np.ones((text_comps.shape[0], 1), dtype=np.float32)
        text_comps = np.hstack([text_comps, score])
        text_comps = la_nms(text_comps, self.text_comp_nms_thr)

        if text_comps.shape[0] >= 1:
            img_h, img_w = center_region_mask.shape
            text_comps[:, 0:8:2] = np.clip(text_comps[:, 0:8:2], 0, img_w - 1)
            text_comps[:, 1:8:2] = np.clip(text_comps[:, 1:8:2], 0, img_h - 1)

            comp_centers = np.mean(
                text_comps[:, 0:8].reshape((-1, 4, 2)), axis=1).astype(np.int32)
            x = comp_centers[:, 0]
            y = comp_centers[:, 1]

            height = (top_height_map[y, x] + bot_height_map[y, x]).reshape(
                (-1, 1))
            width = np.clip(height * self.comp_w_h_ratio, self.min_width,
                            self.max_width)

            cos = cos_map[y, x].reshape((-1, 1))
            sin = sin_map[y, x].reshape((-1, 1))

            _, comp_label_mask = cv2.connectedComponents(
                center_region_mask, connectivity=8)
            comp_labels = comp_label_mask[y, x].reshape(
                (-1, 1)).astype(np.float32)

            x = x.reshape((-1, 1)).astype(np.float32)
            y = y.reshape((-1, 1)).astype(np.float32)
            comp_attribs = np.hstack(
                [x, y, height, width, cos, sin, comp_labels])
            comp_attribs = self.jitter_comp_attribs(comp_attribs,
                                                    self.jitter_level)

            if comp_attribs.shape[0] < self.num_min_comps:
                num_rand_comps = self.num_min_comps - comp_attribs.shape[0]
                rand_comp_attribs = self.generate_rand_comp_attribs(
                    num_rand_comps, 1 - text_mask)
                comp_attribs = np.vstack([comp_attribs, rand_comp_attribs])
        else:
            comp_attribs = self.generate_rand_comp_attribs(self.num_min_comps,
                                                           1 - text_mask)

        num_comps = (np.ones(
            (comp_attribs.shape[0], 1),
            dtype=np.float32) * comp_attribs.shape[0])
        comp_attribs = np.hstack([num_comps, comp_attribs])

        if comp_attribs.shape[0] > self.num_max_comps:
            comp_attribs = comp_attribs[:self.num_max_comps, :]
            comp_attribs[:, 0] = self.num_max_comps

        pad_comp_attribs = np.zeros(
            (self.num_max_comps, comp_attribs.shape[1]), dtype=np.float32)
        pad_comp_attribs[:comp_attribs.shape[0], :] = comp_attribs

        return pad_comp_attribs

    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, tuple)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            polygon = np.array(poly, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """
        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            instance = poly.astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask

    def generate_targets(self, data):
        """Generate the gt targets for DRRG.

        Args:
            data (dict): The input result dictionary.

        Returns:
            data (dict): The output result dictionary.
        """

        assert isinstance(data, dict)

        image = data['image']
        polygons = data['polys']
        ignore_tags = data['ignore_tags']
        h, w, _ = image.shape

        polygon_masks = []
        polygon_masks_ignore = []
        for tag, polygon in zip(ignore_tags, polygons):
            if tag is True:
                polygon_masks_ignore.append(polygon)
            else:
                polygon_masks.append(polygon)

        gt_text_mask = self.generate_text_region_mask((h, w), polygon_masks)
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)
        (center_lines, gt_center_region_mask, gt_top_height_map,
         gt_bot_height_map, gt_sin_map,
         gt_cos_map) = self.generate_center_mask_attrib_maps((h, w),
                                                             polygon_masks)

        gt_comp_attribs = self.generate_comp_attribs(
            center_lines, gt_text_mask, gt_center_region_mask,
            gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map)

        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_top_height_map': gt_top_height_map,
            'gt_bot_height_map': gt_bot_height_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }

        data.update(mapping)
        data['gt_comp_attribs'] = gt_comp_attribs
        return data

    def __call__(self, data):
        data = self.generate_targets(data)
        return data
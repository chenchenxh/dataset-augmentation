from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from PIL import Image
import os
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2
from code import interact
from random import randint
from math import cos, sin, radians

# parameters
MARGIN = 5  # [px]
ANGLE = 15  # [+/- deg]
SCALE = 20  # [+/- %]
ALLOW_DISJOINT_OBJECTS = False
AREA_MAX = 1000  # [px^2]
AREA_MIN = 0  # [px^2]
BLUR_FILTER_SIZE = 5  # [px]
BLUR_EDGES = False
N = 5  # number of pastes

class SegmentedObject:

	""" Class for segmented object that can be pasted on images. """

	def __init__(self, image_crop, mask, original_annotation, poly):
		self.image = image_crop
		self.mask = mask
		self.poly = poly
		
		# extra info
		self.area = original_annotation['area']
		self.iscrowd = original_annotation['iscrowd']
		self.category_id = original_annotation['category_id']
		self.source_image_id = original_annotation['image_id']
		self.original_annotation = original_annotation


class DatasetAugmenter:
	def __init__(self, dataset):
		self.ann_file = './annotations/instances_{}.json'.format(dataset)
		
		self.dataset_path = './{}/'.format(dataset)
		self.augmented_path = './{}_augmented/'.format(dataset)
		if not os.path.exists(self.augmented_path):
			os.makedirs(self.augmented_path)

		self.coco = COCO(self.ann_file)
		self.object_id = 900000000
		self.pasted_id_list = []


	def get_pil_image(self, ann_id=None, image_id=None):

		""" Given an annotation or image id loading the corresponding image as PIL.Image. """

		file_name = None
		if ann_id is not None:
			ann = self.coco.anns[ann_id]
			file_name = self.coco.imgs[ann['image_id']]['file_name']
		elif image_id is not None:
			file_name = self.coco.imgs[image_id]['file_name']
		if file_name is not None:
			img = Image.open(os.path.join(self.dataset_path, file_name))
			return img


	def crop_bbox_simple(self, img, bbox, margin=0, padding_for_rotation=False):

		""" Cropping rectangular area from original or mask image, with added margins. """
		
		bbox = [int(i) for i in bbox]

		# calculating actual margin that accomodates rotation
		m = MARGIN + abs(bbox[2]-bbox[3]) / 2
		origin_x = bbox[0] - m
		origin_y = bbox[1] - m

		if type(img) is np.ndarray:
			# creating an empty image of sufficient size, and pasting original image on it
			padded_image = np.zeros([img.shape[0]+2*m, img.shape[1]+2*m])
			padded_image[m:m+img.shape[0], m:m+img.shape[1]] = img
			# cropping from padded image with margins
			xmin = bbox[0] # Note: margin is both added and subtracted
			ymin = bbox[1]
			xmax = bbox[0] + bbox[2] + 2*m
			ymax = bbox[1] + bbox[3] + 2*m
			cropped_with_margins = padded_image[xmin:xmax,ymin:ymax]
			return np.copy(np.uint8(cropped_with_margins)), origin_x, origin_y
		else:
			# similar calculation as above for PIL.Image
			padded_array = np.zeros([img.height+2*m, img.width+2*m, 3], dtype=np.uint8)
			padded_image = Image.fromarray(padded_array)
			padded_image.paste(img, box=(m, m))
			xmin = bbox[0]
			ymin = bbox[1]
			xmax = bbox[0] + bbox[2] + 2*m
			ymax = bbox[1] + bbox[3] + 2*m
			cropped_with_margins = padded_image.crop(box=(xmin, ymin, xmax, ymax))
			return cropped_with_margins


	
	def get_object(self, ann_id, image_id=None):

		''' Create a SegmentedObejct: create binary mask, get the relevant crops from original and mask. '''

		ann = self.coco.anns[ann_id]

		if len(ann['segmentation']) > 1:
			print('Ignoring disjoint object.')
			return None
		if ann['area'] < AREA_MIN or ann['area'] > AREA_MAX:
			print('Area outside limits [{}, {}]'.format(AREA_MIN, AREA_MAX))
			return None

		if image_id is None:
			image_id = ann['image_id']

		img = self.get_pil_image(image_id=image_id)


		polygons = ann['segmentation']
		rle = mask_util.frPyObjects(polygons, img.height, img.width)  # Run-Length Encoding
		decoded = mask_util.decode(rle)
		if decoded.ndim == 3:	# disjoint object, multiple outline polygons on different channels
			decoded = np.amax(decoded, 2)	# flattening by taking max along channels
		mask = np.squeeze(decoded).transpose()  # binary mask
		mask_crop, origin_x, origin_y = self.crop_bbox_simple(mask, ann['bbox'], margin=MARGIN)
		img_crop = self.crop_bbox_simple(img, ann['bbox'], margin=MARGIN)

		# shift polygon origin
		shifted_polys = []
		for polygon in polygons:
			poly = np.array(polygon)
			poly = np.reshape(polygon, [np.max(poly.shape)/2, 2])
			poly = poly - np.array([origin_x, origin_y])
			shifted_polys.append(poly)

		obj = SegmentedObject(img_crop, mask_crop, ann, shifted_polys)
		return obj

	def show_ann_on_image(self, ann_ids, img=None):

		''' Display object mask on image. '''
		if img is None:
			ann0 = self.coco.anns[ann_ids[0]]
			img = self.coco.imgs[ann0['image_id']]
			I = io.imread(img['coco_url'])
		else:
			I = np.array(img)
		anns = []
		for ann_id in ann_ids:
			ann = self.coco.anns[ann_id]
			anns.append(ann)
			print('Object: {}, size: {:01f}'.format(self.coco.cats[ann['category_id']]['name'], ann['area']))
			
		plt.imshow(I); plt.axis('off')
		self.coco.showAnns(anns)
		plt.show()

	def get_paste_parameters(self, target_image, obj_img):
		angle = randint(-ANGLE, ANGLE)
		scale = randint(100-SCALE, 100+SCALE)/100.0
		# margin is too big at the moment, might calculate better placement parameters from mask
		max_x_pos = target_image.width - int(scale * obj_img.width + MARGIN)
		max_y_pos = target_image.height - int(scale * obj_img.height + MARGIN)
		x = randint(0, max_x_pos)
		y = randint(0, max_y_pos)
		return {'x': x, 'y': y, 'angle': angle, 'scale': scale}

	def create_new_ann(self, obj, target_image, paste_param):
		source_ann = obj.original_annotation
		transformed_polys = []
		for p in obj.poly:
			poly = self.transform_polygon(paste_param, p, obj)
			transformed_polys.append(poly.reshape(-1).tolist())
		new_ann = dict()
		new_ann.update({'image_id': source_ann['image_id'],
						'area': source_ann['area']*paste_param['scale']*paste_param['scale'],
						'iscrowd': source_ann['iscrowd'],
						'category_id': source_ann['category_id'],
						'id': self.object_id,
						'bbox': [0, 0, 1, 1], # TODO
						'segmentation': transformed_polys})
		self.pasted_id_list.append(self.object_id)
		self.object_id += 1

		# update COCO dataset with new annotation
		self.coco.dataset['annotations'].append(new_ann)
		self.coco.anns[new_ann['id']] = new_ann
		self.coco.imgToAnns[new_ann['image_id']].append(new_ann)
		return new_ann

	def transform_polygon(self, param, poly, obj):

		""" Shift, scale and rotate polygon. """

		a = radians(param['angle'])
		rot = np.array([[cos(a), -sin(a)],[sin(a), cos(a)]])
		shift = np.array(obj.mask.shape)/2
		shifted_poly = poly - shift
		rotated_poly = shifted_poly.dot(rot)
		shift_back_poly = rotated_poly + shift
		scaled_poly = shift_back_poly * np.array([param['scale'], param['scale']])
		pasted_poly = scaled_poly + np.array([param['x'], param['y']])
		return pasted_poly

	def paste_object(self, obj, n=N, target_image=None):

		''' Paste the extracted SegmentedObject on an image n times, default target is the source image. '''

		if obj is None:
			return

		if target_image is None:
			target_image = self.get_pil_image(image_id=obj.source_image_id)

		anns = []
		for i in range(n):
			obj_img = obj.image
			mask_img = Image.fromarray(obj.mask.transpose()*255)

			paste_param = self.get_paste_parameters(target_image, obj_img)

			# image transformation
			obj_img = obj_img.rotate(paste_param['angle'], resample=Image.BICUBIC, expand=False)
			mask_img = mask_img.rotate(paste_param['angle'], resample=Image.BICUBIC, expand=False)

			new_size = (np.array(obj_img.size) * paste_param['scale']).astype(np.int)
			obj_img = obj_img.resize(new_size, resample=Image.BICUBIC)
			mask_img = mask_img.resize(new_size, resample=Image.BICUBIC)

			if BLUR_EDGES:
				mask_img = Image.fromarray(cv2.blur(np.array(mask_img), (BLUR_FILTER_SIZE, BLUR_FILTER_SIZE)))


			target_image.paste(obj_img, box=(paste_param['x'], paste_param['y']), mask=mask_img)
			ann = self.create_new_ann(obj, target_image, paste_param)

		self.save_augmented_image(target_image, obj.source_image_id)

		# target_image.save('debug_img.jpg')
		# self.show_ann_on_image(self.pasted_id_list, img=target_image)

	def save_augmented_image(self, target_image, image_id):

		""" Save image to dir of augmented images. """

		file_name = self.coco.imgs[image_id]['file_name']
		out_path = os.path.join(self.augmented_path, file_name)
		target_image.save(out_path)


def main():
	dataset = 'val2017'
	aug = DatasetAugmenter(dataset)
	plt.ion()

	anns = aug.coco.dataset['annotations'][1000:1100]
	for ann in anns:
		obj = aug.get_object(ann_id=ann['id'])
		aug.paste_object(obj, n=10)

	interact(local=locals())
	
	obj = aug.get_object(ann_id=12582)
	if obj is not None:
		aug.paste_object(obj, n=50)
	
if __name__ == '__main__':
	main()

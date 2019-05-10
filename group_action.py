import torch

from constants import BATCH_SIZE
from transform_tensor_batch import TransformTensorBatch


class Group:
    def __init__(self, name, nr_group_elems, base_element):
        self.name = name
        self.nr_group_elems = nr_group_elems
        self.base_element = base_element


class RotationGroupTransformer:
    group: Group
    rotation_batch_tansformer: TransformTensorBatch

    def __init__(self, group, rotation_batch_tansformer, device):
        self.rotation_batch_tansformer = rotation_batch_tansformer
        self.group = group
        self.index_sample = None
        self.elements_sample = None
        self.device = device

    def get_random_sample(self, size=(1,)):
        """
        :return sample_index: tensor of int in range(0, self.group.nr_group_elems) with shape shape=size
        """
        index_sample = torch.randint(0, self.group.nr_group_elems, size=size, device=self.device, dtype=torch.float)
        return index_sample

    def set_elements_sample(self, index=None):
        """
        index should be a int in range(0, self.group.nr_group_elems) and it would be broadcasted for the batch size
        :param index:
        :return:
        """
        if index is None:
            self.index_sample = torch.randint(0, self.group.nr_group_elems, size=(BATCH_SIZE,), device=self.device)
            self.elements_sample = self.index_sample * self.group.base_element
        elif index.shape == (1,):
            element_sample = index * self.group.base_element
            self.elements_sample = torch.ones(size=(BATCH_SIZE,), device=self.device) * element_sample
        else:
            element_sample = index * self.group.base_element
            self.elements_sample = element_sample

    def apply_sample_action_to_input(self, images, index=None):
        if index is None:
            elements_sample = self.elements_sample
        elif index.shape == (1,):
            index_sample = index * torch.ones(size=(images.shape[0],), device=self.device)
            elements_sample = index_sample * self.group.base_element
        else:
            index_sample = index
            elements_sample = index_sample * self.group.base_element

        # ADDED FOR REVERSE EXPERIMENTS:
        # elements_sample = -elements_sample

        rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images,
                                                                                rot_thetas=elements_sample)
        return rotated_images

    def apply_rotation_to_input(self, images, angle):
        rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images,
                                                                                rot_thetas=torch.tensor(angle, device=self.device, dtype=torch.float32))
        return rotated_images

    def apply_identity_action_to_input(self, images):
        index = torch.zeros((1,), device=self.device)
        return self.apply_sample_action_to_input(images, index)

    def apply_sample_action_to_activations(self, images, group_axis, index=None):
        """

        :param images:
        :param group_axis: group axis is probably 2
        :return:
        """
        # if index is None:
        #     images_rolled = self.rotation_batch_tansformer.batch_roll(images, self.index_sample, group_axis)
        #     rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images_rolled,
        #                                                                             rot_thetas=self.elements_sample)
        # elif index.shape == (1,):
        #     index_sample = index * torch.ones(size=(images.shape[0],), device=self.device)
        #     elements_sample = index_sample * self.group.base_element
        #     images_rolled = self.rotation_batch_tansformer.batch_roll(images, index_sample, group_axis)
        #     rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images_rolled,
        #                                                                             rot_thetas=elements_sample)
        # else:
        #     index_sample = index
        #     elements_sample = index_sample * self.group.base_element
        #     images_rolled = self.rotation_batch_tansformer.batch_roll(images, index_sample, group_axis)
        #     rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images_rolled,
        #                                                                             rot_thetas=elements_sample)
        # return rotated_images
        pass

    def apply_roll(self, images, group_axis, index, dim):
        """

        :param images:
        :param group_axis: group axis is probably 2
        :param index: list[int]
            index of roll, how much to roll each image in images
        :return:
        """
        # ADDED FOR REVERSE EXPERIMENTS:
        # index = -index

        images_rolled = self.rotation_batch_tansformer.batch_roll(images, index, group_axis, group_sz=self.group.nr_group_elems, dim=dim)
        return images_rolled


class ScaleGroupTransformer:
    group: Group
    scale_batch_tansformer: TransformTensorBatch

    def __init__(self, group, scale_batch_tansformer, device):
        self.scale_batch_tansformer = scale_batch_tansformer
        self.group = group
        self.index_sample = None
        self.elements_sample = None
        self.device = device
        self.scales = torch.linspace(1, 2, self.group.nr_group_elems, device=device)

    def get_random_sample(self, size=(1,)):
        """
        :return sample_index: tensor of int in range(0, self.group.nr_group_elems) with shape shape=size
        """

        index_sample = torch.randint(0, self.group.nr_group_elems, size=size, device=self.device)
        return index_sample

    def set_elements_sample(self, index=None):
        """
        index should be a int in range(0, self.group.nr_group_elems) and it would be broadcasted for the batch size
        :param index:
        :return:
        """
        if index is None:
            self.index_sample = self.get_random_sample(size=(BATCH_SIZE,))
            self.elements_sample = self.scales[self.index_sample.long()]
        elif index.shape == (1,):
            element_sample = self.scales[index.long()]
            self.elements_sample = torch.ones(size=(BATCH_SIZE,), device=self.device) * element_sample
        else:
            element_sample = self.scales[index.long()]
            self.elements_sample = element_sample

    def apply_sample_action_to_input(self, images, index=None):
        if index is None:
            elements_sample = self.elements_sample
        elif index.shape == (1,):
            index_sample = index * torch.ones(size=(images.shape[0],), device=self.device)
            elements_sample = self.scales[index_sample.long()]
        else:
            index_sample = index
            elements_sample = self.scales[index_sample.long()]

        # ADDED FOR REVERSE EXPERIMENTS:
        # elements_sample = -elements_sample

        scaled_images = self.scale_batch_tansformer.rotate_and_scale_images(images=images,
                                                                            scale_thetas=elements_sample)
        return scaled_images

    def apply_identity_action_to_input(self, images):
        index = torch.zeros((1,), device=self.device)
        return self.apply_sample_action_to_input(images, index)

    def apply_sample_action_to_activations(self, images, group_axis, index=None):
        """

        :param images:
        :param group_axis: group axis is probably 2
        :return:
        """
        # if index is None:
        #     images_rolled = self.rotation_batch_tansformer.batch_roll(images, self.index_sample, group_axis)
        #     rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images_rolled,
        #                                                                             rot_thetas=self.elements_sample)
        # elif index.shape == (1,):
        #     index_sample = index * torch.ones(size=(images.shape[0],), device=self.device)
        #     elements_sample = index_sample * self.group.base_element
        #     images_rolled = self.rotation_batch_tansformer.batch_roll(images, index_sample, group_axis)
        #     rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images_rolled,
        #                                                                             rot_thetas=elements_sample)
        # else:
        #     index_sample = index
        #     elements_sample = index_sample * self.group.base_element
        #     images_rolled = self.rotation_batch_tansformer.batch_roll(images, index_sample, group_axis)
        #     rotated_images = self.rotation_batch_tansformer.rotate_and_scale_images(images=images_rolled,
        #                                                                             rot_thetas=elements_sample)
        # return rotated_images
        pass

    def apply_roll(self, images, group_axis, index, dim):
        """

        :param images:
        :param group_axis: group axis is probably 2
        :param index: list[int]
            index of roll, how much to roll each image in images
        :return:
        """
        # ADDED FOR REVERSE EXPERIMENTS:
        # index = -index

        images_rolled = self.scale_batch_tansformer.batch_roll(images, index, group_axis, group_sz=self.group.nr_group_elems, dim=dim)
        return images_rolled

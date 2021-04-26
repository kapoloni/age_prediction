# Torch implementation of a brain age estimation model.

Torch implementation of EfficientNet B0 to estimate the brain age from hippocampal MR images of cognitively normal subjects.
Efficient Model from: https://github.com/shijianjian/EfficientNet-PyTorch-3D <br>
and paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks<br>

Images from the public ADNI and IXI datasets:<br>
http://adni.loni.usc.edu/ and https://brain-development.org/ixi-dataset/<br>

# This repository was tested using:

    Python                 3.6.9
    torch                  1.7.1
    torchio                0.18.23
    torchvision            0.8.2

# We applied some techniques for data augmentation<br>
Using the TorchIO Python library (https://torchio.readthedocs.io/transforms/augmentation.html) with the following transformations:<br>

    Gaussian noise
    Bias Field
    Translation
    Rotation
 
 <table width="100%" border="0" cellpadding="5">
	<tr>
		<td align="center" valign="center">
		<img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_3a.png" alt="description here" />
		<br />
			Gaussian kernel example.
		</td>
		<td align="center" valign="center">
		<img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_3b.png" alt="description here" />
		<br />
			Line kernel example.
		</td>
		<td align="center" valign="center">
		<img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_3c.png" alt="description here" />
		<br />
			Curve kernel example.
		</td>
		<td align="center" valign="center">
		<img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_4b.png" alt="description here" />
		<br />
			Airy disk pattern.
		</td>
	</tr>
</table>


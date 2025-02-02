参考的论文：
Exposure Fusion: A Simple and Practical Alternative to High Dynamic Range Photography

论文简介：
	• 多张不同曝光度的图像合成LDR图像，不用tune mapping
	• 方案：
		○ 加权平均，权重受对比度C，饱和度S和曝光程度E影响，最终的图像为输入图像序列的加权平均
		○ 对比度C的计算
			§ 对灰度图，拉普拉斯滤波
		○ 饱和度S的计算
			§ RGB的标准差
		○ 曝光程度E的计算
			§ 高斯曲线，作用于Intensity channel
		○ 权重为C,S,E的乘积，指数可以进行控制
	• 改进
		○ 直接这样计算会有seam
		○ 一种改进是用Gauss 平滑权重，或者用边缘感知的平滑滤波器平滑
		○ 最终改进
			拉普拉斯金字塔，高斯金字塔
			
所作工作：
	* 测试的图片是手机拍摄的，为了模拟不同曝光度，用手机的照片编辑功能拖出了三张不同曝光度的图片。
	* 只实现了简单的计算权重然后加权平均，还没有平滑权重，也没有用”金字塔“。
	* 效果：并不好，可以看result文件夹里的图片
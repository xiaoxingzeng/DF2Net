# DF2Net: A Dense-Fine-Finer Network for Detailed 3D Face Reconstruction

This repository contains our work in [ICCV19](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjfw5Gp8fPlAhXMc94KHSbBBOUQFjAAegQICBAC&url=http%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_ICCV_2019%2Fpapers%2FZeng_DF2Net_A_Dense-Fine-Finer_Network_for_Detailed_3D_Face_Reconstruction_ICCV_2019_paper.pdf&usg=AOvVaw3YNf8l3E3XBnzB9GTTZ62h "悬停显示")
This paper proposes a deep Dense-Fine-Finer Network (DF2Net) to address the challenging problem of
high-fidelity 3D face reconstruction from a single image.
DF2Net is composed of three modules, namely D-Net,
F-Net, and Fr-Net. It progressively refines the subtle facial
details such as small crow’s feet and wrinkles. We introduce
three types of data to train DF2Net with different training
strategies. More details can be seen in our paper.
![framework](https://github.com/xiaoxingzeng/DF2Net/tree/master/img/framework.png)  
Xiaoxing Zeng, Xiaojiang Peng, Yu Qiao. DF2Net: A Dense-Fine-Finer Network for Detailed 3D Face Reconstruction. ICCV, 2019    
  
  Dependencies
  ------------
   * Pytorch 0.4.0
   * Python 2.7
   * Dlib
   * PIL

  Run the demo
  ------------
  
  #### Face Crop  
  We crop the raw face image with bounding box of face detection, you can change to other more advanced detector.  
  #### Face Mask Generate  
  We also masking the cropped face with 68 face landmarks detector.  
  #### DownLoad the pretrained model[google drive](https://drive.google.com/open?id=13rNnb__OrD7Zv8Mx3bdwjWr_ELmhUzeI "悬停显示") [baidu drive access code：f5tb](https://pan.baidu.com/s/1-CuHbM6nyWNVV_PanRRfLQ "悬停显示") 	 and copy it to ./model. DownLoad the shape_predictor_68_face_landmarks.dat [google drive](https://drive.google.com/open?id=1SeIs0lG1XAg1JN6bGjiXUgMONsTuTpxy "悬停显示") [biadu drive access code :yi4u ](https://pan.baidu.com/s/1UaozUXwF1_-t7tOqDwEnbQ "悬停显示")
  #### RUN  
   `python demo.py`  
  #### Show the result with Matlab  
   `show_output.m`  
  #### Show the textured raw mesh  
  `python pointcloud2rawmesh.py`
  #### Note  
  There may be some visual different from our results of paper to our demo, this is due to the different of cropping and masking way.
  
  Citation
  --------  
  @inproceedings{zeng2019df2net,  
  title={DF2Net: A Dense-Fine-Finer Network for Detailed 3D Face Reconstruction},  
  author={Zeng, Xiaoxing and Peng, Xiaojiang and Qiao, Yu},  
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},  
  pages={2315--2324},  
  year={2019}  
}  

  Acknowledgements
  ----------------  
  Thanks the authors of [extreme 3d faces](https://github.com/anhttran/extreme_3d_faces "悬停显示"),[PRNet](https://github.com/YadiraF/PRNet "悬停显示"), [SfSNet](https://github.com/senguptaumd/SfSNet "悬停显示"), [pix2vertex](https://github.com/matansel/pix2vertex "悬停显示") for their inspiring works.

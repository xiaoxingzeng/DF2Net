import numpy as np
import scipy.io as sio
import cv2
from scipy import ndimage


cacd_list = open('img_list.txt','r')
depth_mat_path = './output/'
image_path  = './img/'

for line in cacd_list:
    
    line = line.strip('\n')
    print line
    depth_abs_path = depth_mat_path + line.replace('.png','.mat')
    depth_mat_file = sio.loadmat(depth_abs_path)
    depth_mat = depth_mat_file['depth_mat']
    
    
    image = cv2.imread(image_path+line.replace('.png','_crop.png'))
    
    
    mask = image.sum(2)
    mask[mask!=0]=1
    mask_index = np.where(mask==1)
    
    
    ply_name = line.replace('.png','.obj')
    
    ply_name = './out_obj/'+ply_name
    out_ply = open(ply_name,'w')
    
    
    

    face_index_map = np.zeros((512,512))
    sum_nonzero = 1
    for i in range(512):
          #print i
          
          first_line = depth_mat[i,:]
         # first_stack = first_line.sum(1)
          index_first = np.where(first_line!=0)
          
     
          
          index_first = index_first[0]
    
          if len(index_first)>5:
             first_head = index_first[0]
             first_tail = index_first[-1]
          
             for j in range(len(index_first)):
                 lie = index_first[j]
                 out_ply.write('v'+' '+str(i)+' '+str(lie)+' '+str(depth_mat[i,lie])+' '+str(image[i,lie,2])+' '+str(image[i,lie,1])+' '+str(image[i,lie,0])+'\n')
                 
             bb = range(0,len(index_first))
             bb = np.array(bb)
             bb = bb+sum_nonzero
             #print bb
             face_index_map[i,index_first] = bb
             #print face_index_map[i,index_first]
             sum_nonzero += len(index_first)
    
    
    num_face=0
    for m in range(511):
          
          fir_line = depth_mat[m,:]
          #fir_stack = fir_line.sum()
          index_fir = np.where(fir_line!=0)
          index_fir = index_fir[0]
          
          if len(index_fir)>5:
             fir_head = index_fir[0]
             fir_tail = index_fir[-1] 
          
             sec_line = depth_mat[m+1,:]
             
             index_sec = np.where(sec_line!=0)
             index_sec = index_sec[0]
             if len(index_sec)>5:
    
                number_nonzero_first = len(index_fir)
                number_nonzero_second = len(index_sec)
                
                if number_nonzero_first<number_nonzero_second:
                   if index_sec[-1]<=index_fir[-1]:
                       
                      for k in range(number_nonzero_first-1):
                          f_index_1 = long(face_index_map[m,index_fir[k]])
                          f_index_2 = long(face_index_map[m+1,index_sec[number_nonzero_second-number_nonzero_first+k]])
                          f_index_3 = long(face_index_map[m+1,index_sec[number_nonzero_second-number_nonzero_first+k+1]])
                          out_ply.write('f'+' '+str(f_index_1)+' '+ str(f_index_3)+' '+str(f_index_2)+'\n')
                          
                          f_index_4 = long(face_index_map[m,index_fir[k+1]])
                          f_index_5 = long(face_index_map[m,index_fir[k]])
                          f_index_6 = long(face_index_map[m+1,index_sec[number_nonzero_second-number_nonzero_first+k+1]])
                          out_ply.write('f'+' '+str(f_index_5)+' '+ str(f_index_4)+' '+str(f_index_6)+'\n')
                   elif index_fir[0]<=index_sec[0]:
                      
                      for j in range(number_nonzero_first-1):
                          f_index_1 = long(face_index_map[m,index_fir[j]])
                          f_index_2 = long(face_index_map[m+1,index_sec[j]])
                          f_index_3 = long(face_index_map[m+1,index_sec[j+1]])
                          
                          out_ply.write('f'+' '+str(f_index_1)+' '+ str(f_index_3)+' '+str(f_index_2)+'\n')
                          
                          f_index_4 = long(face_index_map[m,index_fir[j+1]])
                          f_index_5 = long(face_index_map[m,index_fir[j]])
                          f_index_6 = long(face_index_map[m+1,index_sec[j+1]])
                          out_ply.write('f'+' '+str(f_index_5)+' '+ str(f_index_4)+' '+str(f_index_6)+'\n')
                   else:
                      idx_ori = (np.abs(index_sec-index_fir[0])).argmin()
                      idx_ori_last = (np.abs(index_sec-index_fir[-1])).argmin()
                      #print 'len index_fir',len(index_fir),index_fir
                      #print 'len index_sec',len(index_sec),index_sec
                      #print 'type idx_ori',(idx_ori)
    #                  if (idx_ori)!=1:
    #                     idx = idx_ori[0]
    #                  else:
    #                     idx = idx_ori
                      idx = idx_ori
                      
                      second_selected = len(index_sec[idx_ori:idx_ori_last+1])
                      
                      common_number = min(second_selected,number_nonzero_first)
                      
                      for r in range(common_number-1):
                          f_index_1 = long(face_index_map[m,index_fir[r]])
                          f_index_2 = long(face_index_map[m+1,index_sec[idx+r]])
                          f_index_3 = long(face_index_map[m+1,index_sec[idx+r+1]])
                          
                          out_ply.write('f'+' '+str(f_index_1)+' '+ str(f_index_3)+' '+str(f_index_2)+'\n')
                          
                          f_index_4 = long(face_index_map[m,index_fir[r+1]])
                          f_index_5 = long(face_index_map[m,index_fir[r]])
                          f_index_6 = long(face_index_map[m+1,index_sec[idx+r+1]])
                          out_ply.write('f'+' '+str(f_index_5)+' '+ str(f_index_4)+' '+str(f_index_6)+'\n')
                                
                         
                   
                elif number_nonzero_first>number_nonzero_second:
                   if index_sec[0]>=index_fir[0]:
                      for n in range(number_nonzero_second-1):
                          
                          f_index_1 = long(face_index_map[m,index_fir[n]])
                          f_index_2 = long(face_index_map[m+1,index_sec[n]])
                          f_index_3 = long(face_index_map[m+1,index_sec[n+1]])
                          
                          out_ply.write('f'+' '+str(f_index_1)+' '+ str(f_index_3)+' '+str(f_index_2)+'\n')
                          
                          f_index_4 = long(face_index_map[m,index_fir[n+1]])
                          f_index_5 = long(face_index_map[m,index_fir[n]])
                          f_index_6 = long(face_index_map[m+1,index_sec[n+1]])
                          out_ply.write('f'+' '+str(f_index_5)+' '+ str(f_index_4)+' '+str(f_index_6)+'\n')
                   elif index_fir[-1]<=index_sec[-1]:
                      for p in range(number_nonzero_second-1):
                          f_index_1 = long(face_index_map[m,index_fir[-number_nonzero_second+number_nonzero_first+p]])
                          f_index_2 = long(face_index_map[m+1,index_sec[p]])
                          f_index_3 = long(face_index_map[m+1,index_sec[p+1]])
                          out_ply.write('f'+' '+str(f_index_1)+' '+ str(f_index_3)+' '+str(f_index_2)+'\n')
                          
                          f_index_4 = long(face_index_map[m,index_fir[-number_nonzero_second+number_nonzero_first+p+1]])
                          f_index_5 = long(face_index_map[m,index_fir[-number_nonzero_second+number_nonzero_first+p]])
                          f_index_6 = long(face_index_map[m+1,index_sec[p+1]])
                          out_ply.write('f'+' '+str(f_index_5)+' '+ str(f_index_4)+' '+str(f_index_6)+'\n') 
                   else:
                      idx_ori = (np.abs(index_fir-index_sec[0])).argmin()
                      print 'type idx_ori',type(idx_ori)
                      idx = idx_ori
                      for q in range(number_nonzero_second-1):
                          f_index_1 = long(face_index_map[m,index_fir[idx+q]])
                          f_index_2 = long(face_index_map[m+1,index_sec[q]])
                          f_index_3 = long(face_index_map[m+1,index_sec[q+1]])
                          out_ply.write('f'+' '+str(f_index_1)+' '+ str(f_index_3)+' '+str(f_index_2)+'\n')
                          
                          f_index_4 = long(face_index_map[m,index_fir[idx+q+1]])
                          f_index_5 = long(face_index_map[m,index_fir[idx+q]])
                          f_index_6 = long(face_index_map[m+1,index_sec[q+1]])
                          out_ply.write('f'+' '+str(f_index_5)+' '+ str(f_index_4)+' '+str(f_index_6)+'\n')
                   
       
                else:
                   for w in range(number_nonzero_second-1):
                       f_index_1 = long(face_index_map[m,index_fir[w]])
                       f_index_2 = long(face_index_map[m+1,index_sec[w+1]])
                       f_index_3 = long(face_index_map[m+1,index_sec[w]])
                       out_ply.write('f'+' '+str(f_index_1)+' '+ str(f_index_2)+' '+str(f_index_3)+'\n')
                       
                       f_index_4 = long(face_index_map[m,index_fir[w+1]])
                       f_index_5 = long(face_index_map[m,index_fir[w]])
                       f_index_6 = long(face_index_map[m+1,index_sec[w+1]])
                       out_ply.write('f'+' '+str(f_index_5)+' '+ str(f_index_4)+' '+str(f_index_6)+'\n')
    
                 
    out_ply.close()

      

         
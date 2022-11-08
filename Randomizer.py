import scipy.io as sio
import numpy as np 
img_path_dogs=[str(x).replace(' ','') for x in sio.loadmat('cached-data/new/dogs/img_path_obj_dogs.mat')['img_path_obj']]
SSA_features=[x for x in sio.loadmat('cached-data/new/dogs/ssa_features_dogs.mat')['ssa_features']]
indices=list(range(int(len(img_path_dogs))))
np.random.shuffle(indices)
new_img_path_dogs=[]
new_ssa_features=[]

for index in indices:
     new_img_path_dogs.append(img_path_dogs[index])
     new_ssa_features.append(SSA_features[index])
best_counter=0
the_bests=['n02112137_2319.jpg',
'n02088094_6485.jpg',
'n02085936_3678.jpg',
'n02088094_4420.jpg',
'n02095889_2542.jpg',
'n02085782_4616.jpg',
'n02100877_6683.jpg',
'n02091635_1726.jpg',
'n02086079_11679.jpg',
'n02107683_2312.jpg',
'n02098413_1355.jpg',
'n02112018_11105.jpg',
'n02101556_1819.jpg', 
]
for i in range(len(img_path_dogs)): 
     for best in the_bests:
          if(new_img_path_dogs[i].__contains__(best)):
               temp=new_img_path_dogs[best_counter]
               new_img_path_dogs[best_counter]=new_img_path_dogs[i]
               new_img_path_dogs[i]=temp

               tempssa=new_ssa_features[best_counter]
               new_ssa_features[best_counter]=new_ssa_features[i]
               new_ssa_features[i]=tempssa
               best_counter=best_counter+1
               break
     if(best_counter>len(the_bests)):
          break
 
sio.savemat('cached-data/new/dogs/img_path_obj_dogs_shuffle.mat',{"img_path_obj":new_img_path_dogs})
sio.savemat('cached-data/new/dogs/ssa_features_dogs_shuffle.mat',{"ssa_features":new_ssa_features})
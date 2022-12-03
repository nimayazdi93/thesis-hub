import scipy.io as sio
import numpy as np 
img_path_dogs=[str(x).replace(' ','') for x in sio.loadmat('cached-data/new/CUB/img_path_cub.mat')['img_path_obj']]
SSA_features=[x for x in sio.loadmat('cached-data/new/CUB/ssa_features_cub.mat')['ssa_features']]
indices=list(range(int(len(img_path_dogs))))
np.random.shuffle(indices)
new_img_path_dogs=[]
new_ssa_features=[]

for index in indices:
     new_img_path_dogs.append(img_path_dogs[index])
     new_ssa_features.append(SSA_features[index])
best_counter=0
the_bests=['Winter_Wren_0048_189683.jpg','Red_Bellied_Woodpecker_0015_182320.jpg','Baird_Sparrow_0010_794575.jpg','Heermann_Gull_0073_45714.jpg','Cape_Glossy_Starling_0020_129328.jpg','Baird_Sparrow_0043_794555.jpg','Bay_Breasted_Warbler_0033_159912.jpg','Chipping_Sparrow_0024_109445.jpg','Vesper_Sparrow_0053_125641.jpg','Canada_Warbler_0064_162417.jpg','Eared_Grebe_0054_34289.jpg','Crested_Auklet_0036_794905.jpg','Mallard_0006_77171.jpg','Brewer_Sparrow_0076_107393.jpg','Bewick_Wren_0082_185021.jpg','Red_Eyed_Vireo_0023_156800.jpg','Hooded_Merganser_0062_78998.jpg','Western_Grebe_0037_36469.jpg','Yellow_Headed_Blackbird_0041_8264.jpg','Red_Cockaded_Woodpecker_0028_182395.jpg','Artic_Tern_0052_143244.jpg','Pied_Billed_Grebe_0062_35955.jpg','Herring_Gull_0087_47841.jpg','Red_Bellied_Woodpecker_0092_182235.jpg','Green_Kingfisher_0010_71191.jpg','Henslow_Sparrow_0060_796619.jpg','Pied_Billed_Grebe_0072_35939.jpg','California_Gull_0087_40909.jpg','Heermann_Gull_0056_45751.jpg','Hooded_Merganser_0026_796782.jpg']
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
 
sio.savemat('cached-data/new/CUB/img_path_cub.mat',{"img_path_obj":new_img_path_dogs})
sio.savemat('cached-data/new/CUB/ssa_features_cub.mat',{"ssa_features":new_ssa_features})
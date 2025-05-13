import os
from tqdm import *
# 批量重命名为 'img_([0-9]+).txt' 的格式
#img_folder_path = r'D:\work_wjp\biaozhu\biaozhu_2\sys_cls_2\Test'
img_folder_path = r'F:/user_wjp/work/jiedan/6_medicinal_materials_classification/Awesome-Backbones-main/yaocai_data/jinyinhua'
# img_folder_new_path = r'./img_7_folder_new'
num = 0
for i, every_img in tqdm(enumerate(sorted(os.listdir(img_folder_path))), total=len(os.listdir(img_folder_path))):
    print("ImageId:", i + 1)
    if every_img.endswith('.jpg'):
        every_img_name = every_img.split('.')[0]
        old_image_file_path = os.path.join(img_folder_path,every_img)
        new_image_file_name = img_folder_path.split("/")[-1] + "_" + str(f'{num:04}')+ '.' + every_img.split('.')[1]
        new_image_file_path = img_folder_path + '/' + new_image_file_name
        os.rename(old_image_file_path,new_image_file_path)

        num += 1

print()
print("All Done")
print()
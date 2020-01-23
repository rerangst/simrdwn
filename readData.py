import os
dir_path= "data/train_data/labels"
su=0
for item in os.listdir(dir_path):
    # print(item)
    with open(os.path.join(dir_path,item),'r') as f:
        tmp = int(f.readline())
        su=su+tmp
print(su)
    # f.writelines("-l" + (i+"\t\\"+"\n") for i in release)
import os
folders = '../dataset'
for folder in os.listdir(folders):
    newname = folder.strip('cropped')
    os.rename(os.path.join(folders, folder), os.path.join(folders, newname))
    # for i, file in enumerate(os.listdir(os.path.join(folders, folder))):
    #     newfilename = folder.strip('cropped')+'-'+str(i)+'.jpg'
    #     os.rename(os.path.join(folders, folder, file), os.path.join(folders, folder, newfilename))
import os

# Train all splits:
def get_cmds(dir):
    cmds = []
    for config_file in sorted(os.listdir(dir)):
        config_file = os.path.join(dir, config_file)
        cmds.append(f"python mmsegmentation/tools/train.py {config_file}")
    return cmds

#Train split-0.py
''''def get_cmds(dir):
    cmds = []
    cmds.append(f"python mmsegmentation/tools/train.py {dir}/split-0.py")
    return cmds
'''

def main():
    dir = "mmsegmentation/configs/segformer/organoid_segmentation"
    cmds = get_cmds(dir)
    for cmd in cmds:
        print(cmd)
        os.system(cmd)
        # break

main()
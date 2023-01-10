# MRI-sellar-region-feature-extractor
1. build the nnUNet enviroment follow "https://github.com/MIC-DKFZ/nnUNet"
2. place folder "2d" to the folder "nnUNet_trained_models"
3. replace the standard python file by the given "generic_UNet.py".
4. run nnUNet_predict -i ./imagesTr -o ./infersTs_demo -t 068 -f all -m 2d
Pretrained weight will be uploaded soom.

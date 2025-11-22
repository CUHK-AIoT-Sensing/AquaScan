# image_reconstruction

image_reconstruction is a Python program for reconstructing the "skip-scan" images. Since our method requires the nearby pixels not the full images, our method itself can be used even if we are collecting data, which decreases the overhead greatly. Noticed that code here is to process the full images, you can try to implement a steaming version.

```bash
image_reconstruction/
├── reconstruction.py
├── README.md
├── prepare.sh
└── remove_recover.sh
```

## Usage

```bash
python3 reconstruction.py --raw path --save_txt txt/ --save_img img/ --skip 2 --offset 1 --scan 1 
```
raw: unreconstructed data path.         
      
save_txt: reconstructed data(txt) saving path.  
 
save_img: reconstructed data(img) saving path.

skip: skip angles.  

offset: initial angle offset.   

scan: scan angles.  

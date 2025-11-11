# image_reconstruction

image_reconstruction is a Python program for reconstructing the "skip-scan" images.

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

# image_reconstruction

image_reconstruction is a Python program for reconstructing the "skip-scan" images.

```bash
image_reconstruction/
├── reconstruction.py
├── recover
│   ├── img
│   └── txt
├── raw_data
│   ├── 2292002
│   ├── 2292004
│   ├── 2292005
│   ├── 08071005
│   ├── 08140001 #background data
│   ├── 08141002
│   ├── 08213000 #background data
│   ├── 08213003
│   ├── 08213004
│   └── 08213005
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

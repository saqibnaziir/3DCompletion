# Data

Put the COMA meshes here (download from https://coma.is.tue.mpg.de/ after accepting the licence), keeping the original folder layout:

```
data/COMA/
├── FaceTalk_170725_00137_TA/
│   ├── bareteeth/
│   │   ├── bareteeth.000001.ply
│   │   └── ...
│   ├── cheeks_in/
│   └── ...                      (12 expressions)
├── FaceTalk_170728_03272_TA/
└── ...                          (12 subjects)
```

Nothing else is needed: `python prepare_data.py` builds the split, the processed cache and the test benchmark from these files. The data is not part of this repository.

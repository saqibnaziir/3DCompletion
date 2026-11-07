# Data

Neither dataset is included in this repository. Download them from their authors (a licence is required) and put them here.

## COMA (faces)

Download from https://coma.is.tue.mpg.de/ and keep the original folder layout:

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

Then: `python prepare_data.py`

## DFAUST (bodies)

Download the registrations from https://dfaust.is.tue.mpg.de/ and put the HDF5 files here as they are:

```
data/DFAUST/
├── registrations_m.hdf5
└── registrations_f.hdf5         (either or both)
```

Then: `python prepare_data.py --config config/dfaust.cfg`. It converts the HDF5 files to meshes (every 6th frame, set by `hdf5_frame_stride`) in `runs/dfaust_subject_split/meshes/<subject>/<motion>/`.

Already-exported meshes also work: point `data_dir` in `config/dfaust.cfg` at a folder of `<subject>/<motion>/<frame>.ply` or `<subject>_<motion>/<frame>.ply`.

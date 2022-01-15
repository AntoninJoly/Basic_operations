# Info
Source: reddit

Date: 2021/06/05

# Post header
H5Records : Store large datasets in one single files with index access  
Recently I tried using TF-Record in Pytorch, however a lot of heavily used features by Pytorch such as data length, index level access isn't available.
So far I have tested it on datasets of size around 200G+ without any major issue. I hope this would be useful for the community when dealing with extreme large datasets.

# Comments section
>Read wrappers over h5py files is great but having wrappers to write data for creating those files can also help a huge tone. 

>I saw great performance gains when preaching batches in memory instead of reading the h5py file directly. Sometimes due to incorrect selection of compression, it may be slower. Creating a custom sampler for such a dataset can be really beneficial as it creates a way to precache stuff in memory and essentially load data faster.

>Do note due to HDF5 variable length data, the video data (channel x height x width x length ) is flatten into one flat dimension and reshaped everytime the video is accessed. So there might be a high compute penalty here.

# Notes
None

# Though:
- Useful format for normalizing data storage
- Large dataset can be handled without too much harsh.
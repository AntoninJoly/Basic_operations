# Info
Source: reddit
Date: 2021/12/02

# Post header
How to load 85.6 GB of XML data into a dataframe  
>pd.read_xml() -> System crashes - High memory usage
>ElementTree -> System crashes - High memory usage
>vaex [OpenSource library] -> Does not handles XML data well. throws an error.

# Comments section

>I've had good results using the HDF5 format in the past with the associated python library. It allows you to load huge amounts of data and process it like a numpy array. It involved preprocessing the data and then feeding it into a new hdf5 data structure that u save to disk. You can then load that in python and use it like a numpy array. https://docs.h5py.org/en/stable/

>We do similar by converting the XML to JSON early in the data pipeline. About 31 TB total; about 700 million individual XML files that range from tiny to quite a few MB.

>Check out dask. It's a library that allows you to work with big data on small computers by lazily evaluating work and only loading what you can deal with on your machine.

>Yes, daskdf can help with this. If you have GPU, I think cudf will help too.

# Notes
wikipedia data handling.  
https://github.com/ramayer/wikipedia_in_spark/blob/main/notebooks/0_spark_preprocessing_of_wikipedia.ipynb

# Though:
- Dask and HDF5 seems the solution for handling large data.

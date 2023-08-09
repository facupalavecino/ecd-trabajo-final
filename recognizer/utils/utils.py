def get_metadata_from_filename(filename: str):
    """Decomposes a filename and returns a tuple containing (class, subject, repetition)
    
    Parameters
    ----------
    filename : str
        Filename to decompose. Must include extension

    Example
    -------
    >>> m = get_metadata_from_filename('001_002_003_right.avi')
    >>> m
    >>> '001', '002', '003'
    """
    filename = filename.split(".")[0]

    metadata = filename.split("_")

    klass = metadata[0]
    subject = metadata[1]
    repetition = metadata[2]

    return klass, subject, repetition

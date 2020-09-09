import zlib
import io
import tqdm

def ungzip_progress(src_file_fullname, trgt_file_fullname, chunk_size=4096) :
    '''
    decompresses a gzip file with progress bar.

    parameters :
      - src_file_fullname (string) : full path to the source (compressed) file
      - trgt_file_fullname (string) : full path to the target (decompressed) file
      - chunk_size (int) : size of the chunk in bytes
    '''

    d = zlib.decompressobj(16+zlib.MAX_WBITS)

    with open(src_file_fullname,'rb') as src_f:
        buffer = src_f.read(chunk_size)

        with open(trgt_file_fullname,'wb') as trgt_f:
            with tqdm.tqdm( total=314395, ncols = 100 ) as pbar:
                while buffer:
                    trgt_f.write( d.decompress(buffer) )
                    buffer = src_f.read(chunk_size)
                    pbar.update(1)

                trgt_f.write( d.flush() )
                pbar.update(1)


#////////////////////////////////////////////////////////////////////////////////////


import requests
import math

def download_progress(src_file_url, trgt_file_fullname, chunk_size=4096) :
    '''
    downloads a file from the Internet into a local file,
    with progress bar.

    parameters :
      - src_file_url (string) : url to the source file
      - file_fullname (string) : full path to the local file
      - chunk_size (int) : size of the chunk in bytes
    '''

    try:
        response = requests.get(src_file_url, stream=True)
        total_length = response.headers.get('content-length')
        total_length = int(total_length)
        steps = math.ceil(total_length/chunk_size)

        with tqdm.tqdm( total=steps, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}' ) as pbar:
            with open(trgt_file_fullname,'wb') as f:
                for buffer in response.iter_content(chunk_size=chunk_size):
                    f.write( buffer )
                    pbar.update(1)
    except Exception as ex:
        raise


#////////////////////////////////////////////////////////////////////////////////////


import requests
import math

def download_ungzip_progress(src_file_url, trgt_file_fullname, chunk_size=4096) :
    '''
    downloads and decompresses a gzip file from the Internet into a local file,
    with progress bar.

    parameters :
      - src_file_url (string) : url to the source (compressed) file
      - file_fullname (string) : full path to the local (decompressed) file
      - chunk_size (int) : size of the chunk in bytes
    '''

    try:
        response = requests.get(src_file_url, stream=True)
        total_length = response.headers.get('content-length')
        total_length = int(total_length)
        steps = math.ceil(total_length/chunk_size)+1 # +1 for the last flush/update

        d = zlib.decompressobj(16+zlib.MAX_WBITS)

        with tqdm.tqdm( total=steps, ncols = 100 ) as pbar:
            with open(trgt_file_fullname,'wb') as f:
                for buffer in response.iter_content(chunk_size=chunk_size):
                    f.write( d.decompress(buffer) )
                    pbar.update(1)
                f.write( d.flush() )
                pbar.update(1)
    except Exception as ex:
        raise


#////////////////////////////////////////////////////////////////////////////////////
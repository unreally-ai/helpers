# 5kvocab.py
usage:
`python3 top5k.py <folder>`

> The folder should contain only datasets

The script will read every data frame in the folder and write a
csv with the top 5k most common words of all data frames combined
to the current directory.

IMPORTANT:
> Note that the row at which it looks for is currently hard-coded to 2.
> Furthermore, it only look for .txt files in the directory!

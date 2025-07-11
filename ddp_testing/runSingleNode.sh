mkdir -p data

python3 pems_ddp.py -m index -np 4 -g true > output.log 2>&1
mkdir PemsIndex
mv *.csv PemsIndex/
mv output.log PemsIndex/
rm flag.txt

python3 opt_baseline.py -m dask -np 4 > output.log 2>&1
mkdir PemsBase
mv *.csv PemsBase/
mv output.log PemsBase/
rm flag.txt
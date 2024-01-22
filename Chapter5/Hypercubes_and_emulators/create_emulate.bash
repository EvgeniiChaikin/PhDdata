#!/bin/bash


module unload python/2.7.15
module load python/3.10.1

Nruns=41

mkdir SFR_raw
mkdir SFR_processed

for i in $(seq 0 $Nruns) ; do 
  echo ${i} ;
  cp ${i}/SFR.txt  SFR_raw/SFR_${i}.txt ; 
done

python3.10  convert_SFH.py

mkdir emulate
mkdir emulate/data
mkdir emulate/data/data_z2p0
mkdir emulate/data/data_z0p0
mkdir emulate/params
mkdir emulate/plots

mv SFR_processed emulate/data/


for i in $(seq 0 $Nruns); do cp ${i}/data_0000.yml  emulate/data/data_z2p0/${i}.yml ; done
for i in $(seq 0 $Nruns); do cp ${i}/data_0001.yml  emulate/data/data_z0p0/${i}.yml ; done
for i in $(seq 0 $Nruns); do cp ${i}/${i}.yml  emulate/params/ ; done


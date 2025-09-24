mkdir ches_ctf
cd ches_ctf

wget https://zenodo.org/records/3733418/files/PinataAcqTask2.1_10k_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.2_10k_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.3_10k_upload.trs
wget https://zenodo.org/records/3733418/files/PinataAcqTask2.4_10k_upload.trs

python ../generate_ches_ctf.py
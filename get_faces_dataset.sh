mkdir 'dataset_faces'
cd ./dataset_faces
wget http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/zippedFaces.tar.gz .
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
tar -xf zippedFaces.tar.gz .
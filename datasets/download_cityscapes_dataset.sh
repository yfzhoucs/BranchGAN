URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/cityscapes.zip
ZIP_FILE=./datasets/cityscapes.zip
TARGET_DIR=./datasets/cityscapes/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE

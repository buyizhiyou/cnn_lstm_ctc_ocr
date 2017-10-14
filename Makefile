all: data-download data-tfrecord train

demo: train

data-download: data-wget data-unpack 

data-wget:
	mkdir -p data
	cd data ; \
	wget http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz

data-unpack:
	mkdir -p data/images
# strip leading mnt/ramdisk/max/90kDICT32px/
	tar xzvf data/mjsynth.tar.gz \
    --strip=4 \
    -C data/images

data-tfrecord:
	mkdir -p data/train data/val data/test 
	cd src ; python gen_tfrecord.py

train:
	cd src ; python train.py # use --help for options

monitor:
	tensorboard --logdir=data/model --port=8008

test:
	cd src ; python test.py # use --help for options

labels='apple book bowtie candle cloud cup door envelope eyeglasses guitar hammer hat leaf scissors star t-shirt pants lightning tree'

for value in $labels
do
	gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/$value.npy ./data/
done
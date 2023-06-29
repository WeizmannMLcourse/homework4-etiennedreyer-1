import os
import sys


def test():
	
	assert os.path.exists('trained_model.pt'), "Need to upload trained_model.pt"
	assert os.path.exists('MPNN_model.py'), "Need to upload MPNN_model.py"

	if not os.path.exists('test_data'):
		if not os.path.exists('test_data.tar'):
			os.system('wget https://www.dropbox.com/s/lt1luqhjludhdxe/test_data.tar')
		os.system('tar -xvf test_data.tar')

	path_to_ds = 'test_data/'

	from evaluate import evaluate_on_dataset

	f1, loss = evaluate_on_dataset(path_to_ds)

	assert (f1 > 0.8) , "F1 score needs to be above 0.8. You got {}".format(f1)
	 

if __name__ == '__main__':
	test()
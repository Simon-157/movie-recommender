import os
from urllib.request import urlretrieve
import zipfile
from tqdm import tqdm


def dowload_data():
	"""
	download the data
	Options: ml-1m, ml-10m, ml-latest
	"""
	data_name = 'ml-1m'
	save_path = 'movielens/'
	dowload_path = save_path + data_name + '.zip'
	url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if os.path.exists(dowload_path):
		print(f'{data_name} already exists')
	else:
		with DLProgress(unit='B', unit_scale=True, miniters=1, desc=f'Downloading {data_name}') as pbar:
			urlretrieve(url, dowload_path, pbar.hook)


def extract_data():
	data_name = 'ml-1m'
	data_path = 'movielens/ml-1m.zip'
	extract_path = 'movielens'
	if not os.path.exists(extract_path):
		os.makedirs(extract_path)
		unzip(data_name, data_path, extract_path)
	else:
		unzip(data_name, data_path, extract_path)
	print('extraction done')


def unzip(data_name, from_path, to_path):
	print(f'Extracting {data_name}...')
	with zipfile.ZipFile(from_path) as zf:
		zf.extractall(to_path)


class DLProgress(tqdm):
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		a hook function
		"""
		self.total=total_size
		self.update((block_num - self.last_block) * block_size)
		self.last_block = block_num


dowload_data()
extract_data()

from encoder import Model
from matplotlib import pyplot as plt
from utils import sst_binary, train_with_reg_cv
import numpy as np

LOAD_DATA = False

model = Model()

if LOAD_DATA is True:
	print('Loading labels...', end='')
	trY = np.load('data/processed/train_y.npy')
	vaY = np.load('data/processed/valid_y.npy')
	teY = np.load('data/processed/test_y.npy')
	print('Completed.')

	print('Loading inputs...', end='')
	trXt = np.load('data/processed/train_x.npy')
	vaXt = np.load('data/processed/valid_x.npy') 
	teXt = np.load('data/processed/test_x.npy')
	print('Completed.')

else:
	print('Fetching data... ', end='')
	trX, vaX, teX, trY, vaY, teY = sst_binary()
	print('Completed.')

	print('Saving processed labels...', end='')
	np.save('data/processed/train_y.npy', trY)
	np.save('data/processed/valid_y.npy', vaY)
	np.save('data/processed/test_y.npy', teY)
	print('Completed.')

	print('Transforming training data...')
	trXt = model.transform(trX)
	print('Saving... ', end='')
	np.save('data/processed/train_x.npy', trXt)
	print('Completed.')

	print('Transforming validation data...')
	vaXt = model.transform(vaX)
	print('Saving... ', end='')
	np.save('data/processed/valid_x.npy', vaXt)
	print('Completed.')

	print('Transforming testing data...')
	teXt = model.transform(teX)
	print('Saving... ', end='')
	np.save('data/processed/test_x.npy', teXt)
	print('Completed.')

print('Training classifier... ', end='')
# classification results
full_rep_acc, c, nnotzero = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
print('Completed.')

print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)

# visualize sentiment unit
sentiment_unit = trXt[:, 2388]
plt.hist(sentiment_unit[trY==0], bins=25, alpha=0.5, label='neg')
plt.hist(sentiment_unit[trY==1], bins=25, alpha=0.5, label='pos')
plt.legend()
plt.show()

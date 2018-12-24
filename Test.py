import configuration
from data_utils import get_data


path = configuration.config['paths']
print(path)
data, labels, word2idx = get_data(path)





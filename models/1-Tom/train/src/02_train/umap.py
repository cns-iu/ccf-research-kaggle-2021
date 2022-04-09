import umap
# import umap.umap_ as umap
import pickle


train_data = pickle.load(open("feature_train", "rb"))
test_data = pickle.load(open("feature_test", "rb"))

embedding = umap.UMAP().fit_transform(train_data)
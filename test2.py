# download CSV file
import tempfile
import urllib
import tensorflow as tf

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

# read CSV file into pandas dataframe
import pandas as pd 
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
			'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']

df_train = pd.read_csv(train_file,names = columns, index_col = False, skipinitialspace = True)
df_test = pd.read_csv(test_file, names = columns, index_col = False, skipinitialspace = True, skiprows = 1)

# show the scope and head of train and test dataframe
print df_train.shape, df_test.shape

# construct column called 'label' equal to 1 if income over 50k, and 0 otherwise
df_train['label'] = (df_train['income_bracket'].apply(lambda x: ">50K" in x)).astype(int)
df_test['label'] = (df_test['income_bracket'].apply(lambda x: ">50K" in x)).astype(int)
print df_train.head(), df_test.head()

# Group columns into categorical and continuous respectively
categorical = ["workclass", "education", "marital_status", "occupation",
				"relationship", "race", "gender", "native_country"]
continuous = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# convert data into tensors
def input_fn(df):
	# creats a dictionary mapping from each continuous feature column to the value of 
	# that column in a constant tensor.
	continuous_col = {k: tf.constant(df[k].values)
						for k in continuous}
	# create a dictionary mapping from each categorical feature column to the values of
	# that column in a tf.SparseTensor
	categorical_col = {k: tf.SparseTensor(
						indices = [[i,0] for i in range(df[k].size)],
						values = df[k].values,
						dense_shape = [df[k].size,1])
						for k in categorical}
	# merge two dictionaries together
	feature_cols = dict(continuous_col.items() + categorical_col.items())

	# convert label column into a constant tensor
	label = tf.constant(df['label'].values)

	# return the feature columns and label
	return feature_cols, label

def train_input_fn():
	return input_fn(df_train)


def test_input_fn():
	return input_fn(df_test)

# define sparse matrix for categorical variable using keys 
gender = tf.contrib.layers.sparse_column_with_keys(
			column_name = 'gender', keys = ['Female', 'Male'])


education = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = 'education', hash_bucket_size = 1000)
race = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = 'race', hash_bucket_size = 100)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = 'education_num', hash_bucket_size = 100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = 'relationship', hash_bucket_size = 100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = 'workclass', hash_bucket_size = 100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = 'occupation', hash_bucket_size = 1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket(column_name = 'native_country', hash_bucket_size = 1000)



# define realValuedColumn for each continuous variable
age = tf.contrib.layers.real_valued_column('age')
education_num = tf.contrib.layers.real_valued_column('education_num')
capital_gain = tf.contrib.layers.real_valued_column('capital_gain')
capital_loss = tf.contrib.layers.real_valued_column('capital_loss')
hours_per_week = tf.contrib.layers.real_valued_column('hours_per_week')

# making age variable categorical
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries = [18,25,30,35,40,45,50,55,60,65])

# interaction variables with CrossedColumn 
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation],hash_bucket_size = int(1e4))
age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size = int(1e6))

# defining logistic regression model
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns = [gender, native_country, education, occupation, workclass, marital_status,
	race, age_buckets, education_x_occupation, age_buckets_x_education_x_occupation], model_dir = model_dir)

# training model 
m.fit(input_fn = train_input_fn, steps = 200)

# evaluating model using test dataset
result = m.evaluate(input_fn = test_input_fn, steps = 1)

for key in sorted(result):
	print '%s: %s' %(key, result[key])









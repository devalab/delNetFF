class predict():

	def __init__(self , datafile , model , scale_file , label_file=None):

		self.n_atoms = 96
		self.data = None
		self.labels = None

		self.load_data(datafile , label_file)
		self.pre_process(scale_file)
		keras.losses.mse_error = mse_error
		self.model = load_model(model)
		self.predict_data(datafile)

	# def load_data(self , datafile , label_file=None):
	#
	# 	ext = datafile.split('.')[-1]
	#
	# 	if ext == 'npz':
	# 		with np.load(datafile) as fh:
	# 			self.data = fh['arr_0']
	# 			self.labels = fh['arr_1'] if 'arr_1' in fh.keys() else None ## We don't know if labels exist
	# 	else:
	# 		self.data = np.loadtxt(datafile)
	#
	# 	if label_file is not None:
	# 		self.labels = np.loadtxt(label_file)

	# def pre_process(self , scale_file):
	#
	# 	with np.load(scale_file) as fh:
	# 		self.min_vals = fh['arr_0']
	# 		self.max_vals = fh['arr_1']
	# 		self.fmean = fh['arr_2']
	# 		self.fstd = fh['arr_3']
	#
	# 	self.data = (self.data - self.fmean)/self.fstd

	def predict_data(self , datafile):

		output = self.model.predict(self.data)
		assert(output.shape[1]==3)

		output_scale = output*(self.max_vals - self.min_vals)/10 + self.min_vals
		self.score = r2_score(self.labels , output_scale) if self.labels is not None else None 
		print(self.score) ## Print r2-score
		output_scale = output_scale.astype('float64')

		out_file = 'forces.out'#%(datafile.split('.')[0]) ## Save in an out file
		np.savetxt(out_file , output_scale , fmt="%.9f" , delimiter=' ')

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Loads data from a file and predicts the using the given model.")
	parser.add_argument('-f','--file', dest='file' ,help="File from which to read the data")
	parser.add_argument('-m','--model' , dest='model' , help='The .h5 model saved by Keras used to predict the output')
	parser.add_argument('-s' , '--scale' , dest='scale' , help="The location of the file for scaling input data")
	parser.add_argument('-l' , '--label' , dest='label' , help="Label File. If given, then prints r2-score along with predictions")

	args = parser.parse_args()

	loadfile = args.file
	model = args.model
	scale_file = args.scale
	label_file = args.label

	pobj = predict(loadfile , model , scale_file , label_file)

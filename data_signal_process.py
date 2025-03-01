import pandas as pd
import numpy as np
import text
import csv
import SomeFns

class Data_Generator:
	name_of_the_file = ""
	log_file = ""
	source_of_data= ""
	num_of_records = 35
	num_of_channels= 14
	"""
	selective_channels option will control what is the channels you want to include in your dataset
	"""
	selective_channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	#selective_channels=[2,3,6,7,8,9,14]
	sf=128
	state=1
	def __init__(self,name="",log_file_name="logs"):
		self.make_new_csv_file_for_data(name)
		self.open_log_file_for_writing(log_file_name)

	def make_new_csv_file_for_data(self,name):
		"""
		    This Function just make new file with the file name to put data in it
		    make the first row define what we have in each coloum
		    make sure theirs no file has the same name that you input it
		"""
		self.name_of_the_file='../'+name+'.csv'
		"""
		    for old approach:
		    first_row = ["delta", "theta", "alpha", "beta", "gamma", "state"]    
		"""

		first_row = ["delta ch(1)", "theta ch(1)", "alpha ch(1)", "beta ch(1)", "gamma ch(1)",
			     "delta ch(2)", "theta ch(2)", "alpha ch(2)", "beta ch(2)", "gamma ch(2)",
			     "delta ch(3)", "theta ch(3)", "alpha ch(3)", "beta ch(3)", "gamma ch(3)",
			     "delta ch(4)", "theta ch(4)", "alpha ch(4)", "beta ch(4)", "gamma ch(4)",
			     "delta ch(5)", "theta ch(5)", "alpha ch(5)", "beta ch(5)", "gamma ch(5)",
			     "delta ch(6)", "theta ch(6)", "alpha ch(6)", "beta ch(6)", "gamma ch(6)",
			     "delta ch(7)", "theta ch(7)", "alpha ch(7)", "beta ch(7)", "gamma ch(7)",
			     "delta ch(8)", "theta ch(8)", "alpha ch(8)", "beta ch(8)", "gamma ch(8)",
			     "delta ch(9)", "theta ch(9)", "alpha ch(9)", "beta ch(9)", "gamma ch(9)",
			     "delta ch(10)", "theta ch(10)", "alpha ch(10)", "beta ch(10)", "gamma ch(10)",
			     "delta ch(11)", "theta ch(11)", "alpha ch(11)", "beta ch(11)", "gamma ch(11)",
			     "delta ch(12)", "theta ch(12)", "alpha ch(12)", "beta ch(12)", "gamma ch(12)",
			     "delta ch(13)", "theta ch(13)", "alpha ch(13)", "beta ch(13)", "gamma ch(13)",
			     "delta ch(14)", "theta ch(14)", "alpha ch(14)", "beta ch(14)", "gamma ch(14)", "state"]

		with open(self.name_of_the_file, 'w+') as csvFile:
		    w = csv.writer(csvFile)
		    w.writerow(first_row)
		csvFile.close()

	def open_log_file_for_writing(self, name):
		"""
		    This function is just to open the log file to write some logs
		"""
		self.log_file = open(name+".txt", "w+")

	def Read_Data_from_Folder(self, name):
		"""
		    This Function is only to define the path of your data to read and extract
		    the features from it
		"""
		self.source_of_data=name

	def Write_DataSample_in_The_CSV_file(self, datasample):
		"""
		    This Function is just to add new datasample in your dataSet file
		"""
		with open(self.name_of_the_file, 'a') as csvFile:
		    w = csv.writer(csvFile)
		    w.writerow(datasample)
		csvFile.close()

	def Generate_Data(self,s=1,name="eeg_record"):
		"""
		    This Function is mainly the part responsible for generate the DataSet by
		    extracting features from the dataset
		    The name represent the name of the files that contain the data
		    we will use the state for know how many samples to get from the data
		"""
		self.state=s
		for file_number in range(1, self.num_of_records):
		    """
			This loop just loop on all the records and read the data in array
		    """
		    name_of_file_to_read=self.source_of_data+'/'+name+str(file_number)+'.csv'
		    self.log_file.write("Reading now ===>> "+name_of_file_to_read+"\n")
		    loadData = pd.read_csv( name_of_file_to_read )
		    data = np.array(loadData)
		    self.Divide_Data_the_generate(data)

	def Divide_Data_the_generate(self,data):
		"""
		    Assume We have a Record of length like This
			x x x x x x x x x
		    We Assume that the parts that we shoud study should be these parts
			x   x x   x x   x    <<<==== Ignore these
			  x     x     x      <<<==== Study This
		"""
		first_channel = data[:, 1]
		any_channel_length = len(first_channel)
		A = int(any_channel_length / 3)
		B = int( A / 3)
		"""
		    A ==> divide the whole record to 3 parts
		    B ==> divide each part to 3 parts to take the one in the middle
		"""
		first_region_dimentions = [ B     , 2*B    ]
		second_region_dimentions= [ A+B   , A+2*B  ]
		third_region_dimentions = [ 2*A+B , 2*A+2*B]
		"""
		    We have 2 options the first one to get one data sample from each region and the second is to 
		    get 3 data sample from each region
		"""
		if(self.state==1):
		    self.Get_One_Sample_each_region(data,first_region_dimentions ,second_region_dimentions,third_region_dimentions)
		elif(self.state==5):
		    self.Get_Five_Sample_each_region(data, first_region_dimentions, second_region_dimentions,third_region_dimentions)


	def Get_One_Sample_each_region( self, data,first_region ,second_region ,third_region):
		"""
		    This function is to loop in each region and get data sample in each one
		"""
		first_data_sample= self.Get_data_Sample_from_region(data,first_region,1)
		second_data_sample = self.Get_data_Sample_from_region(data, second_region,2)
		third_data_sample = self.Get_data_Sample_from_region(data, third_region,3)
		self.Write_DataSample_in_The_CSV_file(first_data_sample)
		self.Write_DataSample_in_The_CSV_file(second_data_sample)
		self.Write_DataSample_in_The_CSV_file(third_data_sample)

	def Get_Five_Sample_each_region( self, data,first_region ,second_region ,third_region):
		"""
		    This function is to loop in each region and get 5 data samples in each one
		    Assume we have each region like this:
			xxxxx
		    we need to deal with each part alone like this:
			x x x x x
		    This is not the best implementation for something like this function But this
		    just to avoid any mistakes for now
		"""
		#Divide 1st part to 5 parts
		len = first_region[1]-first_region[0]+1
		A = int(len/5)
		region_1 = [first_region[0]     , first_region[0]+A]
		region_2 = [first_region[0]+A   , first_region[0]+2*A]
		region_3 = [first_region[0]+2*A , first_region[0]+3*A]
		region_4 = [first_region[0]+3*A , first_region[0]+4*A]
		region_5 = [first_region[0]+4*A , first_region[1]]
		first_data_sample_1 = self.Get_data_Sample_from_region(data, region_1, 1)
		first_data_sample_2 = self.Get_data_Sample_from_region(data, region_2, 1)
		first_data_sample_3 = self.Get_data_Sample_from_region(data, region_3, 1)
		first_data_sample_4 = self.Get_data_Sample_from_region(data, region_4, 1)
		first_data_sample_5 = self.Get_data_Sample_from_region(data, region_5, 1)
		self.Write_DataSample_in_The_CSV_file(first_data_sample_1)
		self.Write_DataSample_in_The_CSV_file(first_data_sample_2)
		self.Write_DataSample_in_The_CSV_file(first_data_sample_3)
		self.Write_DataSample_in_The_CSV_file(first_data_sample_4)
		self.Write_DataSample_in_The_CSV_file(first_data_sample_5)

		# Divide 2nd part to 5 parts
		len = second_region[1] - second_region[0] + 1
		A = int(len/5)
		region_1 = [second_region[0]	 , second_region[0]+A]
		region_2 = [second_region[0]+A	 , second_region[0]+2*A]
		region_3 = [second_region[0]+2*A , second_region[0]+3*A]
		region_4 = [second_region[0]+3*A , second_region[0]+4*A]
		region_5 = [second_region[0]+4*A , second_region[1]]
		second_data_sample_1 = self.Get_data_Sample_from_region(data, region_1, 2)
		second_data_sample_2 = self.Get_data_Sample_from_region(data, region_2, 2)
		second_data_sample_3 = self.Get_data_Sample_from_region(data, region_3, 2)
		second_data_sample_4 = self.Get_data_Sample_from_region(data, region_4, 2)
		second_data_sample_5 = self.Get_data_Sample_from_region(data, region_5, 2)
		self.Write_DataSample_in_The_CSV_file(second_data_sample_1)
		self.Write_DataSample_in_The_CSV_file(second_data_sample_2)
		self.Write_DataSample_in_The_CSV_file(second_data_sample_3)
		self.Write_DataSample_in_The_CSV_file(second_data_sample_4)
		self.Write_DataSample_in_The_CSV_file(second_data_sample_5)

		# Divide 3rd part to 5 parts
		len = third_region[1] - third_region[0] + 1
		A = int(len/5)
		region_1 = [third_region[0]	, third_region[0]+A]
		region_2 = [third_region[0]+A	, third_region[0]+2*A]
		region_3 = [third_region[0]+2*A , third_region[0]+3*A]
		region_4 = [third_region[0]+3*A , third_region[0]+4*A]
		region_5 = [third_region[0]+4*A , third_region[1]]
		third_data_sample_1 = self.Get_data_Sample_from_region(data, region_1, 3)
		third_data_sample_2 = self.Get_data_Sample_from_region(data, region_2, 3)
		third_data_sample_3 = self.Get_data_Sample_from_region(data, region_3, 3)
		third_data_sample_4 = self.Get_data_Sample_from_region(data, region_4, 3)
		third_data_sample_5 = self.Get_data_Sample_from_region(data, region_5, 3)
		self.Write_DataSample_in_The_CSV_file(third_data_sample_1)
		self.Write_DataSample_in_The_CSV_file(third_data_sample_2)
		self.Write_DataSample_in_The_CSV_file(third_data_sample_3)
		self.Write_DataSample_in_The_CSV_file(third_data_sample_4)
		self.Write_DataSample_in_The_CSV_file(third_data_sample_5)

	def Get_data_Sample_from_region( self, data, region , state):
		"""
		    This function just take the region of the datasample and it loop on all the channels to calculate the
		    avg power from all of them
		"""
		data_sample=[0,0,0,0,0,state]
		"""
		    Use this line if you want to take all channels
		    #for channel in range(0, self.num_of_channels):
		"""
		new_data_sample=[]
		for channel in self.selective_channels:
		    data_part=data[region[0]:region[1], (channel-1)]
		    # Define the duration of the window to be 4 seconds
		    win_sec = 4
		    # Compute average absolute power of Delta band
		    delta_power = SomeFns.bandpower(data_part, self.sf, [0.5, 4], win_sec)
		    data_sample[0]+=delta_power
		    new_data_sample.append(delta_power)
		    # Compute average absolute power of Theta band
		    theta_power = SomeFns.bandpower(data_part, self.sf, [4, 8], win_sec)
		    data_sample[1] +=theta_power
		    new_data_sample.append(theta_power)
		    # Compute average absolute power of Alpha band
		    alpha_power = SomeFns.bandpower(data_part, self.sf, [8, 12], win_sec)
		    data_sample[2] +=alpha_power
		    new_data_sample.append(alpha_power)
		    # Compute average absolute power of Beta band
		    beta_power = SomeFns.bandpower(data_part, self.sf, [12, 30], win_sec)
		    data_sample[3] +=beta_power
		    new_data_sample.append(beta_power)
		    # Compute average absolute power of Gamma band
		    gamma_power = SomeFns.bandpower(data_part, self.sf, [30, 100], win_sec)
		    data_sample[4] +=gamma_power
		    new_data_sample.append(gamma_power)
		for feature in range(0, 5):
		    data_sample[feature]=data_sample[feature]/self.num_of_channels
		"""
		    if you want the old approach of 5 numbers per each datasample use this line
		    #return data_sample
		"""
		new_data_sample.append(state)
		return new_data_sample

def main():
	#create file to put the data in it
	Data_Gen = Data_Generator("data_set_file")
	Data_Gen.Read_Data_from_Folder("EEG DATA CSV/")
	"""
	The number that next function takes determines tha number of datasambles taken from the same record 
	"""
	Data_Gen.Generate_Data(5)

if __name__ == "__main__":
    main()
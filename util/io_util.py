import csv
import os
class io():
	def __init__(self):
		headers = ["running.batch_size", "common.history_seq_len", "running.lr", "full_baseline_lstm_hidden", "full_baseline_lstm_layer",
				   "full_baseline_fc_hidden",'time_baseline_input','time_baseline_lstm_hidden',
				   'time_baseline_lstm_layer','time_baseline_adapter_out','date_baseline_input',
				   'date_baseline_lstm_hidden','date_baseline_lstm_layer','date_baseline_adapter_out',
				   'local_cnn_channel1','time_k_q',"date_k_q",
				   'r2','evs','mae','mse','rmse','mape','time']
		#rows = [("1001", "王小一", 18, "西三旗1号院", "50000"), ("1001", "王小二", 19, "西三旗1号院", "30000")]
		with open(r"d:\b.csv", "w") as b:
			b_csv = csv.writer(b)
			b_csv.writerow(headers)
			#b_csv.writerows(rows)

if __name__ == "__main__":
	pass




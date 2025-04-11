import pathlib
import argparse
import time
import sys
import string
import socket
import multiprocessing

import bmemcached

LOG_GRANULARITY = 1

def set_key(cli, key, valsize):
	val = b'x' * valsize
	start_time = time.time()
	cli.set(key, (b'x' * valsize))
	return time.time() - start_time

def get_key(cli, key):
	start_time = time.time()
	result = cli.get(key)
	latency = time.time() - start_time
	return (result is not None, latency)

def parse_trace_file(trace_filepath, num_workers):
	print("partitioning workloads...")
	workloads = [[] for _ in range(num_workers)]
	line_i = 0
	with open(trace_filepath) as trace:
		for line in trace:
			line_i += 1
			# parse the line
			ts, obj_id, obj_size, latency = line.split()
			#obj_size = int(obj_size) // 100
			obj_size = 4096
			workloads[line_i % num_workers].append((obj_id, obj_size))
	return workloads
		
		

def run_trace(cli, workload, worker_id, num_workers, start_time, experiment_name=None, log_folder=None):
	print(f"worker {worker_id}/{num_workers} starting")
	# list of {ts:x, curr throughput:y, curr latency: z, overall throughput: a, overall latency: b}
	my_log = []

	num_accesses = 0
	num_hits = 0
	curr_hits = 0

	# each second, store the throughput and avg latency in that last second
	next_benchmark_second = time.time() + LOG_GRANULARITY

	last_sec_req_processed = 0
	last_sec_total_latency = 0
	total_req_processed = 0
	total_latency = 0

	for obj_id, obj_size in workload:
		# for each (timestamp, obj_id, obj_size, latency):
		if time.time() >= next_benchmark_second:
			# report num req
			last_sec_throughput = last_sec_req_processed / LOG_GRANULARITY
			last_sec_latency = 1000 * last_sec_total_latency / last_sec_req_processed

			overall_throughput = total_req_processed / (time.time() - start_time)
			overall_latency = 1000 * total_latency / total_req_processed
			print(f"{worker_id}: last second latency={last_sec_latency:.5f}ms throughput={last_sec_throughput:.2f} req/s")
			print(f"{worker_id}: overall latency={overall_latency:.5f}ms throughput={overall_throughput:.2f} req/s")

			my_log.append({
				"timestamp": time.time() - start_time,
				"overall_latency": overall_latency,
				"overall_throughput": overall_throughput,
				"last_sec_latency": last_sec_latency,
				"last_sec_throughput": last_sec_throughput,
			})

			last_sec_req_processed = 0
			last_sec_total_latency = 0
			next_benchmark_second = time.time() + LOG_GRANULARITY


		# get it from cache
		# if hit -> do nothing
		# if miss -> set it to a predetermined value
		# track hit rate
		key_hit, req_latency = get_key(cli, obj_id)
		if not key_hit:
			# for now this is going to take some time to generate the strings, but for the purpose of hit rate measurement it's fine
			req_latency += set_key(cli, obj_id, obj_size)
		else:
			num_hits += 1
			curr_hits += 1

		num_accesses += 1
		last_sec_req_processed += 1
		total_req_processed += 1
		last_sec_total_latency += req_latency
		total_latency += req_latency

		PRINT_ITER = 10000
		if num_accesses % PRINT_ITER == 0:
			#print(f"{worker_id}: req {num_accesses}: CURR HIT RATE: {curr_hits/PRINT_ITER*100:.2f}% GLOBAL HIT RATE: {num_hits/num_accesses*100:.2f}%")
			curr_hits = 0
		#time.sleep(0.01)
	
	# now we finished experiment, commit logs to files
	if log_folder is None:
		return

	log_filepath = log_folder / f"{experiment_name}_{worker_id}.csv"
	with open(log_filepath, "w") as log_file:
		log_file.write("timestamp,last_second_latency,last_second_throughput,overall_latency,overall_throughput\n")
		for log_entry in my_log:
			ts = log_entry["timestamp"]
			curr_lat = log_entry["last_sec_latency"]
			curr_thr = log_entry["last_sec_throughput"]
			tot_lat = log_entry["overall_latency"]
			tot_thr = log_entry["overall_throughput"]
			log_file.write(f"{ts},{curr_lat},{curr_thr},{tot_lat},{tot_thr}\n")

def main():
	parser = argparse.ArgumentParser(prog="Memcached benchmark")
	# get the host, port, num_workers, trace file
	parser.add_argument("-H", "--host", required=True)
	parser.add_argument("-p", "--port", required=True)
	parser.add_argument("-n", "--num-workers", required=True)
	parser.add_argument("-t", "--trace-file", required=True)
	parser.add_argument("-N", "--name", default="exp")
	parser.add_argument("-l", "--log-folder")

	args = parser.parse_args()
	port_num = int(args.port)
	host = args.host
	num_workers = int(args.num_workers)
	trace_filename = args.trace_file
	experiment_name = args.name
	log_folder = args.log_folder

	print(f"starting experiment with memcached instance {host}:{port_num} with {num_workers} workers on trace {trace_filename}")

	# make sure the trace exists
	trace_path = pathlib.Path(trace_filename)
	if not trace_path.exists():
		print(f"{trace_path} does not exist, exiting")

	# if the log_folder is given, let's make it
	if log_folder is not None:
		log_folder = pathlib.Path(log_folder)
		log_folder.mkdir(exist_ok=True)

	# TODO: make sure the memcached instance is alive

	# parse the requests from the file, and split them up into the workload for each worker
	worker_workloads = parse_trace_file(trace_path, num_workers)

	procs = []
	clis = []
	start_time = time.time()
	for worker_id in range(num_workers):
		cli = bmemcached.Client((f"{host}:{port_num}"))
		clis.append(cli)
		p = multiprocessing.Process(target=run_trace, args=(clis[worker_id], worker_workloads[worker_id], worker_id, num_workers, start_time, experiment_name, log_folder))
		procs.append(p)

	for worker in procs:
		worker.start()
	for worker in procs:
		worker.join()

if __name__ == "__main__":
	main()

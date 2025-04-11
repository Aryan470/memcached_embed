import time
import sys
import string
import socket
import multiprocessing

import bmemcached

DEBUG = False

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

def run_trace(cli, worker_id, num_workers):
	print(f"worker {worker_id}/{num_workers} starting")

	start_time = time.time()
	num_accesses = 0
	num_hits = 0
	curr_hits = 0
	trace_fname = "/users/aryankh/wiki_cache_00.trace"
	print(f"running experiment with {trace_fname}")
	req_num = 1

	# each second, store the throughput and avg latency in that last second
	next_benchmark_second = start_time + 3

	last_sec_req_processed = 0
	last_sec_total_latency = 0
	total_req_processed = 0
	total_latency = 0

	with open(trace_fname) as trace_file:
		# for each (timestamp, obj_id, obj_size, latency):
		req_i = 0
		for line in trace_file:
			req_i += 1
			if time.time() >= next_benchmark_second:
				# report num req
				last_sec_throughput = last_sec_req_processed / 3
				last_sec_latency = 1000 * last_sec_total_latency / last_sec_req_processed

				overall_throughput = total_req_processed / (time.time() - start_time)
				overall_latency = 1000 * total_latency / total_req_processed
				print(f"{worker_id}: last second latency={last_sec_latency:.5f}ms throughput={last_sec_throughput:.2f} req/s")
				print(f"{worker_id}: overall latency={overall_latency:.5f}ms throughput={overall_throughput:.2f} req/s")

				last_sec_req_processed = 0
				last_sec_total_latency = 0
				next_benchmark_second = time.time() + 3
			if req_i % num_workers != worker_id:
				continue

			ts, obj_id, obj_size, latency = line.split()
			#obj_id = int(obj_id)
			obj_size = int(obj_size) // 100
			obj_size = 4096
			# get it from cache
			# if hit -> do nothing
			# if miss -> set it to a predetermined value
			# track hit rate
			num_accesses += 1

			last_sec_req_processed += 1
			total_req_processed += 1

			key_hit, req_latency = get_key(cli, obj_id)
			if not key_hit:
				# for now this is going to take some time to generate the strings, but for the purpose of hit rate measurement it's fine
				req_latency += set_key(cli, obj_id, obj_size)
			else:
				num_hits += 1
				curr_hits += 1

			last_sec_total_latency += req_latency
			total_latency += req_latency

			PRINT_ITER = 10000
			if req_num % PRINT_ITER == 0:
				#print(f"{worker_id}: req {req_num}: CURR HIT RATE: {curr_hits/PRINT_ITER*100:.2f}% GLOBAL HIT RATE: {num_hits/num_accesses*100:.2f}%")
				curr_hits = 0
			req_num += 1
			#time.sleep(0.01)

def debug_print(text):
	if DEBUG:
		print(text)

def main():
	port_num = int(sys.argv[1])
	host = '127.0.0.1'

	procs = []
	clis = []
	num_workers = 32
	for worker_id in range(num_workers):
		cli = bmemcached.Client((f"{host}:{port_num}"))
		clis.append(cli)
		p = multiprocessing.Process(target=run_trace, args=(clis[worker_id], worker_id, num_workers))
		procs.append(p)

	for worker in procs:
		worker.start()
	for worker in procs:
		worker.join()

if __name__ == "__main__":
	main()

import time
import sys
import string
import socket

import bmemcached

DEBUG = False

def send_command(s, cmd):
    # Send the command as bytes
    s.sendall(cmd.encode('utf-8'))
    # Read response until we see a termination marker in the accumulated buffer
    buffer = b""
    start_time = time.time()
    while True:
        data = s.recv(4096)
        if not data:
            break
        buffer += data
        # Check the entire accumulated buffer for one of the markers.
        # Adjust the marker check depending on the command type.
        if b'END\r\n' in buffer or b'STORED\r\n' in buffer or b'NOT_STORED\r\n' in buffer or buffer.startswith(b'SERVER_ERROR'):
            break

        # Optionally, add a timeout condition that eventually breaks out.
        if time.time() - start_time > 1:
            print("TIMEOUT")
            break

    return buffer.decode('utf-8')

def set_key(cli, key, valsize):
	cli.set(key, (b'x' * valsize))

def get_key(cli, key):
	return cli.get(key) is not None

def raw_set_key(sock, key, valsize):
	#print(f"sending SET for key {key} value size {valsize}")
	command = f"set {key} 0 1000 {valsize}\r\n"
	value = key[0] * valsize
	value += "\r\n"
	response = send_command(sock, command + value)
	debug_print(f"sent SET for key {key} value size {valsize}")
	debug_print(response.strip())
	get_key(sock, key)

def raw_get_key(sock, key):
	#print(f"sending GET for key {key}")
	command = f"get {key}\r\n"
	response = send_command(sock, command).strip()
	status = "HIT" if response.startswith("VALUE") else "MISS"

	debug_print(f"sent GET for key {key}: {status}")
	return status == "HIT"
	#print(response.strip())
	

def test_eviction(sock):
	key_list = string.ascii_lowercase + string.ascii_uppercase + string.digits
	# add a bunch of keys
	for key in key_list:
		valsize = 100000
		set_key(sock, key, valsize)
		debug_print()

	
	# access some
	num_hits = 0
	num_accesses = 0
	for key in key_list + key_list[::-1]:
		num_accesses += 1
		if get_key(sock, key):
			num_hits += 1
		else:
			set_key(sock, key, valsize)

		debug_print()

	print(f"HIT RATE: {num_hits/num_accesses*100:.2f}%")

def run_trace(cli):
	start_time = time.time()
	num_accesses = 0
	num_hits = 0
	curr_hits = 0
	trace_fname = "/users/aryankh/wiki_cache_00.trace"
	print(f"running experiment with {trace_fname}")
	req_num = 1
	# each second, store the throughput and avg latency in that last second
	with open(trace_fname) as trace_file:
		# for each (timestamp, obj_id, obj_size, latency):
		for line in trace_file:
			ts, obj_id, obj_size, latency = line.split()
			#obj_id = int(obj_id)
			obj_size = int(obj_size) // 100
			obj_size = 4096
			# get it from cache
			# if hit -> do nothing
			# if miss -> set it to a predetermined value
			# track hit rate
			num_accesses += 1
			if not get_key(cli, obj_id):
				# for now this is going to take some time to generate the strings, but for the purpose of hit rate measurement it's fine
				set_key(cli, obj_id, obj_size)
			else:
				num_hits += 1
				curr_hits += 1

			PRINT_ITER = 10000
			if req_num % PRINT_ITER == 0:
				print(f"req {req_num}: CURR HIT RATE: {curr_hits/PRINT_ITER*100:.2f}% GLOBAL HIT RATE: {num_hits/num_accesses*100:.2f}%")
				curr_hits = 0
			req_num += 1

def debug_print(text):
	if DEBUG:
		print(text)

def main():
	port_num = int(sys.argv[1])
	host = '127.0.0.1'
	cli = bmemcached.Client((f"{host}:{port_num}"))
	run_trace(cli)
	cli.close()


if __name__ == "__main__":
	main()

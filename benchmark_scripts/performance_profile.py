import subprocess
TRACE_FILE = "/users/aryankh/wiki_cache_00.trace"
LOG_FOLDER = "/users/aryankh/memcached_embed/logs/"
CLIENT_SCRIPT = "/users/aryankh/memcached_embed/benchmark_scripts/multiproc_test.py"

EXECUTABLE_PATH = {
	"EMB": "/users/aryankh/memcached_embed/memcached_emb",
	"LRU": "/users/aryankh/memcached_embed/memcached_lru",
}

def launch_memcached(executable, num_server_workers, memory_limit=12, port=11211):
	return subprocess.Popen(
		[EXECUTABLE_PATH[executable], "-p", str(port), "-m", str(memory_limit), "-t", str(num_server_workers)],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)

def end_memcached(proc):
	proc.terminate()
	try:
		exit_code = proc.wait(timeout=10)
		return exit_code == 0
	except subprocess.TimeoutExpired:
		proc.kill()
		return False

def run_clients(num_client_workers, experiment_name):
	return subprocess.run(
		["python3", CLIENT_SCRIPT, "-H", "127.0.0.1", "-p", str(11211), "-n", str(num_client_workers), "-t", TRACE_FILE, "-N", experiment_name, "-l", LOG_FOLDER],
		capture_output=True,
		text=True
	)
	

def run_experiment(executable, num_server_workers, num_client_workers):
	print(f"running experiment with executable={executable}, num_srv={num_server_workers}, num_cli={num_client_workers}")
	# launch the memcached instance
	print(f"starting up {executable} with {num_server_workers} threads")
	memcached_proc = launch_memcached(executable, num_server_workers)
	# launch the client experiment	
	experiment_name = f"{executable}_srv_{num_server_workers}_cli_{num_client_workers}"
	# starting client experiment
	result = run_clients(num_client_workers, experiment_name)
	if result.returncode == 0:
		print("clients finished successfully")
	else:
		raise ValueError("clients failed!")

	# kill it
	print(f"terminating {executable} with {num_server_workers} threads")
	if end_memcached(memcached_proc):
		print("terminated gracefully")
	else:
		raise ValueError("ERROR in memcached")


def main():
	# requirements: stop and start memcached instances, detect abormal exits
	# once we have run all the experiments, generate associated plots

	# for each policy in [memcached_lru, memcached_emb]
	# for each num of worker threads in [1, 2, 4, 8, 16, 32]
	# for each num of worker clients in [1, 2, 4, 8, 16, 32]
	# launch memcached instance
	# launch multiproc experiment with
	# ./multiproc_exp -H 127.0.0.1 -p 11211 -n {num_cli_work} -t trace -N {policy}_{num_cli}_clients_{num_work}_workers -l logs

	for num_server_workers in [1, 2, 4, 8, 16]:
		for num_client_workers in [1, 2, 4, 8, 16]:
			for executable in ["LRU", "EMB"]:
				run_experiment(executable, num_server_workers, num_client_workers)


if __name__ == "__main__":
	main()

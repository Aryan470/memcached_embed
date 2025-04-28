// bench.cpp
#include <libmemcached/memcached.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// logging granularity in seconds
constexpr int LOG_GRANULARITY = 1;

struct Req {
    std::string key;
    size_t      val_size;
};

struct LogEntry {
    double timestamp;
    double last_latency_ms;
    double last_throughput;
    double last_hit_rate;       // NEW
    double overall_latency_ms;
    double overall_throughput;
    double overall_hit_rate;    // NEW
};

static std::mutex io_mtx;

void run_worker(
    const std::string& host,
    int                port,
    int                worker_id,
    int                num_workers,
    const std::vector<Req>& workload,
    double             start_time,
    const std::string& experiment_name,
    const std::string& log_folder
) {
    // 1) Setup libmemcached client
    memcached_st* memc = memcached_create(nullptr);
    auto* servers = memcached_server_list_append(nullptr, host.c_str(), port, nullptr);
    memcached_server_push(memc, servers);
    memcached_server_list_free(servers);

    // Per‑thread stats
    size_t last_sec_reqs = 0;
    size_t last_sec_hits = 0;        // NEW
    double last_sec_lat  = 0.0;

    size_t total_reqs    = 0;
    size_t total_hits    = 0;        // NEW
    double total_lat     = 0.0;

    double next_log_time = start_time + LOG_GRANULARITY;

    std::vector<LogEntry> logs;
    logs.reserve(workload.size() / 1000);

    {
        std::lock_guard<std::mutex> lk(io_mtx);
        std::cout << "[W" << worker_id << "] Starting, "
                  << workload.size() << " requests" << std::endl;
    }

    for (const auto& r : workload) {
        // 2) GET
        auto t0 = std::chrono::steady_clock::now();
        size_t    value_length = 0;
        // pass nullptr for flags if you don't need them
        memcached_return rc;
        char* val = memcached_get(
            memc,
            r.key.data(), r.key.size(),
            &value_length,
            nullptr,
            &rc
        );
        double req_lat = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0
        ).count();

        bool hit = (rc == MEMCACHED_SUCCESS);
        if (hit) {
            ++total_hits;    // NEW
            ++last_sec_hits; // NEW
        } else {
            // on miss, SET
            auto t1 = std::chrono::steady_clock::now();
            memcached_set(
                memc,
                r.key.data(), r.key.size(),
                std::string(r.val_size, 'x').data(), r.val_size,
                (time_t)0, (uint32_t)0
            );
            req_lat += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t1
            ).count();
        }

        if (val) free(val);

        // update counters
        ++total_reqs;
        ++last_sec_reqs;
        total_lat    += req_lat;
        last_sec_lat += req_lat;

        // 3) time to log?
        double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();

        if (now >= next_log_time) {
            double elapsed = now - start_time;

            double last_thr   = last_sec_reqs / (double)LOG_GRANULARITY;
            double last_lat_ms= last_sec_lat / last_sec_reqs * 1e3;
            double last_hr    = last_sec_reqs
                                ? (100.0 * last_sec_hits / last_sec_reqs)
                                : 0.0;                           // NEW

            double overall_thr= total_reqs / elapsed;
            double overall_lat_ms = total_lat / total_reqs * 1e3;
            double overall_hr  = total_reqs
                                 ? (100.0 * total_hits / total_reqs)
                                 : 0.0;                         // NEW

            {
                std::lock_guard<std::mutex> lk(io_mtx);
                std::cout
                    << "[W" << worker_id << "] "
                    << "last1s: lat="  << last_lat_ms  << "ms "
                    << "thr=" << last_thr    << "r/s "
                    << "hit=" << last_hr     << "%\n"
                    << "[W" << worker_id << "] "
                    << " overall: lat=" << overall_lat_ms << "ms "
                    << "thr="  << overall_thr    << "r/s "
                    << "hit="  << overall_hr     << "%" << std::endl;
            }

            logs.push_back({
                elapsed,
                last_lat_ms,
                last_thr,
                last_hr,               // NEW
                overall_lat_ms,
                overall_thr,
                overall_hr             // NEW
            });

            // reset per‑second counters
            last_sec_reqs = 0;
            last_sec_hits = 0;         // NEW
            last_sec_lat  = 0.0;
            next_log_time += LOG_GRANULARITY;
        }
    }

    // 4) flush CSV if requested
    if (!log_folder.empty()) {
        std::filesystem::create_directories(log_folder);
        auto fn = log_folder + "/" + experiment_name
                  + "_" + std::to_string(worker_id) + ".csv";
        std::ofstream out(fn);
        out << "timestamp,"
               "last_latency_ms,last_throughput,last_hit_rate,"
               "overall_latency_ms,overall_throughput,overall_hit_rate\n";
        for (auto& e : logs) {
            out
              << e.timestamp << ','
              << e.last_latency_ms << ','
              << e.last_throughput << ','
              << e.last_hit_rate << ','
              << e.overall_latency_ms << ','
              << e.overall_throughput << ','
              << e.overall_hit_rate << "\n";
        }
    }

    memcached_free(memc);
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    // make cout auto-flush each << insertion
    std::cout.setf(std::ios::unitbuf);

    std::string host, trace_file, experiment_name="exp", log_folder;
    int port=0, num_workers=0;

    // very minimal arg parsing
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a=="-H"||a=="--host") && i+1<argc) host = argv[++i];
        else if ((a=="-p"||a=="--port") && i+1<argc) port = std::stoi(argv[++i]);
        else if ((a=="-n"||a=="--num-workers") && i+1<argc)
            num_workers = std::stoi(argv[++i]);
        else if ((a=="-t"||a=="--trace-file") && i+1<argc)
            trace_file = argv[++i];
        else if ((a=="-N"||a=="--name") && i+1<argc)
            experiment_name = argv[++i];
        else if ((a=="-l"||a=="--log-folder") && i+1<argc)
            log_folder = argv[++i];
        else {
            std::cerr << "Unknown arg: " << a << "\n";
            return 1;
        }
    }
    if (host.empty()||port<=0||num_workers<=0||trace_file.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " -H host -p port -n num-workers -t trace-file "
                     "[-N name] [-l log-folder]\n";
        return 1;
    }

    // 5) load & partition trace
    std::vector<std::vector<Req>> workloads(num_workers);
    {
        std::ifstream in(trace_file);
        std::string ts, key, size_s, lat;
        size_t lineno = 0;
        while (in >> ts >> key >> size_s >> lat) {
            ++lineno;
            workloads[lineno % num_workers].push_back({ key, 4096 });
        }
    }

    double start_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();

    // 6) spawn threads
    std::vector<std::thread> threads;
    threads.reserve(num_workers);
    for (int w = 0; w < num_workers; ++w) {
        threads.emplace_back(
            run_worker,
            host, port, w, num_workers,
            std::cref(workloads[w]),
            start_time,
            experiment_name,
            log_folder
        );
    }
    for (auto& t : threads) t.join();
    return 0;
}

import datetime
import threading
import time
import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


FNAME_OUT = "num_workers.txt"
FNAME_FLAG = "DASK_RUNNING"


def write_num_workers(fname, client, interval):
    with open(fname, "w") as f:
        while os.path.isfile("DASK_RUNNING"):
            # could also use client.nthreads
            f.write(f"{datetime.datetime.now()}, {len(client.scheduler_info()['workers'])}\n")
            f.flush()
            time.sleep(interval)


def start_tracking_workers(client, interval=1):
    pathlib.Path('DASK_RUNNING').touch()  # file that indicates worker tracking should run
    nworker_thread = threading.Thread(target=write_num_workers, args=("num_workers.txt", client, interval))
    nworker_thread.start()  # track workers in background
    return nworker_thread


def stop_tracking_workers():
    pathlib.Path('DASK_RUNNING').unlink()


def get_timestamps_and_counts():
    print("sleeping for two seconds before reading out file to ensure tracking is done")
    time.sleep(2)
    with open("num_workers.txt") as f:
        content = [l.strip().split(", ") for l in f.readlines()]

    timestamps = [datetime.datetime.strptime(c[0],  '%Y-%m-%d %H:%M:%S.%f') for c in content]
    nworkers = [int(c[1]) for c in content]

    delta_t = [(timestamps[i+1] - timestamps[i]).seconds for i in range(len(timestamps)-1)]
    workers_times_time = [nworkers[:-1][i] * delta_t[i] for i in range(len(timestamps)-1)]
    avg_num_workers = sum(workers_times_time) / (timestamps[-1] - timestamps[0]).seconds

    return timestamps, nworkers, avg_num_workers

def plot_worker_count(timestamps, nworkers, avg_num_workers, times_for_rates, instantaneous_rates):
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax1.plot(timestamps, nworkers, linewidth=2, color="C0")
    ax1.set_xlabel("time")
    ax1.set_ylabel("number of workers", color="C0")
    ax1.set_ylim([0, ax1.get_ylim()[1]*1.1])
    # ax1.set_title(f"worker count over time, average: {avg_num_workers:.1f}")
    ax1.set_title(f"worker count and data rate over time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.tick_params(axis="y", labelcolor="C0")
                              
    for tick in ax1.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
    
    ax2 = ax1.twinx()
    ax2.plot(times_for_rates, instantaneous_rates, marker="v", linewidth=0, color="C1")
    ax2.set_ylabel("data rate [Gbps]", color="C1")
    ax2.set_ylim([0, ax2.get_ylim()[1]*1.1])
    ax2.tick_params(axis="y", labelcolor="C1")
    fig.savefig("worker_count_data_rate_over_time.png")
"""
ljf_semaphore.py
Non-Preemptive Longest Job First (LJF) Scheduling with threading.Semaphore

NOTE: This program SIMULATES scheduling time only. It does NOT actually
allocate CPU execution time to the simulated tasks (i.e., it does not
call time.sleep() to emulate CPU bursts). Instead, it computes the
schedule mathematically and prints the result.
"""

import threading


# ----------------------------------------------------------------------
# Task definition
# ----------------------------------------------------------------------
class Task:
    def __init__(self, tid, arrival, burst):
        self.tid = tid            # Task ID (e.g., "P1")
        self.arrival = arrival    # Arrival time
        self.burst = burst        # CPU burst duration
        self.start = None         # Start time (filled during scheduling)
        self.completion = None    # Completion time
        self.turnaround = None    # Turnaround time
        self.waiting = None       # Waiting time

    def __repr__(self):
        return f"Task({self.tid}, arrival={self.arrival}, burst={self.burst})"


# ----------------------------------------------------------------------
# Shared ready queue protected by a Semaphore
# ----------------------------------------------------------------------
# A Semaphore(1) acts like a mutex: only one thread may hold it at a time.
# Every enqueue/dequeue/peek on `ready_queue` must occur inside the
# critical section guarded by `sem`.
ready_queue = []
sem = threading.Semaphore(1)


def enqueue(task):
    """Producer-side critical section: add a task to the ready queue."""
    with sem:  # acquire on entry, release on exit (context manager)
        ready_queue.append(task)


def dequeue_longest():
    """
    Consumer-side critical section: pick the task with the longest burst.
    Tie-break: earlier arrival first, then smaller task ID (deterministic).
    """
    with sem:
        if not ready_queue:
            return None
        # LJF selection: longest burst; ties -> earliest arrival; ties -> task ID
        chosen = max(
            ready_queue,
            key=lambda t: (t.burst, -t.arrival, )
        )
        # Apply deterministic tie-break if bursts and arrivals are equal
        candidates = [t for t in ready_queue if t.burst == chosen.burst
                      and t.arrival == chosen.arrival]
        if len(candidates) > 1:
            chosen = min(candidates, key=lambda t: t.tid)
        ready_queue.remove(chosen)
        return chosen


def snapshot():
    """Read-only snapshot of the ready queue (still inside critical section)."""
    with sem:
        return list(ready_queue)


# ----------------------------------------------------------------------
# LJF simulation
# ----------------------------------------------------------------------
def ljf_schedule(tasks):
    """
    Non-preemptive LJF scheduling over the given tasks.
    Returns the ordered list of tasks with start/completion filled in.
    """
    # Sort tasks by arrival so we can add them to the ready queue on time
    tasks_by_arrival = sorted(tasks, key=lambda t: (t.arrival, t.tid))

    current_time = 0
    completed = []
    idx = 0                # pointer into tasks_by_arrival

    print("\n" + "=" * 78)
    print("STEP-BY-STEP SCHEDULING TRACE (Non-Preemptive LJF)")
    print("=" * 78)

    while len(completed) < len(tasks):
        # 1) Admit all tasks that have arrived by `current_time`
        while idx < len(tasks_by_arrival) and tasks_by_arrival[idx].arrival <= current_time:
            enqueue(tasks_by_arrival[idx])
            idx += 1

        # 2) Snapshot ready queue for reporting
        ready_now = snapshot()
        ready_ids = ", ".join(t.tid for t in ready_now) if ready_now else "(empty)"

        # 3) Select the longest-burst task (critical section inside dequeue_longest)
        task = dequeue_longest()

        if task is None:
            # Nothing ready yet: advance time to next arrival
            current_time = tasks_by_arrival[idx].arrival
            continue

        # 4) Fill in schedule values
        task.start = current_time
        task.completion = current_time + task.burst
        task.turnaround = task.completion - task.arrival
        task.waiting = task.start - task.arrival

        # Reason string for selection
        reason = f"longest burst among ready tasks ({task.burst})"

        # 5) Print trace entry
        print(f"\n  Simulated time      : {current_time}")
        print(f"  Ready tasks         : {ready_ids}")
        print(f"  Selected task       : {task.tid}")
        print(f"  Reason              : {reason}")
        print(f"  Start time          : {task.start}")
        print(f"  CPU burst duration  : {task.burst}")
        print(f"  Completion time     : {task.completion}")

        # 6) Advance simulated time to completion
        current_time = task.completion
        completed.append(task)

    return completed


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------
def print_initial_table(tasks):
    print("=" * 78)
    print("PROGRAM TITLE  : Non-Preemptive Longest Job First (LJF) Scheduling")
    print("SYNCHRONIZATION: Semaphore (threading.Semaphore)")
    print("=" * 78)
    print("\nINITIAL TASK TABLE")
    print("-" * 40)
    print(f"{'Task':<8}{'Arrival Time':<15}{'CPU Burst':<12}")
    print("-" * 40)
    for t in tasks:
        print(f"{t.tid:<8}{t.arrival:<15}{t.burst:<12}")
    print("-" * 40)


def print_gantt(tasks):
    print("\n" + "=" * 78)
    print("GANTT CHART (proportional to burst, ASCII)")
    print("=" * 78)

    total_time = sum(t.burst for t in tasks)
    scale = 2  # 1 time unit -> 2 characters for readability

    # Timeline line
    line = "|"
    for t in tasks:
        block = t.tid.center(t.burst * scale)
        line += block + "|"
    print(line)

    # Time markers
    marker = f"{0:<{scale + 1}}"
    elapsed = 0
    for t in tasks:
        elapsed += t.burst
        marker += f"{elapsed:<{t.burst * scale + 1}}"
    print(marker.rstrip())

    # Labels under blocks (0–7, 7–12, 12–15)
    print("\nLabels:")
    start = 0
    for t in tasks:
        end = start + t.burst
        print(f"  {start}–{end}: {t.tid}")
        start = end


def print_results(tasks):
    print("\n" + "=" * 78)
    print("FINAL RESULTS TABLE")
    print("=" * 78)
    header = (f"{'Task':<6}{'Arrival':<10}{'Burst':<8}"
              f"{'Start':<8}{'Completion':<12}"
              f"{'Turnaround':<12}{'Waiting':<10}")
    print(header)
    print("-" * len(header))
    for t in tasks:
        print(f"{t.tid:<6}{t.arrival:<10}{t.burst:<8}"
              f"{t.start:<8}{t.completion:<12}"
              f"{t.turnaround:<12}{t.waiting:<10}")
    print("-" * len(header))

    n = len(tasks)
    total_wait = sum(t.waiting for t in tasks)
    total_turn = sum(t.turnaround for t in tasks)
    total_time = sum(t.burst for t in tasks)

    avg_wait = total_wait / n
    avg_turn = total_turn / n
    throughput = n / total_time

    print(f"\nAverage Waiting Time     : {avg_wait:.2f} time units")
    print(f"Average Turnaround Time  : {avg_turn:.2f} time units")
    print(f"Total Execution Time     : {total_time} time units")

    print("\nCPU Utilization")
    print("-" * 40)
    print("  Assumption: All three tasks arrive at t=0, so the CPU is")
    print("  never idle during the simulated interval.")
    print("  CPU Utilization = (Total Busy Time / Total Elapsed Time) * 100%")
    print(f"                  = ({total_time} / {total_time}) * 100% = 100.00%")
    print(f"Throughput       : {throughput:.4f} tasks per time unit "
          f"({n} tasks / {total_time} time units)")


def print_semaphore_explanation():
    print("\n" + "=" * 78)
    print("HOW THE SEMAPHORE PROTECTS THE SHARED READY QUEUE")
    print("=" * 78)
    print("""
  threading.Semaphore(1) behaves like a mutex. Only one thread can
  hold it at a time. All operations that read or modify `ready_queue`
  (enqueue, dequeue_longest, snapshot) are wrapped in `with sem:`,
  which acquires the semaphore on entry and releases it on exit —
  even if an exception occurs. This guarantees:

    1. No two operations inspect/modify the queue simultaneously.
    2. The LJF selection sees a consistent, unmodified queue.
    3. A task removed by dequeue_longest cannot also be seen/removed
       by another consumer (no duplicate scheduling).

  In this simulation the "producer" (admission of arrived tasks) and
  the "consumer" (LJF scheduler) run in the same thread, but the
  semaphore makes the code safe if these steps were moved to
  separate threads.
""")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    tasks = [
        Task("P1", arrival=0, burst=3),
        Task("P2", arrival=0, burst=7),
        Task("P3", arrival=0, burst=5),
    ]

    print_initial_table(tasks)

    scheduled = ljf_schedule(tasks)

    print_gantt(scheduled)
    print_results(scheduled)
    print_semaphore_explanation()


if __name__ == "__main__":
    main()
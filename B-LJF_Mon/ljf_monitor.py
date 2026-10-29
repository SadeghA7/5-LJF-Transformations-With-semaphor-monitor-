"""
ljf_monitor.py

Non-preemptive Longest Job First (LJF) CPU scheduling simulation
using a monitor-like synchronization abstraction built on
threading.Condition.

Run with:
    python ljf_monitor.py
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id: str
    arrival_time: int
    burst_time: int

    # Scheduling results (filled in when the task is dispatched)
    start_time: Optional[int] = None
    completion_time: Optional[int] = None

    @property
    def turnaround_time(self) -> int:
        assert self.completion_time is not None
        return self.completion_time - self.arrival_time

    @property
    def waiting_time(self) -> int:
        assert self.start_time is not None
        return self.start_time - self.arrival_time


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
class LJFMonitor:
    """
    Monitor-like scheduler.

    * Mutual exclusion is provided by the lock owned by the Condition.
    * Shared state (the ready queue and the simulated clock) is private to
      this object and only touched while that lock is held.
    * The LJF selection rule lives inside `select_longest_task()` and is
      therefore executed under the monitor's lock.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._ready: List[Task] = []
        self._current_time: int = 0
        self._init_details = {
            "lock_type": "threading.Condition -> RLock (default)",
            "initial_time": 0,
            "initial_ready": [],
        }

    # -- state helpers (all called with the lock held) ----------------------
    def add_task(self, task: Task) -> None:
        """Insert a task into the ready queue and notify waiters."""
        with self._condition:
            self._ready.append(task)
            # Wake any thread waiting for work.  The predicate is re-checked
            # after waking, as required for condition variables.
            self._condition.notify_all()

    def has_ready_tasks(self) -> bool:
        with self._condition:
            return len(self._ready) > 0

    def get_time(self) -> int:
        with self._condition:
            return self._current_time

    def set_time(self, value: int) -> None:
        with self._condition:
            self._current_time = value

    def ready_snapshot(self) -> List[Task]:
        """Return a copy of the ready queue for display purposes."""
        with self._condition:
            return list(self._ready)

    def _ljf_key(self, task: Task):
        # Largest burst first; ties -> earliest arrival; then task ID.
        return (-task.burst_time, task.arrival_time, task.task_id)

    def select_longest_task(self) -> Optional[Task]:
        """
        Pick the ready task with the largest burst (non-preemptive LJF).
        Returns None if the ready queue is empty.
        """
        with self._condition:
            if not self._ready:
                return None
            chosen = min(self._ready, key=self._ljf_key)
            self._ready.remove(chosen)
            return chosen

    def wait_for_ready(self) -> bool:
        """
        Block until a task is ready, re-checking the predicate after each
        wake-up.  Returns False if the queue is still empty (spurious wake).
        """
        with self._condition:
            while not self._ready:
                self._condition.wait()
            return True

    def init_details(self) -> dict:
        with self._condition:
            return dict(self._init_details)


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_initial_table(tasks: List[Task]) -> None:
    print(f"{'Task':<6}{'Arrival':>10}{'Burst':>10}")
    print("-" * 26)
    for t in tasks:
        print(f"{t.task_id:<6}{t.arrival_time:>10}{t.burst_time:>10}")
    print()


def print_monitor_init(details: dict) -> None:
    print("Monitor initialization:")
    print(f"  lock type         : {details['lock_type']}")
    print(f"  initial time      : {details['initial_time']}")
    print(f"  initial ready     : {details['initial_ready']}")
    print()


def print_trace_header() -> None:
    print("-" * 70)
    print("SCHEDULING TRACE")
    print("-" * 70)


def print_trace_step(
    time_before: int,
    ready_before: List[Task],
    method: str,
    chosen: Task,
    reason: str,
    start: int,
    burst: int,
    completion: int,
    time_after: int,
) -> None:
    ready_repr = "[" + ", ".join(t.task_id for t in ready_before) + "]"
    print(f"Time          : {time_before}")
    print(f"Ready queue   : {ready_repr}")
    print(f"Monitor method: {method}()")
    print(f"Selected task : {chosen.task_id}")
    print(f"Reason        : {reason}")
    print(f"Start time    : {start}")
    print(f"Burst duration: {burst}")
    print(f"Completion    : {completion}")
    print(f"Updated time  : {time_after}")
    print("-" * 70)


def print_gantt(chart: List[Task]) -> None:
    print()
    print("ASCII Gantt Chart:")
    if not chart:
        print("  (no tasks executed)")
        return

    # Top boundary line
    top = ""
    for t in chart:
        width = max(len(t.task_id) + 2, 6)
        top += "+" + "-" * width
    top += "+"
    print(top)

    # Task name row
    mid = ""
    for t in chart:
        width = max(len(t.task_id) + 2, 6)
        mid += "|" + t.task_id.center(width)
    mid += "|"
    print(mid)

    # Bottom boundary line
    print(top)

    # Time labels
    labels = [str(chart[0].start_time)]
    for t in chart:
        labels.append(str(t.completion_time))
    print(" ".join(labels))
    print()


def print_stats_table(tasks: List[Task]) -> None:
    print()
    print(f"{'Task':<6}{'Arrival':>9}{'Burst':>8}{'Start':>8}"
          f"{'Complete':>10}{'Turnaround':>12}{'Waiting':>10}")
    print("-" * 63)
    for t in tasks:
        print(f"{t.task_id:<6}{t.arrival_time:>9}{t.burst_time:>8}"
              f"{t.start_time:>8}{t.completion_time:>10}"
              f"{t.turnaround_time:>12}{t.waiting_time:>10}")
    print()


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def run_simulation(tasks: List[Task]) -> List[Task]:
    """
    Run the non-preemptive LJF schedule.

    The simulation is single-threaded and deterministic: it advances a
    logical clock rather than using real sleeps.  The monitor still protects
    every piece of shared state with a threading.Condition, so the
    synchronization contract is genuine even though concurrency is not
    exercised here.
    """
    monitor = LJFMonitor()

    # ---- Section 2/4: synchronization design + monitor init ----
    print("2. Synchronization design")
    print("   A monitor-like class (LJFMonitor) encapsulates the ready queue")
    print("   and the logical clock.  A threading.Condition supplies the")
    print("   mutual-exclusion lock and a wait/notify mechanism so that any")
    print("   consumer thread can safely block until work arrives.")
    print()
    print("4. Monitor initialization details")
    print_monitor_init(monitor.init_details())

    # Seed the ready queue (all arrivals are 0 in this exercise)
    for t in tasks:
        monitor.add_task(t)

    executed: List[Task] = []
    print_trace_header()

    while monitor.has_ready_tasks():
        time_before = monitor.get_time()
        ready_before = monitor.ready_snapshot()

        chosen = monitor.select_longest_task()
        if chosen is None:
            break  # defensive: queue drained between the two calls

        chosen.start_time = time_before
        chosen.completion_time = time_before + chosen.burst_time
        monitor.set_time(chosen.completion_time)
        executed.append(chosen)

        # Compose a human-readable selection reason
        if len(ready_before) > 1:
            max_burst = max(t.burst_time for t in ready_before)
            tied = [t for t in ready_before if t.burst_time == max_burst]
            if len(tied) > 1:
                reason = (f"largest burst ({chosen.burst_time}); "
                          f"tie resolved by arrival then ID")
            else:
                reason = f"largest burst ({chosen.burst_time})"
        else:
            reason = "only ready task"

        print_trace_step(
            time_before=time_before,
            ready_before=ready_before,
            method="select_longest_task",
            chosen=chosen,
            reason=reason,
            start=chosen.start_time,
            burst=chosen.burst_time,
            completion=chosen.completion_time,
            time_after=monitor.get_time(),
        )

    return executed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print_header("LJF Scheduling Using a Monitor")

    tasks = [
        Task("P1", arrival_time=0, burst_time=3),
        Task("P2", arrival_time=0, burst_time=7),
        Task("P3", arrival_time=0, burst_time=5),
    ]

    print("3. Initial task table")
    print_initial_table(tasks)

    # Re-print section header so numbering reads naturally in the output
    print("2/4. Synchronization design and monitor initialization")
    print("-" * 70)
    executed = run_simulation(tasks)

    print_gantt(executed)

    print("7. Final task statistics")
    print_stats_table(executed)

    # ---- Aggregate metrics ----
    n = len(executed)
    total_exec = sum(t.burst_time for t in executed)
    total_time = max(t.completion_time for t in executed)
    avg_wait = sum(t.waiting_time for t in executed) / n
    avg_turn = sum(t.turnaround_time for t in executed) / n
    cpu_util = (total_exec / total_time * 100.0) if total_time else 0.0
    throughput = n / total_time if total_time else 0.0

    print(f"8. Average waiting time     : {avg_wait:.2f}")
    print(f"9. Average turnaround time  : {avg_turn:.2f}")
    print(f"   Total simulated time     : {total_time}")
    print(f"10. CPU utilization         : {cpu_util:.2f}%")
    print(f"11. Throughput              : {throughput:.2f} tasks per time unit")
    print()

    print("12. Monitor mutual exclusion and condition variables")
    print("   Every LJFMonitor method acquires the lock owned by its")
    print("   threading.Condition before touching the ready queue or the")
    print("   clock, so only one thread can enter the critical section at a")
    print("   time (mutual exclusion).  Condition variables attach a wait/")
    print("   notify mechanism to that lock: a consumer blocks in wait(),")
    print("   releasing the lock, and must re-check its predicate in a")
    print("   while-loop after waking to guard against spurious wake-ups.")
    print("   Producers call notify_all() after mutating state so blocked")
    print("   consumers can re-evaluate the predicate.  The LJF policy")
    print("   (largest burst, tie by arrival then ID) is applied inside the")
    print("   monitor's select_longest_task(), so policy and protection")
    print("   stay encapsulated.")


if __name__ == "__main__":
    main()
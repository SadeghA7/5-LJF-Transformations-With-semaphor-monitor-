"""
ljf_hybrid.py

Non-preemptive Longest Job First (LJF) scheduling simulation using a
hybrid synchronization design:

  * LJFMonitor   : monitor-like scheduler built on threading.Condition.
  * EventBuffer  : bounded producer/consumer buffer built on semaphores
                   plus a plain mutex for the underlying list.

The simulation is single-threaded and deterministic.  It uses a logical
clock (no time.sleep), so it never claims to perform real parallel CPU
execution.  The completion-event buffer exercises the classic
semaphore-based producer/consumer protocol but does not alter scheduling
order or the computed metrics.

Run with:
    python ljf_hybrid.py
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id: str
    arrival_time: int
    burst_time: int
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
# Monitor-like scheduler
# ---------------------------------------------------------------------------
class LJFMonitor:
    """
    Monitor encapsulating the ready queue and the logical clock.

    Mutual exclusion comes from the lock owned by threading.Condition.
    The LJF policy lives in select_longest_task() and therefore runs
    inside the monitor's critical section.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._ready: List[Task] = []
        self._time: int = 0

    # -- insertion --------------------------------------------------------
    def add_task(self, task: Task) -> None:
        with self._cond:
            self._ready.append(task)
            # Wake any waiter; it must re-check the predicate.
            self._cond.notify_all()

    # -- inspection -------------------------------------------------------
    def ready_snapshot(self) -> List[Task]:
        with self._cond:
            return list(self._ready)

    def has_ready(self) -> bool:
        with self._cond:
            return len(self._ready) > 0

    def now(self) -> int:
        with self._cond:
            return self._time

    # -- state update -----------------------------------------------------
    def advance_to(self, t: int) -> None:
        with self._cond:
            self._time = t

    # -- LJF policy -------------------------------------------------------
    @staticmethod
    def _ljf_key(task: Task):
        # Longest burst first, then earliest arrival, then task id.
        return (-task.burst_time, task.arrival_time, task.task_id)

    def select_longest_task(self) -> Optional[Task]:
        with self._cond:
            if not self._ready:
                return None
            chosen = min(self._ready, key=self._ljf_key)
            self._ready.remove(chosen)
            return chosen

    def wait_until_ready(self) -> None:
        """Block until at least one task is ready; re-check on wake-up."""
        with self._cond:
            while not self._ready:
                self._cond.wait()


# ---------------------------------------------------------------------------
# Semaphore-based bounded event buffer
# ---------------------------------------------------------------------------
class EventBuffer:
    """
    Bounded producer/consumer buffer for completion events.

    Two counting semaphores gate the protocol:

        _slots  counts free buffer slots (initially capacity).
        _events counts buffered events     (initially 0).

    A plain mutex protects the list itself.  Semaphores are NEVER waited on
    while the monitor's Condition lock is held, keeping the two
    synchronization concerns independent.
    """

    def __init__(self, capacity: int = 3) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._slots = threading.Semaphore(capacity)  # free slots
        self._events = threading.Semaphore(0)        # available events
        self._mutex = threading.Lock()
        self._items: List[str] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    # -- producer side ----------------------------------------------------
    def produce(self, event: str) -> None:
        print(f"    [buffer] producer: acquire free slot   "
              f"(free before = {self._slots._value})")
        self._slots.acquire()
        with self._mutex:
            self._items.append(event)
            depth = len(self._items)
        print(f"    [buffer] producer: inserted {event!r}, "
              f"buffer depth = {depth}")
        self._events.release()
        print(f"    [buffer] producer: release event count "
              f"(events now = {self._events._value})")

    # -- consumer side ----------------------------------------------------
    def consume(self) -> str:
        print(f"    [buffer] consumer: acquire event count "
              f"(events before = {self._events._value})")
        self._events.acquire()
        with self._mutex:
            event = self._items.pop(0)
            depth = len(self._items)
        print(f"    [buffer] consumer: removed {event!r}, "
              f"buffer depth = {depth}")
        self._slots.release()
        print(f"    [buffer] consumer: release free slot   "
              f"(free now = {self._slots._value})")
        return event


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def rule(char: str = "=") -> None:
    print(char * 72)


def section(title: str) -> None:
    print()
    rule()
    print(title)
    rule()


def print_initial_table(tasks: List[Task]) -> None:
    print(f"{'Task':<6}{'Arrival':>9}{'Burst':>8}")
    print("-" * 23)
    for t in tasks:
        print(f"{t.task_id:<6}{t.arrival_time:>9}{t.burst_time:>8}")
    print()


def print_stats_table(tasks: List[Task]) -> None:
    print(f"{'Task':<6}{'Arrival':>9}{'Burst':>7}"
          f"{'Start':>7}{'Complete':>10}{'Turn':>7}{'Wait':>7}")
    print("-" * 53)
    for t in tasks:
        print(f"{t.task_id:<6}{t.arrival_time:>9}{t.burst_time:>7}"
              f"{t.start_time:>7}{t.completion_time:>10}"
              f"{t.turnaround_time:>7}{t.waiting_time:>7}")
    print()


def print_gantt(executed: List[Task]) -> None:
    print()
    if not executed:
        print("(no tasks executed)")
        return

    widths = [max(len(t.task_id) + 4, 8) for t in executed]
    bar = "".join("+" + "-" * w for w in widths) + "+"
    print(bar)
    print("".join("|" + t.task_id.center(w)
                  for t, w in zip(executed, widths)) + "|")
    print(bar)

    # Time markers: show the start of each slice and the final completion.
    boundary = f"{executed[0].start_time}"
    for t, w in zip(executed, widths):
        seg = str(t.completion_time)
        boundary += " " * (w - len(seg) + 1) + seg
    print(boundary)
    print()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(tasks: List[Task]) -> List[Task]:
    """Non-preemptive LJF with monitor scheduler + semaphore event buffer."""

    monitor = LJFMonitor()
    events = EventBuffer(capacity=3)
    executed: List[Task] = []

    # -- Section 5: monitor init + insertion trace -----------------------
    section("5. Monitor initialization and task insertion trace")
    print(f"  condition lock  : {type(monitor._cond).__name__}")
    print(f"  initial clock   : {monitor.now()}")
    print(f"  initial ready   : {monitor.ready_snapshot()}")
    print()
    for t in tasks:
        print(f"  add_task({t.task_id})  ->  "
              f"ready = {[x.task_id for x in monitor.ready_snapshot()] + [t.task_id]}")
        monitor.add_task(t)
    print()

    # -- Section 6: buffer init ------------------------------------------
    section("6. Semaphore and buffer initialization details")
    print(f"  capacity            : {events.capacity}")
    print(f"  free-slot semaphore : initial value = {events.capacity}")
    print(f"  event semaphore     : initial value = 0")
    print(f"  buffer mutex        : threading.Lock()")
    print(f"  buffer storage      : list (empty at start)")
    print()

    # -- Section 7: scheduling trace -------------------------------------
    section("7. Scheduling trace")
    step = 1
    while monitor.has_ready():
        t_now = monitor.now()
        ready = monitor.ready_snapshot()
        chosen = monitor.select_longest_task()
        if chosen is None:
            break

        # Explain selection
        if len(ready) == 1:
            reason = "only ready task"
        else:
            top_burst = max(x.burst_time for x in ready)
            tied = [x for x in ready if x.burst_time == top_burst]
            if len(tied) > 1:
                reason = (f"largest burst ({chosen.burst_time}); "
                          f"tie broken by arrival time / task id")
            else:
                reason = f"largest burst ({chosen.burst_time})"

        chosen.start_time = t_now
        chosen.completion_time = t_now + chosen.burst_time
        monitor.advance_to(chosen.completion_time)
        executed.append(chosen)

        print(f"  Step {step}: time = {t_now}")
        print(f"    ready before     : {[x.task_id for x in ready]}")
        print(f"    monitor method   : select_longest_task()")
        print(f"    selected         : {chosen.task_id}")
        print(f"    reason           : {reason}")
        print(f"    start / burst    : {chosen.start_time} / {chosen.burst_time}")
        print(f"    completion       : {chosen.completion_time}")
        print(f"    updated clock    : {monitor.now()}")
        print()
        step += 1

    # -- Section 8: event production / consumption -----------------------
    section("8. Completed-task event production and consumption trace")
    # Produce one event per completed task, then consume them all in FIFO
    # order.  This is sequential on purpose: the buffer is exercised, but
    # no real concurrency (or parallelism) is claimed.
    for t in executed:
        event = f"completed:{t.task_id}"
        print(f"  Produce event for {t.task_id}")
        events.produce(event)
        print()
    while True:
        # Drain the buffer with a non-blocking peek on the event semaphore.
        if events._events._value == 0:
            break
        got = events.consume()
        print(f"  Consumed event -> {got}")
        print()

    return executed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("LJF Scheduling Using a Hybrid Monitor + Semaphore Architecture")
    rule()

    # -- Section 2: architecture overview --------------------------------
    section("2. Hybrid architecture overview")
    print("  The scheduler is a monitor (LJFMonitor) built on")
    print("  threading.Condition: it owns the ready queue and the logical")
    print("  clock, and applies the LJF selection policy under its lock.")
    print("  A separate bounded EventBuffer, implemented with two counting")
    print("  semaphores plus a plain mutex, carries completion events.")
    print("  The two mechanisms never nest: the monitor's lock is released")
    print("  before any semaphore wait, so a blocked producer/consumer can")
    print("  never stall the scheduler or vice versa.")
    print()

    # -- Section 3: synchronization responsibilities ---------------------
    section("3. Synchronization responsibilities")
    print("  LJFMonitor   -> mutual exclusion + condition wait/notify;")
    print("                  protects the ready queue and simulated clock.")
    print("  EventBuffer  -> free-slot and available-event counting;")
    print("                  standard producer/consumer with a mutex guarding")
    print("                  the backing list.")
    print()

    # -- Section 4: initial task table -----------------------------------
    section("4. Initial task table")
    tasks = [
        Task("P1", 0, 3),
        Task("P2", 0, 7),
        Task("P3", 0, 5),
    ]
    print_initial_table(tasks)

    executed = run_simulation(tasks)

    # -- Section 9: Gantt chart ------------------------------------------
    section("9. ASCII Gantt chart")
    print_gantt(executed)

    # -- Section 10-15: statistics and metrics ---------------------------
    section("10. Final task statistics")
    print_stats_table(executed)

    n = len(executed)
    total_exec = sum(t.burst_time for t in executed)
    total_time = max(t.completion_time for t in executed)
    avg_wait = sum(t.waiting_time for t in executed) / n
    avg_turn = sum(t.turnaround_time for t in executed) / n
    cpu_util = total_exec / total_time * 100.0 if total_time else 0.0
    throughput = n / total_time if total_time else 0.0

    print(f"11. Average waiting time     : {avg_wait:.2f} time units")
    print(f"12. Average turnaround time  : {avg_turn:.2f} time units")
    print(f"13. Total simulated time     : {total_time} time units")
    print(f"14. CPU utilization          : {cpu_util:.2f}%")
    print( "    Assumptions              : no idle gaps, no scheduling")
    print( "                               overhead, no I/O, tasks are CPU")
    print( "                               bound; utilization = busy / total.")
    print(f"15. Throughput               : {throughput:.2f} tasks per time unit")
    print()

    # -- Section 16: role separation -------------------------------------
    section("16. Why the monitor and the semaphore have different roles")
    print("  The monitor is a mutual-exclusion construct for a critical")
    print("  section that holds state (the ready queue and clock).  Threads")
    print("  that enter it see a consistent view and can use condition")
    print("  wait/notify to block until the queue is non-empty.")
    print()
    print("  The semaphores implement a counting, bounded hand-off between")
    print("  producer and consumer.  Counting is not what conditions do")
    print("  well; conversely, semaphores are awkward as an all-purpose")
    print("  state guard.  Splitting responsibilities keeps each primitive")
    print("  doing what it does best, and because the monitor's lock is")
    print("  released before any semaphore wait, the two never nest.")
    print()
    print("  Note: this hybrid design is deliberately educational.  For")
    print("  three tasks in a single-threaded simulation, one primitive")
    print("  would suffice; the point is to show the division of labour.")


if __name__ == "__main__":
    main()